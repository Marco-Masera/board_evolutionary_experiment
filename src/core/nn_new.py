from settings import SETTINGS
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
import torch.optim as optim

class PlayerNN(nn.Module):

    @staticmethod
    def get_green_nn(width, height):
        return PlayerNN(width, height, action_space=4)
    @staticmethod
    def get_red_nn(width, height):
        return PlayerNN(width, height, action_space=9)

    def __init__(self, width, height, action_space):
        super(PlayerNN, self).__init__()
        self.width = width
        self.height = height
        self.action_space = action_space
        self.selection_space = self.width * self.height

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, padding=1)
        # Add batch normalization for better training stability
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(6)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, height, width)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            conv_output_size = x.numel()
        
        fc_output_size = self.selection_space + self.action_space
        self.fc1 = nn.Linear(conv_output_size+2, fc_output_size)
        self.fc2 = nn.Linear(fc_output_size, fc_output_size)

        self.activation = nn.ReLU()

    def init_random(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return self

    def forward(self, x, energy_per_move, energy_per_creation):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, energy_per_move), dim=1)
        x = torch.cat((x, energy_per_creation), dim=1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def interpret_output(self, raw_output):
        # Split output into selection and action parts
        selection_logits = raw_output[:, :self.selection_space]  # Cell selection
        action_logits = raw_output[:, self.selection_space:]  # Action (Up, Down, Left, Right)
        # Select highest value indices
        selection_indices = torch.argmax(selection_logits, dim=1)
        action_indices = torch.argmax(action_logits, dim=1)
        return selection_indices.tolist()[0], action_indices.tolist()[0]

    def from_board_state(self, board_state, energy_per_move=0, energy_per_creation=0):
        # Ensure board_state is a numpy array
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state)
        
        # Check dimensions
        if board_state.shape != (self.height, self.width):
            raise ValueError(f"Board state shape {board_state.shape} doesn't match expected ({self.height}, {self.width})")
        
        # Create 2-channel input: 
        # Channel 0: 1s where board_state == -1
        # Channel 1: 1s where board_state == 1
        channel_0 = (board_state == -1).astype(np.float32)
        channel_1 = (board_state == 1).astype(np.float32)
        
        # Stack channels and add batch dimension
        input_tensor = np.stack([channel_0, channel_1], axis=0)  # Shape: (2, height, width)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)  # Shape: (1, 2, height, width)
        # Forward pass
        energy_per_move = torch.tensor([[energy_per_move]], dtype=torch.float32)
        energy_per_creation = torch.tensor([[energy_per_creation]], dtype=torch.float32)
        with torch.no_grad():
            raw_output = self.forward(input_tensor, energy_per_move, energy_per_creation)
        # Interpret output
        selection_indices, action_indices = self.interpret_output(raw_output)
        return selection_indices, action_indices

    def pretrain_selection(self, target_value, num_epochs=1000, learning_rate=0.001, batch_size=32, stop_at_loss=0.16):
        """
        Pre-train the network to select cells containing green pawns (-1).
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(batch_size):
                # Generate random board
                board_state = get_random_board(self.width, self.height)
                
                # Find all green pawn positions
                green_positions = np.where(board_state == target_value)
                
                # Skip if no green pawns on board
                if len(green_positions[0]) == 0:
                    continue
                
                # Convert to position indices
                green_indices = green_positions[0] * self.width + green_positions[1]
                
                # Create input tensor
                channel_0 = (board_state == -1).astype(np.float32)
                channel_1 = (board_state == 1).astype(np.float32)
                input_tensor = np.stack([channel_0, channel_1], axis=0)
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                
                # Forward pass
                raw_output = self.forward(input_tensor, torch.zeros(1, 1), torch.zeros(1, 1))
                selection_logits = raw_output[:, :self.selection_space]
                
                # Create target: one-hot encoding for any green position
                # Use soft target: distribute probability among all green positions
                target = torch.zeros_like(selection_logits)
                target[0, green_indices] = 1.0 / len(green_indices)
                
                # Cross-entropy loss
                loss = F.cross_entropy(selection_logits, torch.tensor(green_indices[0:1]))
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0 and (epoch + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
                if avg_loss <= stop_at_loss:
                    print("Stopping early due to reaching target loss.")
                    break
        
        print("Pre-training completed!")
        return self

    def pretrain_with_feedback(self, is_red, num_episodes=1000, learning_rate=0.001, batch_size=32):
        """
        Pre-train using reward-weighted cross-entropy.
        Collect decisions, get feedback, and update based on success/failure.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for episode in range(num_episodes):
            batch_boards = []
            batch_energies_move = []
            batch_energies_create = []
            batch_selections = []
            batch_actions = []
            batch_rewards = []
            
            # Collect batch of experiences
            for _ in range(batch_size):
                # Generate random scenario
                board_state = get_random_board(self.width, self.height)
                energy_move = torch.rand(1, 1) * 1.5
                energy_create = torch.rand(1, 1) * 1.0
                
                # Get network decision
                channel_0 = (board_state == -1).astype(np.float32)
                channel_1 = (board_state == 1).astype(np.float32)
                input_tensor = np.stack([channel_0, channel_1], axis=0)
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                
                raw_output = self.forward(input_tensor, energy_move, energy_create)
                selection_idx, action_idx = self.interpret_output(raw_output)
                
                # EVALUATE: Your custom logic here
                reward = evaluate_decision(is_red, board_state, selection_idx, action_idx)
                
                # Store experience
                batch_boards.append(input_tensor)
                batch_energies_move.append(energy_move)
                batch_energies_create.append(energy_create)
                batch_selections.append(selection_idx)
                batch_actions.append(action_idx)
                batch_rewards.append(reward)
            
            # Update network based on rewards
            if len(batch_boards) > 0:
                # Stack tensors
                boards_tensor = torch.cat(batch_boards, dim=0)
                energies_move_tensor = torch.cat(batch_energies_move, dim=0)
                energies_create_tensor = torch.cat(batch_energies_create, dim=0)
                selections_tensor = torch.tensor(batch_selections, dtype=torch.long)
                actions_tensor = torch.tensor(batch_actions, dtype=torch.long)
                rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                
                # Normalize rewards (optional but helpful)
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
                
                # Forward pass
                raw_output = self.forward(boards_tensor, energies_move_tensor, energies_create_tensor)
                selection_logits = raw_output[:, :self.selection_space]
                action_logits = raw_output[:, self.selection_space:]
                
                # Compute log probabilities
                selection_log_probs = F.log_softmax(selection_logits, dim=1)
                action_log_probs = F.log_softmax(action_logits, dim=1)
                
                # Get log probs of chosen actions
                chosen_selection_log_probs = selection_log_probs.gather(1, selections_tensor.unsqueeze(1)).squeeze()
                chosen_action_log_probs = action_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
                
                # Policy gradient loss (negative because we want to maximize reward)
                loss = -(chosen_selection_log_probs + chosen_action_log_probs) * rewards_tensor
                loss = loss.mean()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (episode + 1) % 100 == 0:
                    avg_reward = sum(batch_rewards) / len(batch_rewards)
                    print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, Loss: {loss.item():.4f}")
        
        print("Feedback-based pre-training completed!")
        return self

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set to evaluation mode after loading

    def get_mutation(self, alpha, param_noise_scale=0.5):
        mutated_net = copy.deepcopy(self)
        with torch.no_grad():
            for name, param in mutated_net.named_parameters():
                if 'conv' in name:
                    noise_scale = alpha * param_noise_scale 
                else:
                    noise_scale = alpha
                # Adaptive noise based on parameter magnitude
                noise = torch.randn_like(param) * noise_scale * param.abs().mean()
                param.add_(noise)
        return mutated_net


WIDTH = 16
HEIGHT = 8
import random
from settings import get_game_settings_for_red_first_training
from chessboard import Chessboard, GameStatistics
def board_avg_distance(board):
    """
    Computes the average Euclidean distance between all pawns of team -1 and all pawns of team 1.
    Returns 0 if there are no pawns of one or both teams.
    """
    # Find positions of -1 and 1 pawns
    pos_neg1 = np.argwhere(board == -1)
    pos_1 = np.argwhere(board == 1)
    
    if len(pos_neg1) == 0 or len(pos_1) == 0:
        return 0.0  # No pawns of one or both teams
    
    # Compute all pairwise distances
    dists = np.linalg.norm(pos_neg1[:, None, :] - pos_1[None, :, :], axis=2)
    avg_dist = dists.mean()
    return avg_dist

def evaluate_decision(is_red, board_state, selection_idx, action_idx):
    statistics = GameStatistics()
    chessboard = Chessboard(
        None, None, get_game_settings_for_red_first_training(0.5), statistics
    )
    chessboard.set_board_force(board_state)
    if is_red:
        chessboard.turn = 1
    else:
        chessboard.turn = 2

    chessboard.apply_step(selection_idx, action_idx)

    if is_red and statistics._red_pieces_captured > 0:
        return 1

    new_board_state = chessboard.board
    old_distance = board_avg_distance(board_state)
    new_distance = board_avg_distance(new_board_state)
    if is_red:
        return 1 if new_distance < old_distance else -1
    else:
        if new_distance > old_distance:
            return 1
        elif new_distance < old_distance:
            return -1
        else:
            return 0


def get_random_board(width, height, num_pieces=None):
    board = np.zeros((height, width), dtype=int)
    if num_pieces is None:
        num_pieces = max(2, int(random.random()*(80)))
    positions = np.random.choice(width * height, num_pieces, replace=False)
    num_green = random.randint(1, num_pieces-1)
    for i in range(num_green):
        row = positions[i] // width
        col = positions[i] % width
        board[row, col] = -1  # Green piece
    for i in range(num_green, num_pieces):
        row = positions[i] // width
        col = positions[i] % width
        board[row, col] = 1  # Red piece
    return board


DIR_NAME = "../../pretrained"
import sys
import os
if __name__ == "__main__":
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)

    args = sys.argv[1:]
    if len(args) < 1:
        raise Exception("Please provide N as argument to specify how many networks to pre-train.")
    N = int(args[0])

    for i in range(N):
        print("Pre training generation: ", i)
        
        print(">Red individual:")
        n_nn = PlayerNN.get_red_nn(WIDTH, HEIGHT).init_random()
        print("  Pre training for selection of items")
        n_nn.pretrain_selection(-1, num_epochs=700, learning_rate=0.001, batch_size=64, stop_at_loss=0.3)
        n_nn.pretrain_selection(0, num_epochs=700, learning_rate=0.001, batch_size=64, stop_at_loss=0.3)
        n_nn.pretrain_selection(1, num_epochs=1000, learning_rate=0.001, batch_size=64, stop_at_loss=0.1)
        print("  Pre training with reinforcment learning")
        n_nn.pretrain_with_feedback(True, num_episodes=1000, learning_rate=0.001, batch_size=32)

        n_nn.save_weights(f"{DIR_NAME}/red_{i}.pth")
        
        print(">Green individual:")
        n_nn.pretrain_selection(1, num_epochs=700, learning_rate=0.001, batch_size=64, stop_at_loss=0.3)
        n_nn.pretrain_selection(0, num_epochs=700, learning_rate=0.001, batch_size=64, stop_at_loss=0.3)
        n_nn.pretrain_selection(-1, num_epochs=1000, learning_rate=0.001, batch_size=64, stop_at_loss=0.1)
        print("  Pre training with reinforcment learning")
        n_nn.pretrain_with_feedback(False, num_episodes=1000, learning_rate=0.001, batch_size=32)
        n_nn.save_weights(f"{DIR_NAME}/green_{i}.pth")

