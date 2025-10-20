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

    def pretrain_green(self, num_epochs=1000, learning_rate=0.001, batch_size=32, stop_at_loss=0.16):
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
                green_positions = np.where(board_state == -1)
                
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

    def pretrain_selection(self, target_value, num_epochs=1000, learning_rate=0.001, batch_size=32, stop_at_loss=0.16):
        """
        Pre-train the network to select cells containing a specific value (1 or -1).
        Only convolutional layers are updated; fully connected layers are restored after training.
        
        Args:
            target_value: Either 1 or -1, the cell value to train the network to select
        """
        if target_value not in [1, 0, -1]:
            raise ValueError("target_value must be either 1 or -1")
        
        # Save original fully connected layer weights
        fc1_weight = self.fc1.weight.data.clone()
        fc1_bias = self.fc1.bias.data.clone() if self.fc1.bias is not None else None
        fc2_weight = self.fc2.weight.data.clone()
        fc2_bias = self.fc2.bias.data.clone() if self.fc2.bias is not None else None
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(batch_size):
                # Generate random board
                board_state = get_random_board(self.width, self.height)
                
                # Find all positions with target value
                target_positions = np.where(board_state == target_value)
                
                # Skip if no target positions on board
                if len(target_positions[0]) == 0:
                    continue
                
                # Convert to position indices
                target_indices = target_positions[0] * self.width + target_positions[1]
                
                # Create input tensor
                channel_0 = (board_state == -1).astype(np.float32)
                channel_1 = (board_state == 1).astype(np.float32)
                input_tensor = np.stack([channel_0, channel_1], axis=0)
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                
                # Forward pass
                energy_per_move = torch.rand(1, 1) * 1.5
                energy_per_creation = torch.rand(1, 1) * 1.0
                raw_output = self.forward(input_tensor, energy_per_move, energy_per_creation)
                selection_logits = raw_output[:, :self.selection_space]
                
                # Cross-entropy loss
                loss = F.cross_entropy(selection_logits, torch.tensor(target_indices[0:1]))
                
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
        
        # Restore original fully connected layer weights
        with torch.no_grad():
            self.fc1.weight.data.copy_(fc1_weight)
            if fc1_bias is not None:
                self.fc1.bias.data.copy_(fc1_bias)
            self.fc2.weight.data.copy_(fc2_weight)
            if fc2_bias is not None:
                self.fc2.bias.data.copy_(fc2_bias)
        
        print(f"Pre-training for target value {target_value} completed! FC layers restored.")
        return self

    def pretrain_red(self, num_epochs=1000, learning_rate=0.001, batch_size=32, stop_at_loss=0.16):
        """
        Pre-train the network to:
        - Select red pawns (1) when action != action_n
        - Select empty cells (0) when action == action_n
        """
        action_n = 8
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(batch_size):
                # Generate random board
                board_state = get_random_board(self.width, self.height)
                
                # Create input tensor
                channel_0 = (board_state == -1).astype(np.float32)
                channel_1 = (board_state == 1).astype(np.float32)
                input_tensor = np.stack([channel_0, channel_1], axis=0)
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                
                # Create energy tensor with random values in 0.1, 2
                energy_tensor = torch.rand(1, 1) * 1.9 + 0.1
                energy_tensor_2 = torch.rand(1, 1) * 1.0 + 0.1
                
                # Forward pass
                raw_output = self.forward(input_tensor, energy_tensor, energy_tensor_2)
                selection_logits = raw_output[:, :self.selection_space]
                action_logits = raw_output[:, self.selection_space:]

                # Choose a target action depending on action_logits produced by forward pass
                target_action = torch.argmax(action_logits, dim=1).item()
                
                # Determine what to select based on target action
                if target_action == action_n:
                    # Select empty cells
                    target_positions = np.where(board_state == 0)
                else:
                    # Select green pawns
                    target_positions = np.where(board_state == 1)
                
                # Skip if no valid positions
                if len(target_positions[0]) == 0:
                    continue
                
                # Convert to position indices
                target_indices = target_positions[0] * self.width + target_positions[1]
                
                # Loss for selection (choose any valid target)
                selection_loss = F.cross_entropy(selection_logits, torch.tensor([target_indices[0]]))
                
                # Loss for action (should predict target_action)
                action_loss = F.cross_entropy(action_logits, torch.tensor([target_action]))
                
                # Combined loss
                loss = selection_loss + action_loss
                
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
        
        print("Pre-training with action completed!")
        return self

    def pretrain_red_capture(self, num_epochs=1000, learning_rate=0.001, batch_size=32, stop_at_loss=0.16, board_size=0):
        """
        Pre-train the network to select cells containing green pawns (-1).
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        operation_offsets = [
            (0, -1),   # 0: Move left
            (0, 1),    # 1: Move right
            (-1, 0),   # 2: Move up
            (1, 0),    # 3: Move down
            (-1, -1),  # 4: Left-Up
            (-1, 1),   # 5: Right-Up
            (1, -1),   # 6: Left-Down
            (1, 1),    # 7: Right-Down
        ]
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(batch_size):
                # Generate random board
                board_state = get_random_board(self.width, self.height, board_size)

                # Chose distance
                distance = random.randint(1, 3)
                # Find a random position in the board
                x_random = np.random.randint(distance, self.height-distance-1)
                y_random = np.random.randint(distance, self.width-distance-1)
                # Set a red pawn at that position
                board_state[x_random, y_random] = 1
                # Chose direction
                direction = random.randint(0, 7)
                # Set a green pawn at that direction and distance
                n_row, y_column = x_random, y_random
                for i in range(1, distance):
                    n_row += operation_offsets[direction][0]
                    y_column += operation_offsets[direction][1]
                    board_state[n_row, y_column] = 0
                n_row += operation_offsets[direction][0]
                y_column += operation_offsets[direction][1]
                board_state[n_row, y_column] = -1

                # Create input tensor
                channel_0 = (board_state == -1).astype(np.float32)
                channel_1 = (board_state == 1).astype(np.float32)
                input_tensor = np.stack([channel_0, channel_1], axis=0)
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                
                # Forward pass
                raw_output = self.forward(input_tensor, torch.zeros(1, 1), torch.zeros(1, 1))
                selection_logits = raw_output[:, :self.selection_space]
                action_logits = raw_output[:, self.selection_space:]
                # Create target: one-hot encoding for piece selection and action:
                # * Piece selection: select the red pawn position
                # * Action: select the direction towards the green pawn
                red_pawn_index = x_random * self.width + y_random
                target_action = direction
                
                
                # Loss for selection (choose the red pawn position)
                selection_loss = F.cross_entropy(selection_logits, torch.tensor([red_pawn_index]))
                
                # Loss for action (should predict the direction towards green pawn)
                action_loss = F.cross_entropy(action_logits, torch.tensor([target_action]))
                
                # Combined loss
                loss = selection_loss + action_loss

                # Cross-entropy loss
                # loss = F.cross_entropy(selection_logits, torch.tensor(green_indices[0:1]))
                
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

def get_random_board(width, height, num_pieces=None):
    board = np.zeros((height, width), dtype=int)
    if num_pieces is None:
        num_pieces = int(random.random()*(80))
    positions = np.random.choice(width * height, num_pieces, replace=False)
    num_green = random.randint(0, num_pieces)
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
        print("Trying generation: ", i)
        print("Testing PlayerNN pre-training for red...")
        n_nn = PlayerNN.get_red_nn(WIDTH, HEIGHT).init_random()
        #n_nn.pretrain_selection(-1, num_epochs=900, learning_rate=0.001, batch_size=64, stop_at_loss=0.2)
        #n_nn.pretrain_selection(0, num_epochs=900, learning_rate=0.001, batch_size=64, stop_at_loss=0.2)
        n_nn.pretrain_red(num_epochs=1000, learning_rate=0.001, batch_size=64)
        n_nn.pretrain_red_capture(num_epochs=1000, learning_rate=0.001, batch_size=64, stop_at_loss=0.1, board_size=2)
        n_nn.pretrain_red_capture(num_epochs=1000, learning_rate=0.001, batch_size=64, stop_at_loss=0.1, board_size=4)
        n_nn.pretrain_red_capture(num_epochs=1000, learning_rate=0.001, batch_size=64, stop_at_loss=0.1, board_size=8)
        n_nn.save_weights(f"{DIR_NAME}/red_{i}.pth")
        print("Testing PlayerNN pre-training for green...")
        n_nn = PlayerNN.get_green_nn(WIDTH, HEIGHT).init_random()
        n_nn.pretrain_selection(1, num_epochs=900, learning_rate=0.001, batch_size=64, stop_at_loss=0.1)
        n_nn.pretrain_selection(0, num_epochs=900, learning_rate=0.001, batch_size=64, stop_at_loss=0.1)
        n_nn.pretrain_green(num_epochs=1000, learning_rate=0.001, batch_size=64, stop_at_loss=0.1)
        n_nn.save_weights(f"{DIR_NAME}/green_{i}.pth")

