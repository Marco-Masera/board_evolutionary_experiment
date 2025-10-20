from settings import SETTINGS
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
import torch.optim as optim


class RedNN(nn.Module):
    def __init__(self, W, H):
        super(RedNN, self).__init__()
        
        # Placeholder values for input dimensions
        self.H = H
        self.W = W
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2)  # 1x1 per-cell mixing
        
        # Calculate flattened size dynamically by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.W, self.H)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flattened_size = x.view(1, -1).shape[1]
        
        # Richer energy embedding
        self.energy_embed = nn.Sequential(
            nn.Linear(1, 6),
            nn.Tanh(),
            nn.Linear(6, 6)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(self.flattened_size + 6, self.H * self.W + 9)
        self.fc2 = nn.Linear(self.H * self.W + 9, self.H * self.W + 9)
        
        # Activation function (tanh keeps values in [-1, 1])
        self.activation = nn.Tanh()
        
        # For backpropagation
        self.last_board_state = None
        self.last_energy = None
        self.last_position_idx = None
        self.last_operation_idx = None
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        
        # Experience buffer for batch training
        self.experience_buffer = []
    
    def forward(self, x, energy):
        # Input shape: (batch_size, H, W) or (batch_size, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, H, W)
        
        # Apply convolutional layers with tanh activation
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        # Compute energy input
        energy = self.energy_embed(energy.unsqueeze(1))
        # Concatenate energy embedding
        x = torch.cat((x, energy), dim=1)
        
        # Fully connected layer with tanh activation
        x = self.activation(self.fc(x))
        x = self.activation(self.fc2(x))
        return x
    
    def init_random(self):
        """
        Randomly initialize all network weights.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return self
    
    def interpret_output(self, raw_output):
        """
        Transform raw network output to meaningful selections.
        
        Args:
            raw_output: Tensor of shape (batch_size, H*W + 9)
        
        Returns:
            tuple: (position_indices, operation_indices)
                - position_indices: Tensor of shape (batch_size,) with values in [0, H*W-1]
                - operation_indices: Tensor of shape (batch_size,) with values in [0, 8]
        """
        input_size = self.H * self.W
        
        # Split output into position and operation parts
        position_logits = raw_output[:, :input_size]  # First INPUT_SIZE values
        operation_logits = raw_output[:, input_size:]  # Last 9 values
        
        # Select highest value indices
        position_indices = torch.argmax(position_logits, dim=1)
        operation_indices = torch.argmax(operation_logits, dim=1)
        
        return position_indices, operation_indices
    
    def save_weights(self, filepath):
        """
        Save network weights to a file.
        
        Args:
            filepath: Path where to save the model weights
        """
        torch.save(self.state_dict(), filepath)
    
    def load_weights(self, filepath):
        """
        Load network weights from a file.
        
        Args:
            filepath: Path to the saved model weights
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set to evaluation mode after loading
    
    def predict(self, board_state, energy):
        """
        Run the network on input and return interpreted results.
        
        Args:
            board_state: Numpy array of shape (H, W) with values in [-1, 1]
            energy: Numpy array or float representing energy value
        
        Returns:
            tuple: (position_index, operation_index)
        """
        # Convert numpy arrays to tensors
        x = torch.from_numpy(board_state).float().unsqueeze(0)  # Add batch dimension
        if isinstance(energy, np.ndarray):
            e = torch.from_numpy(energy).float().unsqueeze(0)
        else:
            e = torch.tensor([energy], dtype=torch.float32)
        
        # Run forward pass
        with torch.no_grad():
            raw_output = self.forward(x, e)
        
        # Interpret output
        position_indices, operation_indices = self.interpret_output(raw_output)
        
        # Store for backpropagation
        self.last_board_state = board_state
        self.last_energy = energy
        self.last_position_idx = position_indices.item()
        self.last_operation_idx = operation_indices.item()
        
        return self.last_position_idx, self.last_operation_idx
    
    def add_experience(self, board_state, energy, position_idx, operation_idx, reward):
        """
        Add an experience to the buffer for batch training.
        
        Args:
            board_state: Numpy array of shape (H, W)
            energy: Float or numpy array
            position_idx: Integer index of chosen position
            operation_idx: Integer index of chosen operation
            reward: Float reward value
        """
        self.experience_buffer.append({
            'board_state': board_state.copy(),
            'energy': energy if isinstance(energy, (int, float)) else energy.copy(),
            'position_idx': position_idx,
            'operation_idx': operation_idx,
            'reward': reward
        })
    
    def train_batch(self):
        """
        Train on accumulated experiences in the buffer.
        Clears the buffer after training.
        """
        if len(self.experience_buffer) == 0:
            return
        
        self.train()  # Set to training mode
        
        # Prepare batch data
        board_states = []
        energies = []
        position_indices = []
        operation_indices = []
        rewards = []
        
        for exp in self.experience_buffer:
            board_states.append(exp['board_state'])
            energies.append(exp['energy'])
            position_indices.append(exp['position_idx'])
            operation_indices.append(exp['operation_idx'])
            rewards.append(exp['reward'])
        
        # Convert to tensors
        x = torch.from_numpy(np.array(board_states)).float()
        e = torch.tensor(energies, dtype=torch.float32)
        position_indices = torch.tensor(position_indices, dtype=torch.long)
        operation_indices = torch.tensor(operation_indices, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Forward pass
        raw_output = self.forward(x, e)
        
        input_size = self.H * self.W
        position_logits = raw_output[:, :input_size]
        operation_logits = raw_output[:, input_size:]
        
        # Get the log probabilities
        position_log_probs = F.log_softmax(position_logits, dim=1)
        operation_log_probs = F.log_softmax(operation_logits, dim=1)
        
        # Gather log probabilities for chosen actions
        chosen_position_log_probs = position_log_probs.gather(1, position_indices.unsqueeze(1)).squeeze(1)
        chosen_operation_log_probs = operation_log_probs.gather(1, operation_indices.unsqueeze(1)).squeeze(1)
        
        # Policy gradient loss: -log(prob) * reward, averaged over batch
        loss = -torch.mean(rewards * (chosen_position_log_probs + chosen_operation_log_probs))
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffer
        self.experience_buffer = []
        
        self.eval()  # Set back to evaluation mode
    
    def backpropagate(self, reward):
        """
        Simple backpropagation for pre-training using PyTorch.
        reward: 1 for success, -1 for failure
        """
        if self.last_board_state is None:
            return
        
        self.train()  # Set to training mode
        
        # Convert to tensors
        x = torch.from_numpy(self.last_board_state).float().unsqueeze(0)
        if isinstance(self.last_energy, np.ndarray):
            e = torch.from_numpy(self.last_energy).float().unsqueeze(0)
        else:
            e = torch.tensor([self.last_energy], dtype=torch.float32)
        
        # Forward pass
        raw_output = self.forward(x, e)
        
        input_size = self.H * self.W
        position_logits = raw_output[:, :input_size]
        operation_logits = raw_output[:, input_size:]
        
        # Get the log probabilities for the actions taken
        position_log_probs = F.log_softmax(position_logits, dim=1)
        operation_log_probs = F.log_softmax(operation_logits, dim=1)
        
        # Get log probability of the chosen actions
        chosen_position_log_prob = position_log_probs[0, self.last_position_idx]
        chosen_operation_log_prob = operation_log_probs[0, self.last_operation_idx]
        
        # Policy gradient loss: -log(prob) * reward
        # If reward is positive, we want to increase probability (minimize -log prob)
        # If reward is negative, we want to decrease probability (maximize -log prob)
        loss = -reward * (chosen_position_log_prob + chosen_operation_log_prob)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.eval()  # Set back to evaluation mode
    
    def get_mutation(self, alpha):
        """
        Returns a copy of the network with mutated weights.
        
        Args:
            alpha: Mutation strength (standard deviation of Gaussian noise)
                   Recommended values: 0.01-0.1 for evolutionary optimization
                   - 0.01: Small mutations, fine-tuning
                   - 0.05: Medium mutations, balanced exploration
                   - 0.1: Large mutations, aggressive exploration
        
        Returns:
            RedNN: A new network instance with mutated weights
        """
        # Create a deep copy of the network
        mutated_net = copy.deepcopy(self)
        
        # Mutate all parameters (weights and biases)
        with torch.no_grad():
            for param in mutated_net.parameters():
                # Add Gaussian noise scaled by alpha
                noise = torch.randn_like(param) * alpha
                param.add_(noise)
        
        return mutated_net

class GreenNN(nn.Module):
    def __init__(self, W, H):
        super(GreenNN, self).__init__()
        
        # Placeholder values for input dimensions
        self.H = H
        self.W = W
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2)  # 1x1 per-cell mixing
        
        # Calculate flattened size dynamically by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.H, self.W)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flattened_size = x.view(1, -1).shape[1]
        

        # Fully connected layer
        self.fc = nn.Linear(self.flattened_size, self.H * self.W + 4)
        self.fc2 = nn.Linear(self.H * self.W + 4, self.H * self.W + 4)
        
        # Activation function (tanh keeps values in [-1, 1])
        self.activation = nn.Tanh()
        
        # For backpropagation
        self.last_board_state = None
        self.last_position_idx = None
        self.last_operation_idx = None
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        
        # Experience buffer for batch training
        self.experience_buffer = []
    
    def forward(self, x):
        # Input shape: (batch_size, H, W) or (batch_size, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, H, W)
        
        # Apply convolutional layers with tanh activation
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer with tanh activation
        x = self.activation(self.fc(x))
        x = self.activation(self.fc2(x))
        
        return x
    
    def init_random(self):
        """
        Randomly initialize all network weights.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return self
    
    def interpret_output(self, raw_output):
        """
        Transform raw network output to meaningful selections.
        
        Args:
            raw_output: Tensor of shape (batch_size, H*W + 4)
        
        Returns:
            tuple: (position_indices, operation_indices)
                - position_indices: Tensor of shape (batch_size,) with values in [0, H*W-1]
                - operation_indices: Tensor of shape (batch_size,) with values in [0, 3]
        """
        input_size = self.H * self.W
        
        # Split output into position and operation parts
        position_logits = raw_output[:, :input_size]  # First INPUT_SIZE values
        operation_logits = raw_output[:, input_size:]  # Last 4 values
        
        # Select highest value indices
        position_indices = torch.argmax(position_logits, dim=1)
        operation_indices = torch.argmax(operation_logits, dim=1)
        
        return position_indices, operation_indices
    
    def save_weights(self, filepath):
        """
        Save network weights to a file.
        
        Args:
            filepath: Path where to save the model weights
        """
        torch.save(self.state_dict(), filepath)
    
    def load_weights(self, filepath):
        """
        Load network weights from a file.
        
        Args:
            filepath: Path to the saved model weights
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set to evaluation mode after loading
    
    def predict(self, board_state):
        """
        Run the network on input and return interpreted results.
        
        Args:
            board_state: Numpy array of shape (H, W) with values in [-1, 1]
        
        Returns:
            tuple: (position_index, operation_index)
        """
        # Convert numpy array to tensor
        x = torch.from_numpy(board_state).float().unsqueeze(0)  # Add batch dimension
        
        # Run forward pass
        with torch.no_grad():
            raw_output = self.forward(x)
        
        # Interpret output
        position_indices, operation_indices = self.interpret_output(raw_output)
        
        # Store for backpropagation
        self.last_board_state = board_state
        self.last_position_idx = position_indices.item()
        self.last_operation_idx = operation_indices.item()
        
        return self.last_position_idx, self.last_operation_idx
    
    def add_experience(self, board_state, position_idx, operation_idx, reward):
        """
        Add an experience to the buffer for batch training.
        
        Args:
            board_state: Numpy array of shape (H, W)
            position_idx: Integer index of chosen position
            operation_idx: Integer index of chosen operation
            reward: Float reward value
        """
        self.experience_buffer.append({
            'board_state': board_state.copy(),
            'position_idx': position_idx,
            'operation_idx': operation_idx,
            'reward': reward
        })
    
    def train_batch(self):
        """
        Train on accumulated experiences in the buffer.
        Clears the buffer after training.
        """
        if len(self.experience_buffer) == 0:
            return
        
        self.train()  # Set to training mode
        
        # Prepare batch data
        board_states = []
        position_indices = []
        operation_indices = []
        rewards = []
        
        for exp in self.experience_buffer:
            board_states.append(exp['board_state'])
            position_indices.append(exp['position_idx'])
            operation_indices.append(exp['operation_idx'])
            rewards.append(exp['reward'])
        
        # Convert to tensors
        x = torch.from_numpy(np.array(board_states)).float()
        position_indices = torch.tensor(position_indices, dtype=torch.long)
        operation_indices = torch.tensor(operation_indices, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Forward pass
        raw_output = self.forward(x)
        
        input_size = self.H * self.W
        position_logits = raw_output[:, :input_size]
        operation_logits = raw_output[:, input_size:]
        
        # Get the log probabilities
        position_log_probs = F.log_softmax(position_logits, dim=1)
        operation_log_probs = F.log_softmax(operation_logits, dim=1)
        
        # Gather log probabilities for chosen actions
        chosen_position_log_probs = position_log_probs.gather(1, position_indices.unsqueeze(1)).squeeze(1)
        chosen_operation_log_probs = operation_log_probs.gather(1, operation_indices.unsqueeze(1)).squeeze(1)
        
        # Policy gradient loss: -log(prob) * reward, averaged over batch
        loss = -torch.mean(rewards * (chosen_position_log_probs + chosen_operation_log_probs))
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffer
        self.experience_buffer = []
        
        self.eval()  # Set back to evaluation mode
    
    def backpropagate(self, reward):
        """
        Simple backpropagation for pre-training using PyTorch.
        reward: 1 for success, -1 for failure
        """
        if self.last_board_state is None:
            return
        
        self.train()  # Set to training mode
        
        # Convert to tensor
        x = torch.from_numpy(self.last_board_state).float().unsqueeze(0)
        
        # Forward pass
        raw_output = self.forward(x)
        
        input_size = self.H * self.W
        position_logits = raw_output[:, :input_size]
        operation_logits = raw_output[:, input_size:]
        
        # Get the log probabilities for the actions taken
        position_log_probs = F.log_softmax(position_logits, dim=1)
        operation_log_probs = F.log_softmax(operation_logits, dim=1)
        
        # Get log probability of the chosen actions
        chosen_position_log_prob = position_log_probs[0, self.last_position_idx]
        chosen_operation_log_prob = operation_log_probs[0, self.last_operation_idx]
        
        # Policy gradient loss: -log(prob) * reward
        loss = -reward * (chosen_position_log_prob + chosen_operation_log_prob)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.eval()  # Set back to evaluation mode
    
    def get_mutation(self, alpha):
        """
        Returns a copy of the network with mutated weights.
        
        Args:
            alpha: Mutation strength (standard deviation of Gaussian noise)
                   Recommended values: 0.01-0.1 for evolutionary optimization
                   - 0.01: Small mutations, fine-tuning
                   - 0.05: Medium mutations, balanced exploration
                   - 0.1: Large mutations, aggressive exploration
        
        Returns:
            GreenNN: A new network instance with mutated weights
        """
        # Create a deep copy of the network
        mutated_net = copy.deepcopy(self)
        
        # Mutate all parameters (weights and biases)
        with torch.no_grad():
            for param in mutated_net.parameters():
                # Add Gaussian noise scaled by alpha
                noise = torch.randn_like(param) * alpha
                param.add_(noise)
        
        return mutated_net

