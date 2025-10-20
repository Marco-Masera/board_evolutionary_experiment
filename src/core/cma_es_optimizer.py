from settings import SETTINGS, get_game_settings
#from nn import GreenNN, RedNN
from nn_new import PlayerNN
import numpy as np
from chessboard import Game, GameStatistics
import time
import signal
import os
import sys
import random
import torch
import torch.nn as nn
from typing import Callable, Tuple, List


class CMAES:
    """CMA-ES optimizer for neural network weights."""
    
    def __init__(self, dimension: int, population_size: int = None, sigma: float = 0.5):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            dimension: Number of parameters to optimize
            population_size: Population size (default: 4 + floor(3*log(dimension)))
            sigma: Initial step size
        """
        self.dim = dimension
        self.sigma = sigma
        
        # Population size
        self.lam = population_size if population_size else int(4 + np.floor(3 * np.log(dimension)))
        self.mu = self.lam // 2  # Number of parents
        
        # Weights for recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        
        # Adaptation parameters
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / 
                      ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Dynamic strategy parameters
        self.mean = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invC = np.eye(self.dim)
        
        # Expectation of ||N(0,I)||
        self.chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        self.generation = 0
        
    def ask(self) -> np.ndarray:
        """Generate population of candidate solutions."""
        population = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y
            population.append(x)
        return np.array(population)
    
    def tell(self, population: np.ndarray, fitness: np.ndarray):
        """Update distribution based on fitness values.
        
        Args:
            population: Array of shape (population_size, dimension)
            fitness: Array of fitness values (higher is better)
        """
        # Sort by fitness (descending)
        idx = np.argsort(fitness)[::-1]
        population = population[idx]
        
        # Select top mu individuals
        selected = population[:self.mu]
        
        # Recombination: new mean
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected
        
        # Cumulation: update evolution paths
        c_diff = self.mean - old_mean
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) / self.sigma * \
                  (self.invC @ c_diff)
        
        hsig = (np.linalg.norm(self.ps) / 
                np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chi_n 
                < 1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) / self.sigma * c_diff
        
        # Adapt covariance matrix
        artmp = (selected - old_mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + 
                           (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)
        
        # Adapt step size
        self.sigma *= np.exp((self.cs / self.damps) * 
                            (np.linalg.norm(self.ps) / self.chi_n - 1))
        
        # Update B, D from C
        if self.generation % (1 / (self.c1 + self.cmu) / self.dim / 10) < 1:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Enforce symmetry
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 1e-10))
            self.invC = self.B @ np.diag(1 / self.D) @ self.B.T
        
        self.generation += 1


class DualCMAES:
    """Manages CMA-ES for two populations (Red and Green teams)."""
    
    def __init__(self, red_nn: nn.Module, green_nn: nn.Module, 
                 red_pop_size: int = None, green_pop_size: int = None,
                 sigma: float = 0.5):
        """
        Initialize dual CMA-ES optimizer.
        
        Args:
            red_nn: Template neural network for red team
            green_nn: Template neural network for green team
            red_pop_size: Red team population size
            green_pop_size: Green team population size
            sigma: Initial step size for both populations
        """
        self.red_nn_template = red_nn
        self.green_nn_template = green_nn
        
        # Get parameter counts
        self.red_dim = sum(p.numel() for p in red_nn.parameters())
        self.green_dim = sum(p.numel() for p in green_nn.parameters())
        
        # Initialize CMA-ES for both populations
        self.red_cmaes = CMAES(self.red_dim, red_pop_size, sigma)
        self.green_cmaes = CMAES(self.green_dim, green_pop_size, sigma)
        
        # Store current populations
        self.red_population = None
        self.green_population = None
        
    def nn_to_weights(self, nn: nn.Module) -> np.ndarray:
        """Extract weights from neural network as flat array."""
        weights = []
        for param in nn.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def weights_to_nn(self, weights: np.ndarray, nn: nn.Module) -> nn.Module:
        """Load weights into neural network."""
        nn_copy = type(nn)(*self._get_nn_init_args(nn)).to(next(nn.parameters()).device)
        nn_copy.load_state_dict(nn.state_dict())
        
        idx = 0
        for param in nn_copy.parameters():
            param_size = param.numel()
            param.data = torch.from_numpy(
                weights[idx:idx + param_size].reshape(param.shape)
            ).float().to(param.device)
            idx += param_size
        
        return nn_copy
    
    def _get_nn_init_args(self, nn: nn.Module):
        """Helper to get init args (override if needed for complex NNs)."""
        return ()
    
    def ask(self) -> Tuple[List[nn.Module], List[nn.Module]]:
        """
        Generate populations of neural networks.
        
        Returns:
            Tuple of (red_nns, green_nns) where each is a list of neural networks
        """
        # Generate weight populations
        self.red_population = self.red_cmaes.ask()
        self.green_population = self.green_cmaes.ask()
        
        # Convert to neural networks
        red_nns = [self.weights_to_nn(w, self.red_nn_template) 
                   for w in self.red_population]
        green_nns = [self.weights_to_nn(w, self.green_nn_template) 
                     for w in self.green_population]
        
        return red_nns, green_nns
    
    def tell(self, red_fitness: np.ndarray, green_fitness: np.ndarray):
        """
        Update both populations based on fitness values.
        
        Args:
            red_fitness: Fitness values for red team (higher is better)
            green_fitness: Fitness values for green team (higher is better)
        """
        self.red_cmaes.tell(self.red_population, red_fitness)
        self.green_cmaes.tell(self.green_population, green_fitness)
    
    def get_best(self) -> Tuple[nn.Module, nn.Module]:
        """
        Get the current best neural networks from each population.
        
        Returns:
            Tuple of (best_red_nn, best_green_nn)
        """
        best_red = self.weights_to_nn(self.red_cmaes.mean, self.red_nn_template)
        best_green = self.weights_to_nn(self.green_cmaes.mean, self.green_nn_template)
        return best_red, best_green
    
    def save_weights(self, red_path: str, green_path: str):
        """Save current mean weights to files."""
        np.save(red_path, self.red_cmaes.mean)
        np.save(green_path, self.green_cmaes.mean)
    
    def load_weights(self, red_path: str, green_path: str):
        """Load mean weights from files."""
        self.red_cmaes.mean = np.load(red_path)
        self.green_cmaes.mean = np.load(green_path)


def _play_and_score(green, red, statistics):
        # Returns: Win scores, Secondary scores for Green. Red is -1*Green
        # Lower is better
        settings = get_game_settings(0.5)
        game = Game(green, red, settings, statistics)
        result = game.play()
        eaten = statistics.get_pieces_eaten_last_match()
        invalid_moves_red = statistics.get_invalid_move_last_match()
        if (result == "green"):
            primary = -1
            secondary = eaten # If green wins, the secondary score is number of pieces it got eaten
        elif (result == "red"):
            primary = 1
            secondary = -delta_time # If green loses, secondary score is time: longer is better
        else:
            raise Exception("Unexpected return value from game.play()")
        return primary, secondary, invalid_moves_red

# Example usage
if __name__ == "__main__":
    # Define simple NNs
    class SimpleNN(nn.Module):
        def __init__(self, input_size=10, hidden_size=20, output_size=5):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # Initialize
    CHECKPOINT_DIR = "../../v1"
    filepath = os.path.join(CHECKPOINT_DIR, f"green_0.pth")
    green_nn = PlayerNN.get_green_nn(16, 8)
    green_nn.load_weights(filepath)
    filepath = os.path.join(CHECKPOINT_DIR, f"red_0.pth")
    red_nn = PlayerNN.get_red_nn(16, 8)
    red_nn.load_weights(filepath)
    
    optimizer = DualCMAES(red_nn, green_nn, red_pop_size=20, green_pop_size=20)

    MAX_SECONDARY = 40
    MAX_ERRORS_NORM_FACTOR = 30000.0
    MAX_ERRORS_NORM = (100000.0) / MAX_ERRORS_NORM_FACTOR
    
    # Training loop
    for generation in range(1000):
        # Get populations
        red_nns, green_nns = optimizer.ask()
        
        # Compute fitness (example: each red plays against each green)
        red_fitness = np.zeros(len(red_nns))
        green_fitness = np.zeros(len(green_nns))
        
        statistics = GameStatistics()
        for i, red in enumerate(red_nns):
            for j, green in enumerate(green_nns):
                primary, secondary, invalid_red = _play_and_score(green, red, statistics)
                
                green_score = (primary * MAX_SECONDARY * 400) + secondary
                red_score = (-primary * MAX_SECONDARY * 400) - secondary
                red_score = (red_score * MAX_ERRORS) + invalid_red/MAX_ERRORS_NORM_FACTOR

                red_fitness[i] += red_score
                green_fitness[j] += green_score
        statistics.print_statistics()
        # Normalize fitness
        red_fitness /= len(green_nns)
        green_fitness /= len(red_nns)
        
        # Update populations
        optimizer.tell(red_fitness, green_fitness)
        
        print(f"Generation {generation}: "
              f"Red avg fitness: {red_fitness.mean():.3f}, "
              f"Green avg fitness: {green_fitness.mean():.3f}")
    
    # Get best networks
    best_red, best_green = optimizer.get_best()
    print("Training complete!")