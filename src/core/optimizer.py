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

POP_SIZE = 20
NUM_COMP = 10
CHECKPOINT_DIR = None
ELITISM = True

class Optimizer:
    def __init__(self):
        self.greens = np.array([
            PlayerNN.get_green_nn(
                SETTINGS["w"], SETTINGS["h"]
            ).init_random()
            for _ in range(POP_SIZE)
        ])
        self.reds = np.array([
            PlayerNN.get_red_nn(
                SETTINGS["w"], SETTINGS["h"]
            ).init_random()
            for _ in range(POP_SIZE)
        ])
        self.alpha = 0.01 #0.01/0.1
        self.parents = 4
        self.generation = 0
        self.red_advantage = 0.5
        self.interrupted = False

    def _play_and_score(self, green, red, statistics):
        # Returns: Win scores, Secondary scores for Green. Red is -1*Green
        # Lower is better
        settings = get_game_settings(self.red_advantage)
        start_time = time.time()
        game = Game(green, red, settings, statistics)
        result = game.play()
        delta_time = time.time() - start_time
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

    def _update_params(self):
        pass #TODO: alpha and parents

    def _update_game_rules(self, green_wins):
        if (green_wins > 0):
            # Green has won more than lost
            self.red_advantage = min(1.0, self.red_advantage + 0.1)
        elif (green_wins < 0):
            # Red has won more than lost
            self.red_advantage = max(0.0, self.red_advantage - 0.1)

    # Lower is better
    def _get_scores(self):
        green_scores = [
            [0, 0] for _ in range(len(self.greens))
        ]
        red_scores = [
            [0, 0, 0] for _ in range(len(self.reds))
        ]
        population_comparisons = set([
            (i, red, 0)
            for i, red in enumerate(self.reds)
        ])
        green_wins = 0
        statistics = GameStatistics()
        for i, green in enumerate(self.greens):
            """for _ in range(NUM_COMP):
                choices = list(population_comparisons)
                selected = random.randint(0, len(choices)-1)
                population_comparisons.remove(choices[selected])
                j, red, count = choices[selected]

                green_score_primary, green_score_secondary = self._play_and_score(green, red, statistics)
                green_wins -= green_score_primary
                red_score_primary, red_score_secondary = -1*green_score_primary, -1*green_score_secondary
                green_scores[i][0] += green_score_primary
                green_scores[i][1] += green_score_secondary
                red_scores[j][0] += red_score_primary
                red_scores[j][1] += red_score_secondary

                if (count + 1 < NUM_COMP):
                    population_comparisons.add((j, red, count + 1))"""

            for j, red in enumerate(self.reds):
                green_score_primary, green_score_secondary, invalid_red = self._play_and_score(green, red, statistics)
                green_wins -= green_score_primary
                red_score_primary, red_score_secondary = -1*green_score_primary, -1*green_score_secondary
                green_scores[i][0] += green_score_primary
                green_scores[i][1] += green_score_secondary
                red_scores[j][0] += red_score_primary
                red_scores[j][1] += red_score_secondary
                red_scores[j][2] += invalid_red
        self._update_game_rules(green_wins)
        print(f"Green wins this generation: {green_wins} out of {len(self.greens)*NUM_COMP}")
        statistics.print_statistics()
        return np.array(green_scores), np.array(red_scores)

    def save_checkpoint(self):
        """Save the current population to disk."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        print(f"\nSaving checkpoint at generation {self.generation}...")
        
        # Save green population
        for i, green in enumerate(self.greens):
            filepath = os.path.join(CHECKPOINT_DIR, f"green_{i}.pth")
            green.save_weights(filepath)
        
        # Save red population
        for i, red in enumerate(self.reds):
            filepath = os.path.join(CHECKPOINT_DIR, f"red_{i}.pth")
            red.save_weights(filepath)
        
        # Save generation number
        with open(os.path.join(CHECKPOINT_DIR, "generation.txt"), "w") as f:
            f.write(str(self.generation))
        
        print(f"Checkpoint saved successfully!")

    def load_checkpoint(self):
        """Load population from disk if checkpoint exists."""
        generation_file = os.path.join(CHECKPOINT_DIR, "generation.txt")
        
        if not os.path.exists(generation_file):
            return False
        
        print("Found existing checkpoint, loading...")
        
        # Load generation number
        if os.path.exists(generation_file):
            with open(generation_file, "r") as f:
                self.generation = int(f.read().strip())
        else:
            print("Warning: generation.txt not found, starting from generation 0")
            self.generation = 0
        
        # Load green population
        new_greens = []
        for i in range(POP_SIZE):
            filepath = os.path.join(CHECKPOINT_DIR, f"green_{i}.pth")
            if not os.path.exists(filepath):
                print(f"Warning: Missing green_{i}.pth, starting fresh")
                if (len(new_greens) == 0):
                    new_greens.append(
                        PlayerNN.get_green_nn(SETTINGS["w"], SETTINGS["h"]).init_random()
                    )
                else:
                    sampled = random.choice(new_greens)
                    new_greens.append(
                        sampled.get_mutation(self.alpha)
                    )
            else:
                green = PlayerNN.get_green_nn(SETTINGS["w"], SETTINGS["h"])
                green.load_weights(filepath)
                new_greens.append(green)
        self.greens = np.array(new_greens)
        
        # Load red population
        new_reds = []
        for i in range(POP_SIZE):
            filepath = os.path.join(CHECKPOINT_DIR, f"red_{i}.pth")
            if not os.path.exists(filepath):
                print(f"Warning: Missing red_{i}.pth, starting fresh")
                if (len(new_reds) == 0):
                    new_reds.append(
                        PlayerNN.get_red_nn(SETTINGS["w"], SETTINGS["h"]).init_random()
                    )
                else:
                    sampled = random.choice(new_reds)
                    new_reds.append(
                        sampled.get_mutation(self.alpha)
                    )
                continue
            red = PlayerNN.get_red_nn(SETTINGS["w"], SETTINGS["h"])
            red.load_weights(filepath)
            new_reds.append(red)
        self.reds = np.array(new_reds)
        
        print(f"Checkpoint loaded successfully! Resuming from generation {self.generation}")
        return True

    def run_generation(self):
        self._update_params()
        green_scores, red_scores = self._get_scores()
        green_order = np.lexsort((green_scores[:,1], green_scores[:,0]))
        red_order = np.lexsort((red_scores[:,2], red_scores[:,1], red_scores[:,0]))
        best_greens = self.greens[green_order[:self.parents]]
        best_reds = self.reds[red_order[:self.parents]]
        children_per_parent = int(POP_SIZE // self.parents)
        additional = POP_SIZE - (children_per_parent * self.parents)
        new_greens = []
        for parent in best_greens:
            for i in range(children_per_parent):
                if (ELITISM and i == 0):
                    new_greens.append(parent)
                else:
                    new_greens.append(
                        parent.get_mutation(self.alpha)
                    )
        for _ in range(additional):
            new_greens.append(
                best_greens[0].get_mutation(self.alpha)
            )
        self.greens = np.array(new_greens)

        # Add new_reds creation
        new_reds = []
        for parent in best_reds:
            for i in range(children_per_parent):
                if (ELITISM and i == 0):
                    new_reds.append(parent)
                else:
                    new_reds.append(
                        parent.get_mutation(self.alpha)
                    )
        for _ in range(additional):
            new_reds.append(
                best_reds[0].get_mutation(self.alpha)
            )
        self.reds = np.array(new_reds)
        print(f"Finished generation {self.generation}")
        self.generation += 1


def signal_handler(signum, frame):
    """Handle Ctrl+C signal."""
    print("\n\nInterrupt received! Saving checkpoint...")
    optimizer.interrupted = True

import sys 
if __name__ == "__main__":
    args = sys.argv[1:]
    if (len(args) >= 1):
        CHECKPOINT_DIR = os.path.join("..", "..", args[0])
    else:
        raise Exception("Checkpoint directory argument is required.")

    print(f"Optimization checkpoint directory: {CHECKPOINT_DIR}")

    optimizer = Optimizer()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Try to load existing checkpoint
    checkpoint_loaded = optimizer.load_checkpoint()
    
    if not checkpoint_loaded:
        print("Starting optimization from scratch...")
    
    try:
        while not optimizer.interrupted:
            optimizer.run_generation()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ask user whether to save checkpoint on exit
        answer = input("Do you want to save the checkpoint before exiting? [y/N]: ").strip().lower()
        if answer == "y":
            optimizer.save_checkpoint()
        print("Exiting...")
        sys.exit(0)