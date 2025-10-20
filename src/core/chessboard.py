from settings import SETTINGS
from nn import GreenNN, RedNN
import numpy as np

class GameStatistics:
    def __init__(self):
        self._matches_played = 0
        self._pieces_placed = 0
        self._piece_placement_failed_no_energy = 0
        self._invalid_move_red_selected_empty = 0
        self._red_pieces_moved = 0
        self._red_pieces_captured = 0
        self._invalid_move_red_selected_opponent = 0
        self._green_pieces_moved = 0
        self._invalid_move_green_destination_not_empty = 0
        self._invalid_move_green_selected_empty = 0
    
    def new_match(self):
        self._matches_played += 1
    
    def piece_placed(self):
        self._pieces_placed += 1
    
    def piece_placement_failed_no_energy(self):
        self._piece_placement_failed_no_energy += 1
    
    def invalid_move_red_selected_empty(self):
        self._invalid_move_red_selected_empty += 1
    
    def red_piece_moved(self):
        self._red_pieces_moved += 1
    
    def red_piece_captured(self):
        self._red_pieces_captured += 1
    
    def invalid_move_red_selected_opponent(self):
        self._invalid_move_red_selected_opponent += 1
    
    def green_piece_moved(self):
        self._green_pieces_moved += 1
    
    def invalid_move_green_destination_not_empty(self):
        self._invalid_move_green_destination_not_empty += 1
    
    def invalid_move_green_selected_empty(self):
        self._invalid_move_green_selected_empty += 1
    
    def print_statistics(self):
        print("=" * 50)
        print("GAME STATISTICS")
        print("=" * 50)
        print(f"Matches played: {self._matches_played}")
        print(f"\nRed Player:")
        print(f"  Pieces placed: {self._pieces_placed}")
        print(f"  Pieces moved: {self._red_pieces_moved}")
        print(f"  Pieces captured: {self._red_pieces_captured}")
        print(f"  Failed placements (no energy): {self._piece_placement_failed_no_energy}")
        print(f"  Invalid moves (selected empty): {self._invalid_move_red_selected_empty}")
        print(f"  Invalid moves (selected opponent): {self._invalid_move_red_selected_opponent}")
        print(f"\nGreen Player:")
        print(f"  Pieces moved: {self._green_pieces_moved}")
        print(f"  Invalid moves (destination not empty): {self._invalid_move_green_destination_not_empty}")
        print(f"  Invalid moves (selected empty): {self._invalid_move_green_selected_empty}")
        print("=" * 50)

class Chessboard:
    
    def __init__(self, green_network, red_network, settings, statistics = GameStatistics()):
        self.game_statistics = statistics
        self.settings = settings
        self.green_network = green_network
        self.red_network = red_network
        self.width = SETTINGS["w"]
        self.height = SETTINGS["h"]
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.red_energy = self.settings["red_start_energy"]
        self.turn = 1
    
    def setup_board(self):
        # Initialize pieces for both players
        # Red is 1, green is -1
        self.game_statistics.new_match()
        self._place_starting_pieces(player=1, count=self.settings["starting_pieces_red"])
        self._place_starting_pieces(player=-1, count=self.settings["starting_pieces_green"])

    def _place_starting_pieces(self, player, count):
        """Place starting pieces for a player on empty positions of the board."""
        # Find all empty positions (where board value is 0)
        empty_positions = np.argwhere(self.board == 0)
        
        # Randomly select 'count' positions from empty positions
        if len(empty_positions) < count:
            raise ValueError(f"Not enough empty positions to place {count} pieces")
        
        selected_indices = np.random.choice(len(empty_positions), size=count, replace=False)
        selected_positions = empty_positions[selected_indices]
        
        # Place the player's pieces at selected positions
        for pos in selected_positions:
            self.board[pos[0], pos[1]] = player

    def step(self):
        if (self.turn == 1):
            # Red turn
            position_index, operation_index = self.red_network.from_board_state(self.board, self.red_energy / 30, self.red_energy / (SETTINGS["energy_for_new_piece"]*10))
            # Find position from index
            row = position_index // self.width
            col = position_index % self.width
            # Selected an empty cell
            if (self.board[row, col] == 0):
                if operation_index == 8:
                    # Place a new piece if enough energy
                    if self.red_energy >= SETTINGS["energy_for_new_piece"]:
                        self.board[row, col] = 1
                        self.red_energy -= SETTINGS["energy_for_new_piece"]
                        self.game_statistics.piece_placed()
                    else:
                        self.red_energy -= 1 # Penalty for invalid operation
                        self.game_statistics.piece_placement_failed_no_energy()
                else:
                    self.red_energy -= 1 # Penalty for invalid operation
                    self.game_statistics.invalid_move_red_selected_empty()
            elif (self.board[row, col] == 1):
                self.red_energy -= 1
                # Check operation
                dest_row, dest_col = row, col
                if operation_index == 0:
                    # Move left
                    dest_col -= 1
                elif operation_index == 1:
                    # Move right
                    dest_col += 1
                elif operation_index == 2:
                    # Move up
                    dest_row -= 1
                elif operation_index == 3:
                    # Move down
                    dest_row += 1
                elif operation_index == 4:
                    # Left-Up
                    dest_row -= 1
                    dest_col -= 1
                elif operation_index == 5:
                    # Right-Up
                    dest_row -= 1
                    dest_col += 1
                elif operation_index == 6:
                    # Left-Down
                    dest_row += 1
                    dest_col -= 1
                elif operation_index == 7:
                    # Right-Down
                    dest_row += 1
                    dest_col += 1
                
                # Validate bounds - if out of bounds, don't move at all
                if dest_row < 0 or dest_row >= self.height or dest_col < 0 or dest_col >= self.width:
                    dest_row, dest_col = row, col  # Stay in place
                print(dest_row, dest_col, self.height, self.width)
                # Execute move if destination is empty or capture if opponent
                if self.board[dest_row, dest_col] == 0:
                    self.board[dest_row, dest_col] = 1
                    self.board[row, col] = 0
                    self.game_statistics.red_piece_moved()
                elif self.board[dest_row, dest_col] == -1:
                    # Capture opponent's piece
                    self.board[dest_row, dest_col] = 1
                    self.board[row, col] = 0
                    self.red_energy += SETTINGS["energy_gain_stolen_piece"]
                    self.game_statistics.red_piece_captured()
            else:
                self.red_energy -= 1  # Penalty for invalid selection
                self.game_statistics.invalid_move_red_selected_opponent()
        elif (self.turn == 2):
            # Green turn
            position_index, operation_index = self.green_network.from_board_state(self.board)
            # Find position from index
            row = position_index // self.width
            col = position_index % self.width
            if (self.board[row, col] == -1):
                # Check operation
                dest_row, dest_col = row, col
                if operation_index == 0:
                    # Move left
                    dest_col -= 1
                elif operation_index == 1:
                    # Move right
                    dest_col += 1
                elif operation_index == 2:
                    # Move up
                    dest_row -= 1
                elif operation_index == 3:
                    # Move down
                    dest_row += 1
                
                # Validate bounds - if out of bounds, don't move at all
                if dest_row < 0 or dest_row >= self.height or dest_col < 0 or dest_col >= self.width:
                    dest_row, dest_col = row, col  # Stay in place
                
                # Execute move if destination is empty
                if self.board[dest_row, dest_col] == 0:
                    self.board[dest_row, dest_col] = -1
                    self.board[row, col] = 0
                    self.game_statistics.green_piece_moved()
                else:
                    self.game_statistics.invalid_move_green_destination_not_empty()
            else:
                self.game_statistics.invalid_move_green_selected_empty()
        else:
            raise Exception("Invalid turn value")

        if (self.red_energy <= 0):
            return "green"
        if (np.sum(self.board == -1) == 0):
            return "red"
        self.turn = (self.turn % 2) + 1  # Switch turns between 1 and 2
        return None

    def print_stats(self):
        red_pieces = np.sum(self.board == 1)
        green_pieces = np.sum(self.board == -1)
        print(f"Red pieces: {red_pieces}, Green pieces: {green_pieces}, Red energy: {self.red_energy}")

class Game:
    def __init__(self, nn_green, nn_red, settings, statistics = GameStatistics()):
        self.chessboard = Chessboard(nn_green, nn_red, settings, statistics)
    
    def play(self):
        self.chessboard.setup_board()
        num_turns = 0
        winner = None
        while winner is None:
            winner = self.chessboard.step()
            num_turns += 1
            if (num_turns % 1500 == 0):
                self.chessboard.print_stats()
        #self.chessboard.print_stats()
        return winner
        return None

    def print_stats(self):
        red_pieces = np.sum(self.board == 1)
        green_pieces = np.sum(self.board == -1)
        print(f"Red pieces: {red_pieces}, Green pieces: {green_pieces}, Red energy: {self.red_energy}")

class Game:
    def __init__(self, nn_green, nn_red, settings, statistics = GameStatistics()):
        self.chessboard = Chessboard(nn_green, nn_red, settings, statistics)
    
    def play(self):
        self.chessboard.setup_board()
        num_turns = 0
        winner = None
        while winner is None:
            winner = self.chessboard.step()
            num_turns += 1
            if (num_turns % 1500 == 0):
                self.chessboard.print_stats()
        #self.chessboard.print_stats()
        return winner
