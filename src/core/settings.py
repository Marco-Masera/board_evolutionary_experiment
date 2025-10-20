SETTINGS = {
    "w": 16,
    "h": 8,
    "energy_for_new_piece": 10,
    "energy_gain_stolen_piece": 80,

    "red_start_energy_min": 15,
    "red_start_energy_max": 100,
    "starting_pieces_min": 10,
    "starting_pieces_max": 80,
}


def get_game_settings(red_advantage = 0.5):
    # Interpolate red's starting energy (min when red_advantage=0, max when red_advantage=1)
    red_start_energy = int(
        SETTINGS["red_start_energy_min"] + 
        red_advantage * (SETTINGS["red_start_energy_max"] - SETTINGS["red_start_energy_min"])
    )
    
    # Interpolate red's starting pieces (min when red_advantage=0, max when red_advantage=1)
    starting_pieces_red = int(
        SETTINGS["starting_pieces_min"] + 
        red_advantage * (SETTINGS["starting_pieces_max"] - SETTINGS["starting_pieces_min"])
    )
    
    # Interpolate green's starting pieces (max when red_advantage=0, min when red_advantage=1)
    starting_pieces_green = int(
        SETTINGS["starting_pieces_max"] - 
        red_advantage * (SETTINGS["starting_pieces_max"] - SETTINGS["starting_pieces_min"])
    )
    
    return {
        "red_start_energy": red_start_energy,
        "starting_pieces_red": starting_pieces_red,
        "starting_pieces_green": starting_pieces_green,
    }