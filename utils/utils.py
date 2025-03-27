import numpy as np
import ast
import random

TEAM_ID_MAP_2023_24 = {
    "Arsenal": 1,
    "Aston Villa": 2,
    "Bournemouth": 3,
    "Brentford": 4,
    "Brighton": 5,
    "Burnley": 6,
    "Chelsea": 7,
    "Crystal Palace": 8,
    "Everton": 9,
    "Fulham": 10,
    "Liverpool": 11,
    "Luton": 12,
    "Man City": 13,
    "Man Utd": 14,
    "Newcastle": 15,
    "Nott'm Forest": 16,
    "Sheffield Utd": 17,
    "Spurs": 18,
    "West Ham": 19,
    "Wolves": 20,
    None: None
}

def team_to_id_converter_2023_24(team_name: str) -> int:
    """ Converts a team's name to their id for the 2023/24 season """
    return TEAM_ID_MAP_2023_24[team_name]

def id_to_team_converter_2023_24(team_id: int) -> str:
    """ Converts a team's id to their name for the 2023/24 season """
    id_map = {id: team_name for team_name, id in TEAM_ID_MAP_2023_24.items()}
    return id_map[team_id]

def position_score_points_map(position: str) -> int:
    """ Map a player's position to FPL-based points for scoring a goal """
    position_points = {
        "GKP": 10,
        "DEF": 6,
        "MID": 5,
        "FWD": 4
    }
    return position_points[position]

def position_clean_sheet_points_map(position: str) -> int:
    """ Map a player's position to FPL-based points for assisting a goal """
    position_points = {
        "GKP": 4,
        "DEF": 4,
        "MID": 1,
        "FWD": 0
    }
    return position_points[position]

def np_array_to_list(np_array: np.array) -> str:
    return str(np_array.tolist())

def string_list_to_np_array(string_list: str) -> np.array:
    """ Converts a string representation of a list to a numpy array """
    return np.array(ast.literal_eval(string_list))

def random_bool():
    """ Return a random boolean value """
    random.seed(19)
    return random.choice([True, False])
