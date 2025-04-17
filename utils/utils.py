import numpy as np
import ast
import random
import pandas as pd
from typing import Dict, Tuple
from data_registry import DataRegistry
from constants import RANDOM_SEED, POSITIONS

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
    None: None,
}


def team_to_id_converter_2023_24(team_name: str) -> int:
    """Converts a team's name to their id for the 2023/24 season"""
    return TEAM_ID_MAP_2023_24[team_name]


def id_to_team_converter_2023_24(team_id: int) -> str:
    """Converts a team's id to their name for the 2023/24 season"""
    id_map = {id: team_name for team_name, id in TEAM_ID_MAP_2023_24.items()}
    return id_map[team_id]


def position_score_points_map(position: str) -> int:
    """Map a player's position to FPL-based points for scoring a goal"""
    position_points = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
    return position_points[position]


def position_clean_sheet_points_map(position: str) -> int:
    """Map a player's position to FPL-based points for assisting a goal"""
    position_points = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    return position_points[position]


def position_num_players(position: str) -> int:
    """Map a position to the allowed number of players from the same position in an FPL team"""
    position_count = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    return position_count[position]


def string_list_to_np_array(string_list: str) -> np.array:
    """Converts a string representation of a list to a numpy array"""
    if isinstance(string_list, np.ndarray):
        # If the input is already a NumPy array, return it as is
        return string_list
    elif isinstance(string_list, str):
        # If the input is a string, evaluate it and convert to NumPy array
        return np.array(ast.literal_eval(string_list))
    else:
        raise ValueError(f"Unsupported input type: {type(string_list)}")


def random_bool():
    """Return a random boolean value"""
    random.seed(RANDOM_SEED)
    return random.choice([True, False])


def format_season_name(season_start_year: str) -> str:
    """Create data file name based on the season_start_year"""
    shortened_season_end_year = str(int(season_start_year) + 1)[2:]
    return f"{season_start_year + " - " + shortened_season_end_year}"


def update_gw_data(
    gameweek_data: pd.DataFrame,
    player_points_df: pd.DataFrame,
    cumulative_real_player_points: Dict[str, int],
    gw: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    - Updates the passed player_points_df with players' cumulative real points and prices
    """

    gameweek_data = gameweek_data[gameweek_data["name"].isin(player_points_df["name"])]
    player_points_df["cumulative_real_points"] = 0
    # Players without price tags will be unpurchasable by default
    player_points_df["price"] = 1000.0
    player_points_df["name_index"] = player_points_df["name"]
    player_points_df.set_index("name_index", inplace=True)
    for _, player_row in gameweek_data.iterrows():
        player_name = player_row["name"]
        # Scale players' pricess appropriately: $£55 == £5.5
        player_points_df.loc[player_name, "price"] = float(player_row["value"]) / 10.0
        if gw > 1:
            cumulative_real_player_points[player_name] += player_row["total_points"]
            player_points_df.loc[
                player_name, "cumulative_real_points"
            ] = cumulative_real_player_points[player_name]
    return player_points_df, cumulative_real_player_points


def choose_formation(team_id: int) -> Dict[str, int]:
    """
    - Choose a formation based of EPL statistics for the 2023/24 season
    - Source: https://www.premierleague.com/news/4030093
    """
    # 4-5-1 -> 4-2-3-1
    # 3-6-1 -> 3-4-2-1
    formation_map = {
        "Arsenal": "4-3-3",
        "Aston Villa": "4-5-1",
        "Bournemouth": "4-5-1",
        "Brentford": "3-5-2",
        "Brighton": "4-5-1",
        "Burnley": "4-4-2",
        "Chelsea": "4-5-1",
        "Crystal Palace": "4-3-3",
        "Everton": "4-5-1",
        "Fulham": "4-5-1",
        "Liverpool": "4-5-1",  # Modified from 4-3-3 (inadequate forwards)
        "Luton": "3-6-1",
        "Man City": "4-5-1",
        "Man Utd": "4-5-1",
        "Newcastle": "4-4-2",  # Modified from 4-3-3 (inadequate forwards)
        "Nott'm Forest": "4-5-1",
        "Sheffield Utd": "3-5-2",
        "Spurs": "4-5-1",
        "West Ham": "4-5-1",
        "Wolves": "3-6-1",
    }
    # 1-(formation sequence) represents the Goalkeeper
    formation = ("1-" + formation_map[(id_to_team_converter_2023_24(team_id))]).split(
        "-"
    )
    return {position: int(count) for position, count in zip(POSITIONS, formation)}
