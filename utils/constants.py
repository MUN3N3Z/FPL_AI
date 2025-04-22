from typing import List

# English Premier League/Fantasy Premier League facts
GAMEWEEK_COUNT: int = 38
STARTING_PLAYERS: int = 11
MATCH_MINUTES: int = 90
MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS: int = 60
ASSIST_POINTS: int = 3
HALF_GAME_MINUTES: int = 45
NUM_PLAYERS_IN_FPL: int = 15
POSITIONS: List[str] = ["GK", "DEF", "MID", "FWD"]
TEAM_COUNT_LIMIT: int = 3
NUM_BENCHED_PLAYERS: int = 4
RELEGATED_TEAMS_2021_22: List[str] = ["Burnley", "Watford", "Norwich"]
PROMOTED_TEAMS_2021_22: List[str] = ["Fulham", "Bournemouth", "Nott'm Forest"]
RELEGATED_TEAMS_2022_23: List[str] = ["Leicester", "Southampton", "Leeds"]
PROMOTED_TEAMS_2022_23: List[str] = ["Luton", "Sheffield Utd", "Burnley"]
TRANSFER_BUDGET = 100.0

# Sampling constants
RANDOM_SEED: int = 19
NUM_SAMPLES: int = 5000
BURN_SAMPLES: int = 500
CHAINS: int = 4
TARGET_ACCEPT: float = 0.9

# Project file paths
DATA_FOLDER: str = "../data"
