from typing import List

# English Premier League/Fantasy Premier League facts
GAMEWEEK_COUNT: int = 38
STARTING_PLAYERS: int = 11
MATCH_MINUTES: int = 90
MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS: int = 60
ASSIST_POINTS: int = 3
HALF_GAME_MINUTES: int = 45
NUM_PLAYERS_IN_FPL: int = 15
POSITIONS: List[str] = ["GK, DEF, MID, FWD"]
TEAM_COUNT_LIMIT: int = 3
NUM_BENCHED_PLAYERS: int = 4
PROMOTED_TEAMS_2022_23: List[str] = ["Luton", "Sheffield Utd", "Burnley"]
RELEGATED_TEAMS_2022_23: List[str] = ["Leicester", "Southampton", "Leeds"]

# Sampling constants
RANDOM_SEED: int = 19
NUM_SAMPLES: int = 5000
BURN_SAMPLES: int = 500
CHAINS: int = 4
TARGET_ACCEPT: float = 0.9

# Project file paths
DATA_FOLDER: str = "../data"
PLAYER_ABILITY_FILE: str = "player_ability.csv"
POSITION_MINUTES_FILE: str = "player_position_minutes.csv"
FIXTURE_FILE: str = "fixtures.csv"
