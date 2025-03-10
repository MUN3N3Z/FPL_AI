from utils import get_fixtures 
from fpl.models import Fixture, Player
from typing import List
import pandas as pd


def simulate_fixture(fixture: Fixture, players_home: List[Player], players_away: List[Player]) -> pd.DataFrame:
    pass

def simulate_gameweek(season_start_year: str, gameweek: str) -> pd.DataFrame:
    """
        - This function simulates a gameweek by:
            - Sampling players (from available players) proportionally to their probability to start a match
            - Sample the minute each player leaves the pitch from the global multinomial distributions
            - For substitutions, we randomly select a benched player as a replacement - proportionally to their 
            probability of being subbed

    """
    shortened_season_end_year = str(int(season_start_year) + 1)[2:]
    fixtures_csv_file_path = f"./data/{season_start_year + "-" + shortened_season_end_year}/fixtures.csv"
    fixtures_df = pd.read_csv(filepath_or_buffer=fixtures_csv_file_path)
    
    gameweek_fixtures_df = fixtures_df[fixtures_df["GW"] == int(gameweek)]
    for fixture in gameweek_fixtures_df.iterrows():
        h


