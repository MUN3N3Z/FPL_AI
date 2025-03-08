from fpl.models import Fixture
import requests
import pandas as pd
import argparse

"""
    - This script retrieves and saves the fixtures for a specific season.
"""

GAMEWEEK_COUNT = 38

def get_gameweek_fixtures(season_start_year:str, gameweek: str) -> pd.DataFrame:
    """
        - Retrieve fixtures for the respective 'season' and 'gameweek' 
    """
    url = f"https://fantasy.premierleague.com/api/fixtures/?season={season_start_year}&event={gameweek}"
    try:
        fixtures = requests.get(url).json()
        fixture_object_list = [Fixture(fixture) for fixture in fixtures]
        fixture_df = pd.DataFrame(columns=["home_team", "away_team", "GW"])
        for fixture in fixture_object_list:
            fixture_df.loc[len(fixture_df)] = [fixture.team_h, fixture.team_a, gameweek]
        return fixture_df
    except Exception as e:
        print("Error retrieving fixtures:", str(e))
    
def aggregate_and_save_fixtures(season_start_year: str) -> None:
    """
        - Aggregate fixtures for a specific season from 'get_gameweek_fixtures()' and save them to a 
        csv file '/data/<season>/fixtures.csv
    """
    season_fixtures_df = pd.DataFrame()
    for gameweek in range(1, GAMEWEEK_COUNT + 1):
        gameweek_fixtures_df = get_gameweek_fixtures(season_start_year, str(gameweek))
        season_fixtures_df = pd.concat([season_fixtures_df, gameweek_fixtures_df], ignore_index=True)
    shortened_season_end_year = str(int(season_start_year) + 1)[2:]
    season_fixtures_df.to_csv(f"./data/{season_start_year + "-" + shortened_season_end_year}/fixtures.csv", index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and save fixtures as a csv file for a given English Premier League season")
    parser.add_argument("season_start_year", type=str, help="The start year of the season (e.g., for the 2023/24 season it's '2023').")
    args = parser.parse_args()
    aggregate_and_save_fixtures(args.season_start_year)