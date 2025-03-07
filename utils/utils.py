from fpl.models import Fixture
from typing import List
import requests

def get_fixtures(season_start_year:str, gameweek: str) -> List[Fixture]:
    """
        - Retrieve fixtures for the respective 'season' and 'gameweek' and 
    """
    url = f"https://fantasy.premierleague.com/api/fixtures/?season={season_start_year}&event={gameweek}"
    try:
        fixtures = requests.get(url).json()
        return [Fixture(fixture) for fixture in fixtures]
    except Exception as e:
        print("Error retrieving fixtures:", str(e))