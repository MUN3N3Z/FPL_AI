
def team_converter(team_name: str) -> int:
    """Converts a team's name to their id."""
    team_map = {
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
    return team_map[team_name]