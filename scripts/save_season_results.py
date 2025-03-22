import pandas as pd
import argparse

def save_season_results(season_start_year: str) -> None:
    """
        - Retrieves season results data from www.football-data.co.uk and saves the required data columns
        ["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"] as a csv file
    """
    url_season_repr = season_start_year[-2:] + str(int(season_start_year[-2:]) + 1)
    url = f"https://www.football-data.co.uk/mmz4281/{url_season_repr}/E0.csv"
    file_system_season_repr = season_start_year + "-" + str(int(season_start_year[-2:]) + 1)
    csv_storage_path = f"../data/{file_system_season_repr}/fixture_results.csv"
    data = pd.read_csv(filepath_or_buffer=url)
    data = data[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    data = data.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
    data.to_csv(path_or_buf=csv_storage_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and save results as a csv file for a given English Premier League season")
    parser.add_argument("season_start_year", type=str, help="The start year of the season (e.g., for the 2023/24 season it's '2023').")
    args = parser.parse_args()
    save_season_results(args.season_start_year)