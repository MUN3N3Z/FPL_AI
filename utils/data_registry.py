import pandas as pd

class DataRegistry:
    """ A class to manage loading and storage of FPL data from CSV files. """
    def __init__(self):
        self.file_paths = {
            "2016-17" : {
                "player_data": "./data/2016-17/cleaned_players.csv",
                "merged_gameweek_data": "./data/2016-17/merged_gw.csv"
            },
            "2017-18" : {
                "player_data": "./data/2017-18/cleaned_players.csv",
                "merged_gameweek_data": "./data/2017-18/merged_gw.csv"
            },
            "2018-19" : {
                "player_data": "./data/2018-19/cleaned_players.csv",
                "merged_gameweek_data": "./data/2018-19/merged_gw.csv"
            },
            "2019-20" : {
                "player_data": "./data/2019-20/cleaned_players.csv",
                "merged_gameweek_data": "./data/2019-20/merged_gw.csv"
            },
            "2020-21" : {
                "player_data": "./data/2020-21/cleaned_players.csv",
                "merged_gameweek_data": "./data/2020-21/merged_gw.csv"
            },
            "2021-22" : {
                "player_data": "./data/2021-22/cleaned_players.csv",
                "merged_gameweek_data": "./data/2021-22/merged_gw.csv"
            },
            "2022-23" : {
                "player_data": "./data/2022-23/cleaned_players.csv",
                "merged_gameweek_data": "./data/2022-23/merged_gw.csv"
            },
            "2023-24" : {
                "player_data": "./data/2023-24/cleaned_players.csv",
                "merged_gameweek_data": "./data/2023-24/merged_gw.csv"
            }
        }
        self.gameweek_data = {}
        self.player_data = {}
        self.load_gameweek_data()

    def load_gameweek_data(self) -> None:
        """ Use ISO-8859-1 file encoding to load seasonal gameweek and player data into the registry. """
        for season, _ in self.file_paths.items():
            gameweek_file_path = self.file_paths[season]["merged_gameweek_data"]
            player_file_path = self.file_paths[season]["player_data"]
            try:
                self.gameweek_data[season] = pd.read_csv(gameweek_file_path, encoding="ISO-8859-1")
                self.player_data[season] = pd.read_csv(player_file_path, encoding="ISO-8859-1")
            except UnicodeDecodeError:
                raise Exception(f"Error loading data for {season}")