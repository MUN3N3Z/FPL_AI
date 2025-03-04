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
        self.seasons = list(self.file_paths.keys())
        self.gameweek_data = {}
        self.player_data = {}
        self.load_gameweek_data()
        # Script to clean up the data - player names and starting stats
        # self.standardize_data()

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
    
    def _standardize_player_names(self) -> None:
        """ 
            Standardize player names in gameweek data to match those in player data. 
            - 2019/20 and 2018/19 merged_gw.csv files have player names in the format "First_Last_ID"
            - 2017/18 and 2016/17 merged_gw.csv files have player names in the format "First_Last"
            - We need to standardize these names to match this format "First Last"
        """
        for season, gameweek_df in self.gameweek_data.items():
            if season in ["2016-17", "2017-18"]:
                gameweek_df["name"] = gameweek_df["name"].apply(lambda x: " ".join(x.split("_")))
            elif season in ["2018-19", "2019-20"]:
                gameweek_df["name"] = gameweek_df["name"].apply(lambda x: " ".join(x.split("_")[:-1]))

    def _standardize_player_starts_sub_unused_data(self) -> None:
        """
            - Add one columns to each "merged_gw_csv" that show if a player started a match
            - 2016/17 - 2021/22 seasons only have data on player minutes
                - Assumption: Player started the game if he played >=50 minutes. Therefore, they got subbed into the match if they played < 50 minutes.
        """
        seasons_not_to_edit = {"2022-23", "2023-24"}
        for season, gameweek_df in self.gameweek_data.items():
            if season not in seasons_not_to_edit:
                gameweek_df["starts"] = gameweek_df["minutes"].apply(lambda x: 1 if x >= 50 else 0)
        return

    def _standardize_player_positions(self) -> None:
        """
            - Annoyingly discovered that there the goalkeeper position in the "positions" column in <season>/merged_gw.csv is not uniform
                - Solution: replace "GKP" with "GK"
            - Standard position names: ['FWD' 'DEF' 'MID' 'GK']
        """
        seasons_with_player_position_data = {"2020-21", "2021-22", "2022-23"}
        for season, gameweek_df in self.gameweek_data.items(): 
            if season in seasons_with_player_position_data:
                gameweek_df["position"] = gameweek_df["position"].replace("GKP", "GK")
        return
    def standardize_data(self) -> None:
        """
            - Standardize names and minutes played format for players in gameweek data and save the result
        """
        self._standardize_player_names()
        self._standardize_player_starts_sub_unused_data()
        self._standardize_player_positions()
        for season, file_path_dict in self.file_paths.items():
            output_file_path = file_path_dict["merged_gameweek_data"]
            self.gameweek_data[season].to_csv(output_file_path, index=False)
        print("Standardized gameweek data saved to csv!")
        return
        
        
    