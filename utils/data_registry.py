import pandas as pd
from typing import Dict, List
from constants import DATA_FOLDER
import numpy as np
from numpy.typing import NDArray
from utils import format_season_name


class DataRegistry:
    """A class to manage loading and storage of FPL data from CSV files."""

    def __init__(self, gw_data_columns: List[str] = None):
        self.file_paths: Dict[str, Dict[str, str]] = {
            "2016-17": {
                "player_data": f"{DATA_FOLDER}/2016-17/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2016-17/merged_gw.csv",
            },
            "2017-18": {
                "player_data": f"{DATA_FOLDER}/2017-18/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2017-18/merged_gw.csv",
            },
            "2018-19": {
                "player_data": f"{DATA_FOLDER}/2018-19/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2018-19/merged_gw.csv",
            },
            "2019-20": {
                "player_data": f"{DATA_FOLDER}/2019-20/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2019-20/merged_gw.csv",
            },
            "2020-21": {
                "player_data": f"{DATA_FOLDER}/2020-21/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2020-21/merged_gw.csv",
            },
            "2021-22": {
                "player_data": f"{DATA_FOLDER}/2021-22/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2021-22/merged_gw.csv",
            },
            "2022-23": {
                "player_data": f"{DATA_FOLDER}/2022-23/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2022-23/merged_gw.csv",
            },
            "2023-24": {
                "player_data": f"{DATA_FOLDER}/2023-24/cleaned_players.csv",
                "merged_gameweek_data": f"{DATA_FOLDER}/2023-24/merged_gw.csv",
            },
        }
        self.gameweek_data: Dict[str, pd.DataFrame] = {}
        self.player_data: Dict[str, pd.DataFrame] = {}
        self._load_gameweek_data(gw_data_columns)
        # Script to clean up the data - player names and starting stats
        # self.standardize_data()

    def _load_gameweek_data(self, gw_data_columns: List[str] | None) -> None:
        """
        - Use ISO-8859-1 file encoding to load seasonal gameweek and player data into the registry.
        - "gw_data_columns" describes the columns to filter in the merged_gw.csv data files
        """
        for season, _ in self.file_paths.items():
            gameweek_file_path = self.file_paths[season]["merged_gameweek_data"]
            player_file_path = self.file_paths[season]["player_data"]
            try:
                self.gameweek_data[season] = pd.read_csv(
                    gameweek_file_path, encoding="ISO-8859-1"
                )
                if gw_data_columns:
                    self.gameweek_data[season] = self.gameweek_data[season].filter(
                        items=gw_data_columns
                    )

                self.player_data[season] = pd.read_csv(
                    player_file_path, encoding="ISO-8859-1"
                )
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
                gameweek_df["name"] = gameweek_df["name"].apply(
                    lambda x: " ".join(x.split("_"))
                )
            elif season in ["2018-19", "2019-20"]:
                gameweek_df["name"] = gameweek_df["name"].apply(
                    lambda x: " ".join(x.split("_")[:-1])
                )

    def _standardize_player_starts_sub_unused_data(self) -> None:
        """
        - Add one columns to each "merged_gw_csv" that show if a player started a match
        - 2016/17 - 2021/22 seasons only have data on player minutes
            - Assumption: Player started the game if he played >=50 minutes. Therefore, they got subbed into the match if they played < 50 minutes.
        """
        seasons_not_to_edit = {"2022-23", "2023-24"}
        for season, gameweek_df in self.gameweek_data.items():
            if season not in seasons_not_to_edit:
                gameweek_df["starts"] = gameweek_df["minutes"].apply(
                    lambda x: 1 if x >= 50 else 0
                )
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

    def extract_player_names(self, seasons: List[str]) -> NDArray:
        """Return list of unique player names from specified 'seasons'"""
        players = np.array([], dtype=str)
        for season in seasons:
            formatted_season_name = format_season_name(season)
            season_data = self.gameweek_data[formatted_season_name]
            new_players = season_data[
                ~season_data["name"].isin(players.tolist())
            ].name.unique()
            players = np.concatenate((players, new_players))

        return players
