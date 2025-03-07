from data_registry import DataRegistry
import numpy as np
import pandas as pd

def train_player_position_minutes_priors():
    """
        - Creates four multinomial distributions of the minutes when players are
        obeserved to leave a match given they started it for each position (GK, DEF, MID, FWD)
        - A value of 90 drawn from any of these distributions corresponds to a player finishing 
        the match without being substituted
        - Underlying assumption: a player started a match if they played >= 50 minutes
    """
    seasonal_gameweek_player_data = DataRegistry()
    # Global multinomial distributions for players' minutes played at each position
    player_positions = seasonal_gameweek_player_data.player_data["2023-24"]["element_type"].unique()
    player_position_minutes_df = pd.DataFrame({
        "position": player_positions,
        "minutes": [np.ones(91) for _ in range(len(player_positions))] # Dirichlet prior for minutes at each position
    })
    seasons_with_player_position_data = {"2020-21", "2021-22", "2022-23"}
    # Use a dictionary for faster indexing in the for loop
    position_priors = {
        position: np.ones(91).astype(int)
        for position in player_positions
    }
    for season, gameweek_data_df in seasonal_gameweek_player_data.gameweek_data.items():
        if season in seasons_with_player_position_data:
            group_data = gameweek_data_df.groupby(["position"])
            for (position,), position_group_df in group_data:
                players_who_started = position_group_df[position_group_df["minutes"] >= 50]
                position_priors[position] += np.bincount(players_who_started["minutes"], minlength=91)

    # Transfer priors to player_position_minutes_df
    for position, priors in position_priors.items():
        player_position_minutes_df.loc[player_position_minutes_df["position"] == position, "minutes"] = [priors]
    # Save player_position_minutes_df as a csv file
    player_position_minutes_df.to_csv(path_or_buf="./data/player_position_minutes.csv", index=False)
    return 

if __name__=="__main__":
    train_player_position_minutes_priors()