import numpy as np
import pandas as pd
from data_registry import DataRegistry
from utils import team_to_id_converter_2023_24, np_array_to_list

def train_player_ability_priors():
    """
        Updates ability priors for players in the 2023/24 season using historical gameweek data.

        This function iteratively updates priors for player abilities (playing time,
        goal scoring, and assist making) using Bayesian inference and historical
        data from 2016/17 - 2022/23 seasons. The priors are updated sequentially for each
        season, using Beta-Binomial and Dirichlet-Multinomial conjugates.

        The trained priors are saved to 'player_ability.csv'.
    """
    # There are 38 gameweeks in a season
    GAMEWEEK_COUNT = 38
    seasonal_gameweek_player_data = DataRegistry()
    # Initialize player ability priors for players in the 2023/24 season.
    # Priors are defined for:
    # - ρ (rho): Playing status probabilities (Dirichlet prior) - (played, subbed, not_used)
    # - ω (omega): Goal scoring ability (Beta prior)
    # - ψ (psi): Assist making ability (Beta prior)

    # Drop 2023/24 season from the data set
    season_2023_24_gameweek_data_unique_player_rows = seasonal_gameweek_player_data.gameweek_data.pop("2023-24").drop_duplicates(subset=["name"])
    player_ability_df = pd.DataFrame({
        "name": season_2023_24_gameweek_data_unique_player_rows["name"],
        "position": season_2023_24_gameweek_data_unique_player_rows["position"],
        "team": None,
        "ρ_β": [np.array([0, 0, 0])] * len(season_2023_24_gameweek_data_unique_player_rows), # Dirichlet prior for ρ
        "ω_α": np.ones(len(season_2023_24_gameweek_data_unique_player_rows)), # Beta prior for ω
        "ω_β": np.ones(len(season_2023_24_gameweek_data_unique_player_rows)), # Beta prior for ω
        "ψ_α": np.ones(len(season_2023_24_gameweek_data_unique_player_rows)), # Beta prior for ψ
        "ψ_β": np.ones(len(season_2023_24_gameweek_data_unique_player_rows)) # Beta prior for ψ
    })
    # Priors obtained from https://cdn.aaai.org/ojs/8259/8259-13-11787-1-2-20201228.pdf
    player_priors = {
        player_name: {
            "team": None,
            "ρ_β": np.array([0.25, 0.25, 0.5]),
            "ω_α": 0.0,
            "ω_β": 5.0,
            "ψ_α": 0.0,
            "ψ_β": 5.0
        }
        for player_name in player_ability_df["name"]
    }
    for season, gameweek_data_df in seasonal_gameweek_player_data.gameweek_data.items():
        gameweek_data_df = gameweek_data_df.filter(items=["name", "minutes", "starts", "goals_scored", "assists", "position"])
        # Filter out players not present in the 2023/24 season
        gameweek_data_df = gameweek_data_df[gameweek_data_df["name"].isin(player_ability_df["name"])]
        grouped_data = gameweek_data_df.groupby(["name"])
        print(f"Season: {season}; matched players: {len(grouped_data['name'].unique())}")
        for (player_name,), player_gameweek_data in grouped_data:
            if player_priors[player_name]["team"] is None:
                player_priors[player_name]["team"] = team_to_id_converter_2023_24(
                    season_2023_24_gameweek_data_unique_player_rows.loc[season_2023_24_gameweek_data_unique_player_rows["name"] == player_name, "team"].values[0]
                )
            ρ_observed = np.array(player_gameweek_data.apply(
                lambda row: [1, 0, 0] if row["starts"] == 1 else [0, 1, 0] if row["minutes"] > 0 else [0, 0, 1],
                axis=1
            ).tolist()).sum(axis=0)
            ω_observed = player_gameweek_data["goals_scored"].sum()
            ψ_observed = player_gameweek_data["assists"].sum()
            total_gameweeks_played = len(player_gameweek_data)
            # Conjugate prior updates
            # Dirichlet prior update for ρ
            player_priors[player_name]["ρ_β"] += ρ_observed
            # Beta prior update for ω
            player_priors[player_name]["ω_α"] += ω_observed
            player_priors[player_name]["ω_β"] += (total_gameweeks_played - ω_observed)
            # Beta prior update for ψ
            player_priors[player_name]["ψ_α"] += ψ_observed
            player_priors[player_name]["ψ_β"] += (total_gameweeks_played - ψ_observed)

    # Update player_ability_df with the updated priors from player_priors
    for player_name, priors in player_priors.items():
        player_ability_df.loc[player_ability_df["name"] == player_name, "team"] = priors["team"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ρ_β"]  = player_ability_df.loc[player_ability_df["name"] == player_name, "ρ_β"].apply(lambda x: x + priors["ρ_β"])
        player_ability_df.loc[player_ability_df["name"] == player_name, "ω_α"] = priors["ω_α"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ω_β"] = priors["ω_β"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ψ_α"] = priors["ψ_α"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ψ_β"] = priors["ψ_β"]
    # Convert "ρ_β" column from np.array type to list for accurate loading
    player_ability_df["ρ_β"] = player_ability_df["ρ_β"].apply(np_array_to_list)
    # Save player_ability_df as a csv file
    player_ability_df.to_csv(path_or_buf="./data/2023-24/player_ability.csv", index=False)
    return 

if __name__ == "__main__":
    train_player_ability_priors()