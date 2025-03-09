import numpy as np
import pandas as pd
from data_registry import DataRegistry

def train_player_ability_priors():
    """
        Updates player ability priors using historical gameweek data via.

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
    player_ability_df = pd.DataFrame({
        "name": seasonal_gameweek_player_data.player_data["2023-24"]["first_name"] + " " + seasonal_gameweek_player_data.player_data["2023-24"]["second_name"],
        "ρ_β": [np.array([0, 0, 0])] * len(seasonal_gameweek_player_data.player_data["2023-24"]), # Dirichlet prior for ρ
        "ω_α": np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])), # Beta prior for ω
        "ω_β": np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])), # Beta prior for ω
        "ψ_α": np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])), # Beta prior for ψ
        "ψ_β": np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])) # Beta prior for ψ
    })
    # Priors obtained from https://cdn.aaai.org/ojs/8259/8259-13-11787-1-2-20201228.pdf
    player_priors = {
        player_name: {
            "ρ_β": np.array([0.25, 0.25, 0.5]),
            "ω_α": 0.0,
            "ω_β": 5.0,
            "ψ_α": 0.0,
            "ψ_β": 5.0
        }
        for player_name in player_ability_df["name"]
    }
    for season, gameweek_data_df in seasonal_gameweek_player_data.gameweek_data.items():
        gameweek_data_df = gameweek_data_df.filter(items=["name", "minutes", "starts", "goals_scored", "assists"])
        # Filter out players not present in the 2023/24 season
        gameweek_data_df = gameweek_data_df[gameweek_data_df["name"].isin(player_ability_df["name"])]
        grouped_data = gameweek_data_df.groupby(["name"])
        print(f"Season: {season}; matched players: {len(grouped_data['name'].unique())}")
        count = 0
        for (player_name,), player_gameweek_data in grouped_data:
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
            count += 1

    # Update player_ability_df with the updated priors from player_priors
    for player_name, priors in player_priors.items():
        # Unable to perf
        player_ability_df.loc[player_ability_df["name"] == player_name, "ρ_β"]  = player_ability_df.loc[player_ability_df["name"] == player_name, "ρ_β"].apply(lambda x: x + priors["ρ_β"])
        player_ability_df.loc[player_ability_df["name"] == player_name, "ω_α"] = priors["ω_α"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ω_β"] = priors["ω_β"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ψ_α"] = priors["ψ_α"]
        player_ability_df.loc[player_ability_df["name"] == player_name, "ψ_β"] = priors["ψ_β"]
    # Save player_ability_df as a csv file
    player_ability_df.to_csv(path_or_buf="./data/player_ability.csv", index=False)
    return 

if __name__ == "__main__":
    train_player_ability_priors()