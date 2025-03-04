from utils.data_registry import DataRegistry
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

def train_player_position_minutes_priors():
    seasonal_gameweek_player_data = DataRegistry()
    # Global multinomial distributions for players' minutes played at each position
    player_positions = seasonal_gameweek_player_data.player_data["2023-24"]["element_type"].unique()
    player_position_minutes_df = pd.DataFrame({
        "position": player_positions,
        "minutes": [np.zeros(91) for _ in range(len(player_positions))] # Dirichlet prior for minutes at each position
    })
    seasons_with_player_position_data = {"2020-21", "2021-22", "2022-23"}
    # Use a dictionary for faster indexing in the for loop
    position_priors = {
        position: np.ones(91)
        for position in player_positions
    }
    for season, gameweek_data_df in seasonal_gameweek_player_data.gameweek_data.items():
        if season in seasons_with_player_position_data:
            group_data = gameweek_data_df.groupby(["position"])
            for (position,), group_df in group_data:
                # Aggregate observed minutes played for the position
                observed_minutes = np.zeros(91)
                for minutes_played in group_df["minutes"]:
                    if 0 <= minutes_played <= 90:
                        observed_minutes[int(minutes_played)] += 1
                with pm.Model() as player_position_minutes_played_model:
                    # Dirichlet prior for minutes played at each position
                    print(season)
                    prior = pm.Dirichlet(position, a=position_priors[position], shape=91)
                    likelihood = pm.Multinomial(
                        name=f"likelihood_{position}", 
                        n=np.sum(observed_minutes),
                        p=prior,
                        observed=observed_minutes)
                    # Sample the posterior
                    trace = pm.sample(draws=1000, tune=500, chains=2)
                    # Update priors for the next iteration
                    posterior_means = az.summary(trace, var_names=[position]).loc[:, "mean"].values
                    position_priors[position] = posterior_means
    # Transfer priors to player_position_minutes_df
    for position, priors in position_priors.items():
        player_position_minutes_df.loc[player_position_minutes_df["position"] == position, "minutes"] = [priors]
    # Save player_position_minutes_df as a csv file
    player_position_minutes_df.to_csv(path_or_buf="./data/player_position_minutes.csv", index=False)
    return 