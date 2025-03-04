from utils.data_registry import DataRegistry
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

def train_player_ability_priors():
    # There are 38 gameweeks in a season
    GAMEWEEK_COUNT = 38
    seasonal_gameweek_player_data = DataRegistry()
    # Define player ability (only for players in the 2023/24 season) priors to reflect the occurrences of previous seasons i.e. 2016/17 - 2022/23
    player_ability_df = pd.DataFrame({
        "name": seasonal_gameweek_player_data.player_data["2023-24"]["first_name"] + " " + seasonal_gameweek_player_data.player_data["2023-24"]["second_name"],
        "ρ_β": [np.array([1/4, 1/4, 1/4])] * len(seasonal_gameweek_player_data.player_data["2023-24"]), # Dirichlet prior for ρ
        ("ω", "α"): np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])), # Beta prior for ω
        ("ω", "β"): np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])), # Beta prior for ω
        ("ψ", "α"): np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])), # Beta prior for ψ
        ("ψ", "β"): np.ones(len(seasonal_gameweek_player_data.player_data["2023-24"])) # Beta prior for ψ
    })
    player_priors = {
        player_name: {
            "ρ_β": np.array([1/4, 1/4, 1/4]),
            "ω_α": 1.0,
            "ω_β": 1.0,
            "ψ_α": 1.0,
            "ψ_β": 1.0
        }
        for player_name in player_ability_df["name"]
    }

    for season, gameweek_data_df in seasonal_gameweek_player_data.gameweek_data.items():
        for gameweek_count in range(1, GAMEWEEK_COUNT + 1):
            # Filter out players not present in the 2023/24 season
            gameweek_data_df = gameweek_data_df[gameweek_data_df["name"].isin(player_ability_df["name"])]
            grouped_data = gameweek_data_df.groupby(["name", "GW"])
            for (player_name, _), player_gameweek_data in grouped_data:
                priors = player_priors[player_name]
                with pm.Model() as sequential_player_ability_model:

                    # Priors for the current iteration
                    ρ = pm.Dirichlet('ρ', a=priors["ρ_β"])
                    ω = pm.Beta('ω', alpha=priors["ω_α"], beta=priors["ω_β"])
                    ψ = pm.Beta('ψ', alpha=priors["ψ_α"], beta=priors["ψ_β"])

                    # ρ Likelihood (multinomial) -> (played, subbed, not_used)
                    ρ_observed = np.zeros(3)
                    if player_gameweek_data["minutes"].iloc[0] == 0:
                        ρ_observed[2] = 1
                    else:
                        if player_gameweek_data["starts"].iloc[0] == 1:
                            ρ_observed[0] = 1
                        else:
                            ρ_observed[1] = 1
                    ρ_likelihood = pm.Multinomial(name="ρ_likelihood", n=np.sum(ρ_observed), p=ρ, observed=ρ_observed)
                    # ω Likelihood (binomial)
                    ω_observed = player_gameweek_data["goals_scored"].sum()
                    ω_likelihood = pm.Binomial(name="ω_likelihood", n=ω_observed, p=ω, observed=ω_observed)
                    # ψ Likelihood (binomial)
                    ψ_observed = player_gameweek_data["assists"].sum()
                    ψ_likelihood = pm.Binomial(name="ψ_likelihood", n=ψ_observed, p=ψ, observed=ψ_observed)

                    # Sample the posterior
                    trace = pm.sample(draws=1000, tune=500, chains=2)
                
                    # Update priors for the next iteration
                    posterior_means = az.summary(trace, var_names=["ρ", "ω", "ψ"])
                    priors["ρ_β"] = posterior_means.loc["ρ_β", "mean"][:3]
                    priors["ω_α"] = posterior_means.loc[:, 'mean'][3]
                    priors["ω_β"] = 1 - priors["ω_α"]
                    priors["ψ_α"] = posterior_means.loc[:, 'mean'][4]
                    priors["ψ_β"] = 1 - priors["ψ_α"]
                print(f"{player_name}")
                # Save intermediate results every 5 gameweeks
            if gameweek_count % 5 == 0:
                player_ability_df.to_csv(path_or_buf=f"./data/player_ability_results/player_ability_gw_{gameweek_count}.csv", index=False)

    # Update player_ability_df with the updated priors from player_priors
    for player_name, priors in player_priors.items():
        player_ability_df.loc[player_ability_df["name"] == player_name, "ρ_β"] = [priors["ρ_β"]]
        player_ability_df.loc[player_ability_df["name"] == player_name, ("ω", "α")] = priors["ω_α"]
        player_ability_df.loc[player_ability_df["name"] == player_name, ("ω", "β")] = priors["ω_β"]
        player_ability_df.loc[player_ability_df["name"] == player_name, ("ψ", "α")] = priors["ψ_α"]
        player_ability_df.loc[player_ability_df["name"] == player_name, ("ψ", "β")] = priors["ψ_β"]
    # Save player_ability_df as a csv file
    player_ability_df.to_csv(path_or_buf="./data/player_ability.csv", index=False)
    return 

if __name__ == "__main__":
    train_player_ability_priors()