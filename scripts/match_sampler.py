from fpl.models import Fixture, Player
from typing import List
import pandas as pd
import pymc as pm
import numpy as np

NUM_SAMPLES = 5000
BURN_SAMPLES = 1000
CHAINS = 2

STARTING_PLAYERS = 11

def simulate_fixture(fixture: Fixture, players_home: List[Player], players_away: List[Player]) -> pd.DataFrame:
    pass

def sample_team_stats(team_players_abilities: pd.DataFrame) -> pd.DataFrame:
    """
        - Sample each player's probability of starting a game, scoring a goal, and assisting one from their priors
        - Returns the team's players' abilities data frame with additional columns that contain the respective stats
    """
    sampled_data = []
    # Convert the 'ρ_β' column from string to NumPy arrays
    team_players_abilities["ρ_β"] = team_players_abilities["ρ_β"].apply(lambda string_list: np.array(eval(string_list)))
    for _, player_row in team_players_abilities.iterrows():
        with pm.Model() as player_match_model:
            # Define the distributions
            start_sub_unused_dirichlet_dist = pm.Dirichlet("start_sub_unused_dirichlet_dist", 
                                                           a=player_row["ρ_β"])
            score_beta_dist = pm.Beta("score_beta", alpha=player_row["ω_α"], beta=player_row["ω_β"])
            assist_beta_dist = pm.Beta("assist_beta", alpha=player_row["ψ_α"], beta=player_row["ψ_β"])
            # Sample from the distributions
            trace = pm.sample(draws=NUM_SAMPLES, tune=BURN_SAMPLES, chains=CHAINS, return_inferencedata=True)
        # Extract samples
        start_sub_unused_dirichlet_samples = trace.posterior["start_sub_unused_dirichlet_dist"].mean(dim=["chain", "draw"]).values
        score_beta_sample = trace.posterior["score_beta"].mean(dim=["chain", "draw"]).values
        assist_beta_sample = trace.posterior["assist_beta"].mean(dim=["chain", "draw"]).values
        # Append the sampled data
        sampled_data.append({
            "name": player_row["name"],
            "start_prob": start_sub_unused_dirichlet_samples[0],
            "sub_prob": start_sub_unused_dirichlet_samples[1],
            "unused_prob": start_sub_unused_dirichlet_samples[2],
            "score_prob": score_beta_sample,
            "assist_prob": assist_beta_sample,
        })
    # Convert the sampled data into a DataFrame and merge it with the original team data frame
    sampled_team_stats = pd.DataFrame(sampled_data)
    team_players_abilities_with_stats = team_players_abilities.merge(right=sampled_team_stats, how="inner", on="name")
    return team_players_abilities_with_stats




def simulate_gameweek(season_start_year: str, gameweek: str) -> pd.DataFrame:
    """
        - This function simulates a gameweek by:
            - Sampling players (from available players) proportionally to their probability to start a match
            - Sample the minute each player leaves the pitch from the global multinomial distributions
            - For substitutions, we randomly select a benched player as a replacement - proportionally to their 
            probability of being subbed
    """
    shortened_season_end_year = str(int(season_start_year) + 1)[2:]
    fixtures_csv_file_path = f"./data/{season_start_year + "-" + shortened_season_end_year}/fixtures.csv"
    fixtures_df = pd.read_csv(filepath_or_buffer=fixtures_csv_file_path)
    fixtures_df = fixtures_df[fixtures_df["GW"] == int(gameweek)] # Filter fixtures for specified gameweek
    player_ability_file_path = f"./data/{season_start_year + "-" + shortened_season_end_year}/player_ability.csv"
    players_ability_df = pd.read_csv(filepath_or_buffer=player_ability_file_path)
    for _, fixture in fixtures_df.iterrows():
        # Sample individual players' stats
        home_team_players_with_stats = sample_team_stats(
            players_ability_df[players_ability_df["team"] == fixture["away_team"]]
        )
        away_team_players_with_stats = sample_team_stats(
            players_ability_df[players_ability_df["team"] == fixture["home_team"]]
        )
        # Sample starting linup for each team
        home_team_starting = home_team_players_with_stats.sample(n=STARTING_PLAYERS, weights="start_prob")
        away_team_starting = away_team_players_with_stats.sample(n=STARTING_PLAYERS, weights="start_prob")
        print(home_team_starting.head(n=10))
        print(away_team_starting.head(n=10))
        break
    return None


if __name__=="__main__":
    simulate_gameweek("2023", "1")

        
        


