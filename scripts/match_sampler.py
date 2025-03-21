from fpl.models import Fixture, Player
from typing import List
import pandas as pd
import pymc as pm
import numpy as np
from utils import string_list_to_np_array

NUM_SAMPLES = 5000
BURN_SAMPLES = 1000
CHAINS = 4

STARTING_PLAYERS = 11

def sample_player_stats(team_players_abilities: pd.DataFrame) -> pd.DataFrame:
    """
        - Sample each player's probability of starting a game, scoring a goal, and assisting one from their priors
        - Returns the team's players' abilities data frame with additional columns that contain the respective stats
    """
    sampled_data = []
    # Convert the 'ρ_β' column from string to NumPy arrays
    team_players_abilities["ρ_β"] = team_players_abilities["ρ_β"].apply(string_list_to_np_array)
    for _, player_row in team_players_abilities.iterrows():
        with pm.Model() as player_match_model:
            # print(f"start, sub, unused: {player_row["ρ_β"]}")
            # print(f"score: alpha - {player_row["ω_α"]}; beta {player_row["ω_β"]}")
            # print(f"assist: alpha - {player_row["ψ_α"]}; beta {player_row["ψ_β"]}")
            
            # Define the distributions
            # Enforced a minimum value i.e. 1e-3 for alpha and beta values to ensure they're always positive
            pm.Dirichlet("start_sub_unused_dirichlet_dist", a=player_row["ρ_β"])
            pm.Beta("score_beta", alpha=max(player_row["ω_α"], 1e-3), beta=max(player_row["ω_β"], 1e-3))
            pm.Beta("assist_beta", alpha=max(player_row["ψ_α"], 1e-3), beta=max(player_row["ψ_β"], 1e-3))
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

def sample_players_minutes_played(starting_lineup: pd.DataFrame, position_minutes_df: pd.DataFrame) -> pd.DataFrame:
    """
        - Sample the minute at which each player in the starting lineup leaves the pitch from the global Dirichlet distribution 'position_minutes_df'
        - Returns a pd.DataFrame with a new "minutes_played" column
    """
    sampled_data = []
    # Convert the 'minutes' column from string to NumPy arrays
    position_minutes_df["minutes"] = position_minutes_df["minutes"].apply(string_list_to_np_array) 

    for _, player_row in starting_lineup.iterrows():
        with pm.Model():
            pm.Dirichlet(
                "minutes_played", 
                a=position_minutes_df.loc[player_row["position"], "minutes"]
            )
            # Sample from distribution
            trace = pm.sample(draws=NUM_SAMPLES, tune=BURN_SAMPLES, chains=CHAINS, return_inferencedata=True)
        # Extract sample
        minutes_played = trace.posterior["minutes_played"].mean(dim=["chain", "draw"]).values
        # Append the sampled data
        sampled_data.append({
            "name": player_row["name"],
            "minutes_played": minutes_played
        })
    sampled_minutes_played = pd.DataFrame(sampled_data)
    starting_lineup_with_minutes_played = starting_lineup.merge(right=sampled_minutes_played, how="inner", on="name")
    return starting_lineup_with_minutes_played


def simulate_fixture(fixture: pd.Series, players_ability_df: pd.DataFrame, position_minutes: pd.DataFrame) -> pd.DataFrame:
    """ 
        - 
    """
    # Sample individual players' stats
    home_team_players_with_stats = sample_player_stats(
        players_ability_df[players_ability_df["team"] == fixture["away_team"]]
    )
    away_team_players_with_stats = sample_player_stats(
        players_ability_df[players_ability_df["team"] == fixture["home_team"]]
    )
    # Sample starting lineup for each team
    home_team_starting = home_team_players_with_stats.sample(n=STARTING_PLAYERS, weights="start_prob")
    away_team_starting = away_team_players_with_stats.sample(n=STARTING_PLAYERS, weights="start_prob")
    # Sample minutes played for starting lineup
    home_team_starting = sample_players_minutes_played(home_team_starting, position_minutes)
    away_team_starting = sample_players_minutes_played(away_team_starting, position_minutes)
    
    return None

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

    position_minutes_file_path = "./data/player_position_minutes.csv"
    position_minutes_df = pd.read_csv(filepath_or_buffer=position_minutes_file_path)
    
    for _, fixture_row in fixtures_df.iterrows():
        simulate_fixture(fixture_row, players_ability_df, position_minutes_df)
        break
    return None


if __name__=="__main__":
    simulate_gameweek("2023", "1")

        
        


