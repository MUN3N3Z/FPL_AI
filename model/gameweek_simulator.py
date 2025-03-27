import pandas as pd
import pymc as pm
import numpy as np
from numpy.typing import NDArray
from utils import string_list_to_np_array, id_to_team_converter_2023_24, position_score_points_map, position_clean_sheet_points_map, random_bool
from dixon_coles import DixonColesModel
import os

RANDOM_SEED = 19

NUM_SAMPLES = 5000
BURN_SAMPLES = 500
CHAINS = 4
TARGET_ACCEPT = 0.9

STARTING_PLAYERS = 11
MATCH_MINUTES = 90
MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS = 60
ASSIST_POINTS = 3
HALF_GAME_MINUTES = 45

DATA_FOLDER = "../data"
PLAYER_ABILITY_FILE = "player_ability.csv"
POSITION_MINUTES_FILE = "player_position_minutes.csv"
FIXTURE_FILE = "fixtures.csv"

def sample_player_stats(team_players_abilities: pd.DataFrame) -> pd.DataFrame:
    """
        - Sample each player's probability of starting a game, scoring a goal, and assisting one from their priors
        - Returns the team's players' abilities data frame with additional columns that contain the respective stats
    """
    sampled_data = []
    # Convert the 'ρ_β' column from string to NumPy arrays
    team_players_abilities = team_players_abilities.copy()
    team_players_abilities["ρ_β"] = team_players_abilities["ρ_β"].apply(string_list_to_np_array)
    for _, player_row in team_players_abilities.iterrows():
        with pm.Model() as player_match_model:
            # Define the distributions
            # Enforced a minimum value i.e. 1e-3 for alpha and beta values to ensure they're always positive
            pm.Dirichlet("start_sub_unused_dirichlet_dist", a=player_row["ρ_β"])
            pm.Beta("score_beta", alpha=max(player_row["ω_α"], 1e-3), beta=max(player_row["ω_β"], 1e-3))
            pm.Beta("assist_beta", alpha=max(player_row["ψ_α"], 1e-3), beta=max(player_row["ψ_β"], 1e-3))
            # Sample from the distributions
            trace = pm.sample(
                draws=NUM_SAMPLES, 
                tune=BURN_SAMPLES, 
                chains=CHAINS, 
                return_inferencedata=True, 
                target_accept=TARGET_ACCEPT
            )
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
    position_minutes_df.loc[:,"minutes"] = position_minutes_df["minutes"].apply(string_list_to_np_array) 
    position_minutes_df.index = position_minutes_df["position"]

    for _, player_row in starting_lineup.iterrows():
        with pm.Model():
            pm.Dirichlet(
                "minutes_played", 
                a=position_minutes_df.loc[player_row["position"], "minutes"]
            )
            # Sample from distribution
            trace = pm.sample(
                draws=NUM_SAMPLES, 
                tune=BURN_SAMPLES, 
                chains=CHAINS, 
                return_inferencedata=True, 
                target_accept=TARGET_ACCEPT
            )
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

def score_minutes_played(fixture: pd.Series, players_in_field: pd.DataFrame, home_team_benched: pd.DataFrame, away_team_benched:pd.DataFrame) -> pd.DataFrame:
    """
        - Account for minutes played 
            - 2 pts for playing 60 minutes or more
            - 1 pt for any other minutes played
    """
    players_in_field.index = players_in_field["name"]
    players_in_field["points"] = 0
    subbed_players = []
    for _, player_row in players_in_field.iterrows():
        # Score minutes played
        player_game_score = 0
        if player_row["minutes_played"] < MATCH_MINUTES:
            # Player was subbed
            if player_row["minutes_played"] >= MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS:
                player_game_score += 2
                substitute_player = (
                    home_team_benched.sample(1, weights="sub_prob", replace=False, random_state=RANDOM_SEED)
                    if player_row["team"] == fixture["home_team"]
                    else away_team_benched.sample(1, weights="sub_prob", replace=False, random_state=RANDOM_SEED)
                )
                substitute_player["points"] = 1
                substitute_player["minutes_played"] = MATCH_MINUTES - player_row["minutes_played"] # Assume replacement player is never taken off
            else:
                player_game_score += 1
                substitute_player = (
                    home_team_benched.sample(1, weights="sub_prob", replace=False, random_state=RANDOM_SEED)
                    if player_row["team"] == fixture["home_team"]
                    else away_team_benched.sample(1, weights="sub_prob", replace=False, random_state=RANDOM_SEED)
                )
                if player_row["minutes_played"] < HALF_GAME_MINUTES:
                    substitute_player_minutes = HALF_GAME_MINUTES + (HALF_GAME_MINUTES - player_row["minutes_played"])
                    substitute_player["points"] = 2 if substitute_player_minutes >= MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS else 1
                    substitute_player["minutes_played"] = substitute_player_minutes
                else:
                    substitute_player["points"] = 1
                    substitute_player["minutes_played"] = MATCH_MINUTES - player_row["minutes_played"]
            subbed_players.append(substitute_player)
        else:
            # Player played full match
            player_game_score += 2
        players_in_field.loc[player_row["name"], "points"] = player_game_score

    subbed_players.append(players_in_field)
    players_in_field = pd.concat(subbed_players, axis=0)

    return players_in_field

def score_goals_assists(players_in_field: pd.DataFrame, team: str, team_score:int) -> pd.DataFrame:
    """
        - Assists made - 3 pts for any player who assists
        - Goals scored - 10 pts for GKP, 6 pts for DEF, 5 pts for MID, 4 pts for FWD
    """
    players_in_field.index = players_in_field["name"]
    for _ in range(team_score):
        team_played = players_in_field[players_in_field["team"] == team]
        scorer = team_played.sample(1, weights="score_prob", replace=random_bool()) # Introduce noise?
        scorer_position = scorer.iloc[0]["position"]
        players_in_field.loc[scorer.iloc[0]["name"], "points"] += position_score_points_map(scorer_position)
        # Assume that all goals have attributed assists
        assister = team_played.sample(1, weights="assist_prob", replace=random_bool(), random_state=RANDOM_SEED)
        players_in_field.loc[assister.iloc[0]["name"], "points"] += ASSIST_POINTS
    return players_in_field

def score_clean_sheets(players_in_field: pd.DataFrame, team: str) -> pd.DataFrame:
    """
        - Clean sheet points only awarded to players with 60+ minutes in a game
        - 4 pts for GKP/DEF, 1 pt for MID, 0 for FWD
    """
    players_in_field.index = players_in_field["name"]
    team_players = players_in_field[
        (players_in_field["team"] == team) & 
        (players_in_field["minutes_played"] >= MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS)
    ]
    for _, player_row in team_players.iterrows():
        players_in_field.loc[player_row["name"], "points"] += position_clean_sheet_points_map(player_row["position"])
    return players_in_field

def simulate_fixture(fixture: pd.Series, players_ability_df: pd.DataFrame, position_minutes: pd.DataFrame, match_score_matrix: NDArray) -> pd.DataFrame:
    """ 
        Simulates a single fixture by:
            - Sampling starting lineups and substituted players for both teams.
            - Sampling minutes played for each player.
            - Scoring goals, assists, and clean sheets based on the match score matrix.
        Args:
            fixture (pd.Series): The fixture data for the match.
            players_ability_df (pd.DataFrame): Player ability data.
            position_minutes (pd.DataFrame): Position-specific minutes distribution.
            match_score_matrix (NDArray): Predicted match score probabilities.

        Returns:
            pd.DataFrame: Player (only players who started/were subbed on) points for the fixture
    """
    # Sample individual players' stats
    home_team_players_with_stats = sample_player_stats(
        players_ability_df[players_ability_df["team"] == fixture["away_team"]]
    )
    away_team_players_with_stats = sample_player_stats(
        players_ability_df[players_ability_df["team"] == fixture["home_team"]]
    )
    # Sample starting lineups
    home_team_starting = home_team_players_with_stats.sample(n=STARTING_PLAYERS, weights="start_prob", replace=False)
    away_team_starting = away_team_players_with_stats.sample(n=STARTING_PLAYERS, weights="start_prob", replace=False)
    
    # Sample minutes played for starting lineup
    home_team_starting = sample_players_minutes_played(home_team_starting, position_minutes)
    away_team_starting = sample_players_minutes_played(away_team_starting, position_minutes)
    
    # Benched players 
    home_team_benched = home_team_players_with_stats[~ home_team_players_with_stats["name"].isin(home_team_starting["name"])]
    away_team_benched = away_team_players_with_stats[~ away_team_players_with_stats["name"].isin(away_team_starting["name"])]
    
    home_team_score, _ = np.unravel_index(np.tril(match_score_matrix).argmax(), match_score_matrix.shape)
    _, away_team_score = np.unravel_index(np.triu(match_score_matrix).argmax(), match_score_matrix.shape)
   
    players_in_field = pd.concat([home_team_starting, away_team_starting], axis=0, ignore_index=True).sort_values(by=["minutes_played"], inplace=False, ascending=True)

    # Award points
    players_in_field = score_minutes_played(fixture, players_in_field, home_team_benched, away_team_benched)
    players_in_field = score_goals_assists(players_in_field, fixture["home_team"], home_team_score)
    players_in_field = score_goals_assists(players_in_field, fixture["away_team"], away_team_score)
    if home_team_score == 0:
        players_in_field = score_clean_sheets(players_in_field, fixture["home_team"])
    if away_team_score == 0:
        players_in_field = score_clean_sheets(players_in_field, fixture["away_team"])

    return players_in_field

def simulate_gameweek(season_start_year: str, gameweek: str, fixtures_df: pd.DataFrame, position_minutes_df: pd.DataFrame) -> pd.DataFrame:
    """
        - Concatenate data frames of sampled players and their points for each fixture in a gameweek
        - ONLY includes players who were sampled to start or get substituted into the game

        Args:
            season_start_year (str): The starting year of the season (e.g., "2023").
            gameweek (str): The gameweek number as a string (e.g., "1").
            fixtures_df (pd.DataFrame): A DataFrame containing fixture data
            position_minutes_df (pd.DataFrame): A DataFrame containing position-specific minutes distribution
    """
    # Load data files
    shortened_season_end_year = str(int(season_start_year) + 1)[2:]
    year_folder = f"{season_start_year + "-" + shortened_season_end_year}"
    player_ability_file_path = os.path.join(DATA_FOLDER, year_folder, PLAYER_ABILITY_FILE)
    players_ability_df = pd.read_csv(filepath_or_buffer=player_ability_file_path)

    # Prediction model
    dixon_coles_prediction_model = DixonColesModel()
    dixon_coles_team_parameters = dixon_coles_prediction_model.solve_parameters(season_start_year, int(gameweek))
    fixture_points_df = []
    for _, fixture_row in fixtures_df.iterrows():
        match_score_matrix = dixon_coles_prediction_model.simulate_match(
            homeTeam=id_to_team_converter_2023_24(fixture_row["home_team"]),
            awayTeam=id_to_team_converter_2023_24(fixture_row["away_team"]),
            params=dixon_coles_team_parameters,
        )
        fixture_points = simulate_fixture(fixture_row, players_ability_df, position_minutes_df, match_score_matrix)
        fixture_points_df.append(fixture_points)
    # Concatenate all player points into a single DataFrame
    gameweek_player_points_df = pd.concat(fixture_points_df, ignore_index=True)
    
    return gameweek_player_points_df
