import pandas as pd
import pymc as pm
import numpy as np
from numpy.typing import NDArray
import os
from constants import (
    NUM_SAMPLES,
    MATCH_MINUTES,
    MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS,
    HALF_GAME_MINUTES,
    RANDOM_SEED,
    DATA_FOLDER,
    ASSIST_POINTS,
)
from dixon_coles import DixonColesModel
from utils import (
    string_list_to_np_array,
    id_to_team_converter_2023_24,
    position_score_points_map,
    position_clean_sheet_points_map,
    random_bool,
    format_season_name,
    choose_formation,
)
import pickle


class GameweekSimulator:
    def __init__(self):
        pass

    def simulate_gameweek(
        self,
        season_start_year: str,
        gameweek: str,
        fixtures_df: pd.DataFrame,
        position_minutes_df: pd.DataFrame,
        players_ability_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate a gameweek by sampling player stats, minutes played, and scoring points.

        Args:
            season_start_year (str): The starting year of the season (e.g., "2023").
            gameweek (str): The gameweek number as a string (e.g., "1").
            fixtures_df (pd.DataFrame): A DataFrame containing fixture data.
            position_minutes_df (pd.DataFrame): A DataFrame containing position-specific minutes distribution.

        Returns:
            pd.DataFrame: Player points for the gameweek.
        """
        season_folder_name = format_season_name(season_start_year)

        dixon_coles_prediction_model = DixonColesModel()

        # Load parameters if present
        gameweek_parameters_file = os.path.join(
            DATA_FOLDER, season_folder_name, "model_data", f"parameters_{gameweek}.pkl"
        )
        if os.path.exists(gameweek_parameters_file):
            with open(file=gameweek_parameters_file, mode="rb") as f:
                dixon_coles_team_parameters = pickle.load(f)
        else:
            dixon_coles_team_parameters = dixon_coles_prediction_model.solve_parameters(
                season_start_year, int(gameweek)
            )
            with open(file=gameweek_parameters_file, mode="wb") as f:
                pickle.dump(obj=dixon_coles_team_parameters, file=f)

        fixture_points_df = []
        for _, fixture_row in fixtures_df.iterrows():
            match_score_matrix = dixon_coles_prediction_model.simulate_match(
                homeTeam=id_to_team_converter_2023_24(fixture_row["home_team"]),
                awayTeam=id_to_team_converter_2023_24(fixture_row["away_team"]),
                params=dixon_coles_team_parameters,
            )
            fixture_points = self._simulate_fixture(
                fixture_row, players_ability_df, position_minutes_df, match_score_matrix
            )
            fixture_points_df.append(fixture_points)

        gameweek_player_points_df = pd.concat(fixture_points_df)
        return gameweek_player_points_df

    def _simulate_fixture(
        self,
        fixture: pd.Series,
        players_ability_df: pd.DataFrame,
        position_minutes: pd.DataFrame,
        match_score_matrix: NDArray,
    ) -> pd.DataFrame:
        """
        Simulates a single fixture by sampling starting lineups, minutes played, and scoring points.
        """
        # Sample individual players' stats
        players_with_stats = self._sample_player_stats(players_ability_df)

        home_team = players_with_stats[
            players_with_stats["team"] == fixture["home_team"]
        ]
        away_team = players_with_stats[
            players_with_stats["team"] == fixture["away_team"]
        ]

        # Sample starting lineups
        home_team_starting = self._sample_starting_lineup(
            home_team, fixture["home_team"]
        )
        away_team_starting = self._sample_starting_lineup(
            away_team, fixture["away_team"]
        )

        # Sample minutes played for starting lineup players - both teams at once
        starting_players_home_away_minutes = pd.concat(
            [home_team_starting, away_team_starting], axis=0
        )
        starting_players_home_away_minutes = self._sample_players_minutes_played(
            starting_players_home_away_minutes, position_minutes
        )

        # Benched players
        home_team_benched = home_team.loc[
            ~home_team["name"].isin(home_team_starting["name"])
        ]
        away_team_benched = away_team.loc[
            ~away_team["name"].isin(away_team_starting["name"])
        ]

        home_score_max, away_score_max = match_score_matrix.shape
        flat_indices = np.random.choice(
            a=(home_score_max * away_score_max),
            p=(match_score_matrix.flatten() / np.sum(match_score_matrix)),
        )
        home_team_score, away_team_score = np.unravel_index(
            indices=flat_indices, shape=match_score_matrix.shape
        )

        starting_players_home_away_minutes.sort_values(
            by=["minutes_played"], ascending=True, inplace=True
        )

        # Award points
        starting_players_home_away_minutes["sampled_points"] = 0
        players_in_field = self._score_minutes_played(
            fixture,
            starting_players_home_away_minutes,
            home_team_benched,
            away_team_benched,
        )
        players_in_field = self._score_goals_assists(
            players_in_field, fixture["home_team"], home_team_score
        )
        players_in_field = self._score_goals_assists(
            players_in_field, fixture["away_team"], away_team_score
        )
        if home_team_score == 0:
            players_in_field = self._score_clean_sheets(
                players_in_field, fixture["home_team"]
            )
        if away_team_score == 0:
            players_in_field = self._score_clean_sheets(
                players_in_field, fixture["away_team"]
            )

        return players_in_field

    def _sample_starting_lineup(self, team: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """Choose 11 starting players using a random popular EPL formation"""
        formation = choose_formation(team_id)
        starting_lineup = [
            team[team["position"] == position].sample(
                n=num_players, weights="start_prob", replace=False
            )
            for position, num_players in formation.items()
        ]
        starting_lineup_df = pd.concat(starting_lineup, axis=0)
        return starting_lineup_df

    def _sample_player_stats(
        self, team_players_abilities: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Sample each player's probability of starting, scoring, and assisting.
        """
        team_players_abilities.loc[:, "ρ_β"] = team_players_abilities.ρ_β.apply(
            func=lambda np_str_array: string_list_to_np_array(np_str_array)
        )
        with pm.Model():
            # convert players' abilities from np objects to arrays for Tensor sampling
            dirichlet_alphas = np.array(
                [array for array in team_players_abilities["ρ_β"].values]
            )
            # Enforced a minimum value i.e. 1e-3 for alpha and beta values to ensure they're always positive
            score_alphas = np.maximum(
                team_players_abilities["ω_α"].values.astype(np.float64), 1e-3
            )
            score_betas = np.maximum(
                team_players_abilities["ω_β"].values.astype(np.float64), 1e-3
            )
            assist_alphas = np.maximum(
                team_players_abilities["ψ_α"].values.astype(np.float64), 1e-3
            )
            assist_betas = np.maximum(
                team_players_abilities["ψ_β"].values.astype(np.float64), 1e-3
            )
            # Define the distributions
            pm.Dirichlet(
                "start_sub_unused_dirichlet_dist",
                a=dirichlet_alphas.astype(np.float64),
                shape=(len(team_players_abilities), 3),
            )
            pm.Beta(
                "score_beta",
                alpha=score_alphas,
                beta=score_betas,
                shape=len(team_players_abilities),
            )
            pm.Beta(
                "assist_beta",
                alpha=assist_alphas,
                beta=assist_betas,
                shape=len(team_players_abilities),
            )
            # Sample from the distributions
            prior_samples = pm.sample_prior_predictive(samples=NUM_SAMPLES)
            # Extract samples
            start_sub_unused_dirichlet_samples = (
                prior_samples.prior["start_sub_unused_dirichlet_dist"]
                .mean(dim=["chain", "draw"])
                .values
            )
            score_beta_samples = (
                prior_samples.prior["score_beta"].mean(dim=["chain", "draw"]).values
            )
            assist_beta_samples = (
                prior_samples.prior["assist_beta"].mean(dim=["chain", "draw"]).values
            )
        # Append the sampled data
        sampled_data = pd.DataFrame(
            {
                "name": team_players_abilities["name"].values,
                "" "start_prob": start_sub_unused_dirichlet_samples[:, 0],
                "sub_prob": start_sub_unused_dirichlet_samples[:, 1],
                "unused_prob": start_sub_unused_dirichlet_samples[:, 2],
                "score_prob": score_beta_samples,
                "assist_prob": assist_beta_samples,
            }
        )
        team_players_abilities_with_stats = team_players_abilities.merge(
            right=sampled_data, how="inner", on="name"
        )
        return team_players_abilities_with_stats

    def _sample_players_minutes_played(
        self, starting_lineup: pd.DataFrame, position_minutes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Sample the minute at which each player leaves the pitch.
        """
        position_minutes_df.index = position_minutes_df["position"]
        player_minutes_alphas = np.array(
            [
                position_minutes_df.loc[position, "minutes"]
                for position in starting_lineup["position"].values
            ]
        )

        with pm.Model():
            minutes_probs = pm.Dirichlet(
                "minutes_probs",
                a=player_minutes_alphas,
                shape=(len(player_minutes_alphas), 91),
            )
            minutes_played = pm.Categorical(
                "minutes_played", p=minutes_probs, shape=len(starting_lineup)
            )
            # Sample from distribution
            prior_sample = pm.sample_prior_predictive(samples=NUM_SAMPLES)
        # Extract sample
        sampled_minutes_played = (
            prior_sample.prior["minutes_played"]
            .mean(dim=["chain", "draw"])
            .values.astype(int)
        )
        # Append the sampled data
        sampled_data = pd.DataFrame(
            {
                "name": starting_lineup["name"].values,
                "minutes_played": sampled_minutes_played,
            }
        )
        starting_lineup_with_minutes_played = starting_lineup.merge(
            right=sampled_data, how="inner", on="name"
        )
        return starting_lineup_with_minutes_played

    def _score_minutes_played(
        self,
        fixture: pd.Series,
        players_in_field: pd.DataFrame,
        home_team_benched: pd.DataFrame,
        away_team_benched: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score points based on minutes played.
        """
        players_in_field.index = players_in_field["name"]
        subbed_players = []
        for _, player_row in players_in_field.iterrows():
            # Score minutes played
            player_game_score = 0
            if player_row["minutes_played"] < MATCH_MINUTES:
                # Player was subbed
                if (
                    player_row["minutes_played"]
                    >= MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS
                ):
                    player_game_score += 2
                    if player_row["team"] == fixture["home_team"]:
                        substitute_player = home_team_benched.sample(
                            1,
                            weights="sub_prob",
                            replace=False,
                            random_state=RANDOM_SEED,
                        )
                        home_team_benched = home_team_benched[
                            ~home_team_benched["name"].isin(substitute_player["name"])
                        ]
                    else:
                        substitute_player = away_team_benched.sample(
                            1,
                            weights="sub_prob",
                            replace=False,
                            random_state=RANDOM_SEED,
                        )
                        away_team_benched = away_team_benched[
                            ~away_team_benched["name"].isin(substitute_player["name"])
                        ]
                    substitute_player["sampled_points"] = 1
                    substitute_player["minutes_played"] = (
                        MATCH_MINUTES - player_row["minutes_played"]
                    )  # Assume replacement player is never taken off
                else:
                    player_game_score += 1
                    if player_row["team"] == fixture["home_team"]:
                        substitute_player = home_team_benched.sample(
                            1,
                            weights="sub_prob",
                            replace=False,
                            random_state=RANDOM_SEED,
                        )
                        home_team_benched = home_team_benched[
                            ~home_team_benched["name"].isin(substitute_player["name"])
                        ]
                    else:
                        substitute_player = away_team_benched.sample(
                            1,
                            weights="sub_prob",
                            replace=False,
                            random_state=RANDOM_SEED,
                        )
                        away_team_benched = away_team_benched[
                            ~away_team_benched["name"].isin(substitute_player["name"])
                        ]
                    # Determine if sub player is eligible for full playing points (> 60 minutes)
                    if player_row["minutes_played"] < HALF_GAME_MINUTES:
                        substitute_player_minutes = HALF_GAME_MINUTES + (
                            HALF_GAME_MINUTES - player_row["minutes_played"]
                        )
                        substitute_player["sampled_points"] = (
                            2
                            if substitute_player_minutes
                            >= MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS
                            else 1
                        )
                        substitute_player["minutes_played"] = substitute_player_minutes
                    else:
                        substitute_player["sampled_points"] = 1
                        substitute_player["minutes_played"] = (
                            MATCH_MINUTES - player_row["minutes_played"]
                        )
                subbed_players.append(substitute_player)
            else:
                # Player played full match
                player_game_score += 2
            players_in_field.loc[
                player_row["name"], "sampled_points"
            ] = player_game_score

        subbed_players.append(players_in_field)
        players_in_field = pd.concat(subbed_players, axis=0)

        return players_in_field

    def _score_goals_assists(
        self, players_in_field: pd.DataFrame, team: str, team_score: int
    ) -> pd.DataFrame:
        """
        Score points for goals and assists.
        """
        players_in_field.index = players_in_field["name"]
        scoring_rate_per_minute = (
            -np.log(1 - players_in_field.score_prob) / MATCH_MINUTES
        )
        assisting_rate_per_minute = (
            -np.log(1 - players_in_field.assist_prob) / MATCH_MINUTES
        )
        players_in_field["weighted_score_prob"] = 1 - np.exp(
            -scoring_rate_per_minute * players_in_field.minutes_played
        )
        players_in_field["weighted_assist_prob"] = 1 - np.exp(
            -assisting_rate_per_minute * players_in_field.minutes_played
        )
        # Two copies for removing sampled players by assist or goals separately
        assists_copy = players_in_field[players_in_field["team"] == team]
        scoring_copy = assists_copy.copy()
        for _ in range(team_score):
            replace_scorer = random_bool()
            scorer = scoring_copy.sample(
                1,
                weights="weighted_score_prob",
                replace=replace_scorer,
                random_state=RANDOM_SEED,
            )  # Introduce noise?
            scorer_position = scorer.iloc[0]["position"]
            players_in_field.loc[
                scorer.iloc[0]["name"], "sampled_points"
            ] += position_score_points_map(scorer_position)
            # Randomly keep/remove scorer from scoring sample
            if replace_scorer:
                scoring_copy = scoring_copy[scoring_copy.name.isin(scorer.name)]

            replace_assister = random_bool()
            # Assume that all goals have attributed assists
            assister = assists_copy.sample(
                1,
                weights="weighted_assist_prob",
                replace=replace_assister,
                random_state=RANDOM_SEED,
            )
            players_in_field.loc[
                assister.iloc[0]["name"], "sampled_points"
            ] += ASSIST_POINTS
            # Randomly keep/remove scorer from scoring sample
            if replace_assister:
                assists_copy = assists_copy[assists_copy.name.isin(scorer.name)]
        return players_in_field

    def _score_clean_sheets(
        self, players_in_field: pd.DataFrame, team: str
    ) -> pd.DataFrame:
        """
        Score points for clean sheets.
        """
        players_in_field.index = players_in_field["name"]
        team_players = players_in_field[
            (players_in_field["team"] == team)
            & (
                players_in_field["minutes_played"]
                >= MINUTES_THRESHOLD_FOR_FULL_PARTICIPATION_POINTS
            )
        ]
        for _, player_row in team_players.iterrows():
            players_in_field.loc[
                player_row["name"], "sampled_points"
            ] += position_clean_sheet_points_map(player_row["position"])
        return players_in_field
