import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import pandas as pd
from typing import Dict, List, Set
import os
import subprocess
from pprint import pprint
from constants import (
    PROMOTED_TEAMS_2021_22,
    RELEGATED_TEAMS_2021_22,
    PROMOTED_TEAMS_2022_23,
    RELEGATED_TEAMS_2022_23,
)

# A Premier League season has 380 fixtures; 20 teams
GAMES_PER_GAMEWEEK = 10


class DixonColesModel:
    """
    - Creates a Dixon-Coles prediction model using fixture results from the previous season and
    preceeding gameweeks if the 'gameweek' parameter is passed and is greater than 1
    - Handles season transitions with promoted and relegated teams
    """

    def __init__(self):
        self.previous_params = None  # Store previously calculated parameters

    def _load_results(self, season_start_year: str) -> pd.DataFrame:
        """
        - Ensures that the results file is present before loading and returning it as a DataFrame
        """
        data_folder_name = (
            season_start_year + "-" + str(int(season_start_year[-2:]) + 1)
        )
        fixture_results_path = os.path.abspath(
            f"../data/{data_folder_name}/fixture_results.csv"
        )
        try:
            return pd.read_csv(filepath_or_buffer=fixture_results_path)
        except FileNotFoundError as e:
            # Save the respective fixture results file locally
            relative_path = "../scripts/save_season_results.py"
            absolute_path = os.path.abspath(relative_path)
            try:
                # Ensure the script is executable
                subprocess.run(
                    ["python3", relative_path, season_start_year], check=True
                )
            except subprocess.SubprocessError as e:
                print(
                    f"An error occurred while running the script -> {absolute_path}: {e}"
                )
        try:
            return pd.read_csv(filepath_or_buffer=fixture_results_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to locate the results file at {fixture_results_path} even after attempting to generate it."
            )

    def _rho_correction(self, x: int, y: int, lambda_x: float, mu_y: float, rho: float):
        """
        - Represents the tau function in the Dixon-Coles model that applies a correction to
        low-scoring match results
        - rho parameter controls the strength of the correction
        """
        if x == 0 and y == 0:
            return 1 - (lambda_x * mu_y * rho)
        elif x == 0 and y == 1:
            return 1 + (lambda_x * rho)
        elif x == 1 and y == 0:
            return 1 + (mu_y * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def _dc_log_like(self, x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
        """
        - Constructs the likelihood function that will be maximized via Maximum Likelihood Estimation
        to find the coefficients (parameters) that maximize it
        """
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x)
        epsilon = 1e-10  # Small value to prevent log(0)
        return (
            np.log(self._rho_correction(x, y, lambda_x, mu_y, rho) + epsilon)
            + np.log(poisson.pmf(x, lambda_x) + epsilon)
            + np.log(poisson.pmf(y, mu_y) + epsilon)
        )

    def _get_current_season_teams(self, season_start_year: str) -> Set[str]:
        """
        Get the set of teams participating in the current season

        Args:
            season_start_year: Starting year of the season (e.g., "2022")

        Returns:
            Set of team names
        """
        # Get teams from previous season
        prev_season_start_year = str(int(season_start_year) - 1)
        prev_season_df = self._load_results(prev_season_start_year)
        all_prev_teams = set(prev_season_df["HomeTeam"].unique())

        # Determine promoted and relegated teams for this season
        if season_start_year == "2022":
            promoted_teams = set(PROMOTED_TEAMS_2022_23)
            relegated_teams = set(RELEGATED_TEAMS_2022_23)
        elif season_start_year == "2021":
            promoted_teams = set(PROMOTED_TEAMS_2021_22)
            relegated_teams = set(RELEGATED_TEAMS_2021_22)
        else:
            raise ValueError(
                f"No promoted/relegated team data for season {season_start_year}"
            )

        # Create set of current teams by removing relegated and adding promoted
        current_teams = (all_prev_teams - relegated_teams) | promoted_teams

        return current_teams

    def solve_parameters(
        self, season_start_year: str, gameweek: int = 0
    ) -> Dict[str, float]:
        """
        Find parameters that maximize likelihood function

        Args:
            season_start_year: Starting year of the season (e.g., "2022")
            gameweek: Current gameweek (0 for pre-season)

        Returns:
            Dictionary of parameter values
        """
        # If we've already calculated parameters for this gameweek, return them
        if self.previous_params is not None and gameweek == 0:
            return self.previous_params

        # Get the set of teams for the current season
        current_teams = self._get_current_season_teams(season_start_year)

        # Load fixture results data for previous season
        prev_season_start_year = str(int(season_start_year) - 1)
        prev_season_results_df = self._load_results(prev_season_start_year)

        if gameweek > 0:
            # Filter previous season results to only include current teams
            prev_season_results_df = prev_season_results_df[
                prev_season_results_df["HomeTeam"].isin(current_teams)
                & prev_season_results_df["AwayTeam"].isin(current_teams)
            ]

            # Get current season results up to the previous gameweek
            try:
                current_season_df = self._load_results(season_start_year)
                current_season_results = current_season_df.iloc[
                    : (GAMES_PER_GAMEWEEK * (gameweek - 1)) + 1
                ]

                # Combine previous and current season data
                fixture_results_df = pd.concat(
                    [prev_season_results_df, current_season_results]
                )
            except FileNotFoundError:
                # If current season data not available, use filtered previous season data
                print(
                    f"Warning: No data found for season {season_start_year}. Using only previous season data."
                )
                fixture_results_df = prev_season_results_df
        else:
            # If gameweek is 0, use only previous season data
            fixture_results_df = prev_season_results_df

        # Get unique teams in the dataset
        teams = np.sort(fixture_results_df["HomeTeam"].unique())
        away_teams = np.sort(fixture_results_df["AwayTeam"].unique())

        missing_home, missing_away = None, None
        # Verify that home and away teams match
        if not np.array_equal(teams, away_teams):
            missing_home = set(away_teams) - set(teams)
            missing_away = set(teams) - set(away_teams)
            print(f"Warning: Home and away team sets differ.")
            print(f"Teams in away but not home: {missing_home}")
            print(f"Teams in home but not away: {missing_away}")

            # Filter to common teams to ensure consistent parameters
            common_teams = set(teams).intersection(set(away_teams))
            fixture_results_df = fixture_results_df[
                fixture_results_df["HomeTeam"].isin(common_teams)
                & fixture_results_df["AwayTeam"].isin(common_teams)
            ]
            teams = np.sort(list(common_teams))

        n_teams = len(teams)
        print(f"Fitting model with {n_teams} teams:")
        print(", ".join(teams))

        # Random initialization of model parameters
        init_vals = np.concatenate(
            (
                np.random.uniform(0, 1, (n_teams)),  # attack strength
                np.random.uniform(0, -1, (n_teams)),  # defence strength
                np.array([0, 1.0]),  # rho (score correction), gamma (home advantage)
            )
        )

        def estimate_parameters(params):
            score_coefs = dict(zip(teams, params[:n_teams]))
            defend_coefs = dict(zip(teams, params[n_teams : (2 * n_teams)]))
            rho, gamma = params[-2:]
            log_like = [
                self._dc_log_like(
                    row.HomeGoals,
                    row.AwayGoals,
                    score_coefs[row.HomeTeam],
                    defend_coefs[row.HomeTeam],
                    score_coefs[row.AwayTeam],
                    defend_coefs[row.AwayTeam],
                    rho,
                    gamma,
                )
                for row in fixture_results_df.itertuples()
            ]
            return -sum(log_like)

        # Constraint: sum of attack strengths = n_teams
        constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]

        # Optimize
        opt_output = minimize(
            fun=estimate_parameters,
            x0=init_vals,
            options={"disp": True, "maxiter": 100},
            constraints=constraints,
        )

        # Create parameter dictionary
        params = dict(
            zip(
                ["attack_" + team for team in teams]
                + ["defence_" + team for team in teams]
                + ["rho", "home_adv"],
                opt_output.x,
            )
        )

        # Add parameters for missing teams if needed
        current_teams = self._get_current_season_teams(season_start_year)
        missing_teams = (missing_home or set()).union(
            missing_away or set(), current_teams
        )
        for team in missing_teams:
            if f"attack_{team}" not in params:
                # Calculate median values for imputation
                median_attack = np.median(
                    [v for k, v in params.items() if k.startswith("attack_")]
                )
                median_defence = np.median(
                    [v for k, v in params.items() if k.startswith("defence_")]
                )

                # Impute values for missing team
                params[f"attack_{team}"] = median_attack
                params[f"defence_{team}"] = median_defence
                print(f"Imputed parameters for {team} using median values")

        # Store parameters for future use
        self.previous_params = params.copy()

        return params

    def simulate_match(
        self, homeTeam: str, awayTeam: str, params: Dict[str, float], max_goals=10
    ):
        """
        Simulate a match between two teams

        Args:
            homeTeam: Name of the home team
            awayTeam: Name of the away team
            params: Dictionary of model parameters
            max_goals: Maximum number of goals to consider in the score matrix

        Returns:
            Score probability matrix
        """
        # Check if parameters exist for both teams
        if f"attack_{homeTeam}" not in params or f"attack_{awayTeam}" not in params:
            missing_team = homeTeam if f"attack_{homeTeam}" not in params else awayTeam
            raise ValueError(f"No parameters available for team: {missing_team}")

        def calc_means(param_dict, homeTeam, awayTeam):
            return [
                np.exp(
                    param_dict["attack_" + homeTeam]
                    + param_dict["defence_" + awayTeam]
                    + param_dict["home_adv"]
                ),
                np.exp(
                    param_dict["defence_" + homeTeam] + param_dict["attack_" + awayTeam]
                ),
            ]

        # Calculate expected goals for each team
        team_avgs = calc_means(params, homeTeam, awayTeam)

        # Calculate probability mass functions for goals
        team_pred = [
            [poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)]
            for team_avg in team_avgs
        ]

        # Create score matrix
        output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

        # Apply Dixon-Coles correction for low scores
        correction_matrix = np.array(
            [
                [
                    self._rho_correction(
                        home_goals,
                        away_goals,
                        team_avgs[0],
                        team_avgs[1],
                        params["rho"],
                    )
                    for away_goals in range(2)
                ]
                for home_goals in range(2)
            ]
        )
        output_matrix[:2, :2] = output_matrix[:2, :2] * correction_matrix

        return output_matrix
