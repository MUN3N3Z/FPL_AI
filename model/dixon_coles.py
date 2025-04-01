import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import pandas as pd
from typing import Dict
import os
import subprocess
from functools import lru_cache
from typing import Tuple
import pickle

# A Premier League season has 380 fixtures; 20 teams
GAMES_PER_GAMEWEEK = 10

class DixonColesModel:
    """
        - Creates a Dixon-Coles prediction model using fixture results from the previous season and
        preceeding gameweeks if the 'gamweek' parameter is passed and is greater than 1
    """
    def __init__(self):
        pass

    def _load_results(self, season_start_year: str) -> pd.DataFrame:
        """
            - Ensures that the results file is present before loading and returning it as a DataFrame
        """
        data_folder_name = season_start_year + "-" + str(int(season_start_year[-2:]) + 1)
        fixture_results_path = os.path.abspath(f"../data/{data_folder_name}/fixture_results.csv")
        try:
            return pd.read_csv(filepath_or_buffer=fixture_results_path)
        except FileNotFoundError as e:
            # Save the respective fixture results file locally
            relative_path = "save_season_results.py"
            absolute_path = os.path.abspath(relative_path)
            try:
                # # Ensure the script is executable
                subprocess.run(["python3", absolute_path, season_start_year], check=True)
            except subprocess.SubprocessError as e:
                print(f"An error occurred while running the script -> {absolute_path}: {e}")
        try:
            return pd.read_csv(filepath_or_buffer=fixture_results_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Failed to locate the results file at {fixture_results_path} even after attempting to generate it.")
    
    def _rho_correction(self, x: int, y: int, lambda_x: float, mu_y: float, rho: float):
        """
            - Represents the tau function in the Dixon-Coles model that applies a correction to 
            low-scoring match results
            - rho parameter controls the strength of the correction
        """
        if x==0 and y==0:
            return 1- (lambda_x * mu_y * rho)
        elif x==0 and y==1:
            return 1 + (lambda_x * rho)
        elif x==1 and y==0:
            return 1 + (mu_y * rho)
        elif x==1 and y==1:
            return 1 - rho
        else:
            return 1.0
        
    def _dc_log_like(self, x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
        """
            - Constructs the likelihood function that will be maximized via Maximum Likelihood Estimation
            to find the coeficients (parameters) that maximize it
        """
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return (np.log(self._rho_correction(x, y, lambda_x, mu_y, rho)) + 
                np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))

    @lru_cache(maxsize=None)
    def solve_parameters(self, season_start_year: str, gameweek: int = 0) -> Dict[str, float]:
        """
            - This function employs scipy's minimize optimization function to find the parameters that maximize the 
            likelihood function described in 'self._dc_log_like'
        """
        # Load fixture results data for previous season
        fixture_results_df = self._load_results(str(int(season_start_year) - 1))
        if gameweek > 1:
            # Retrieve current season's results till the previous gameweek
            current_season_results = self._load_results(season_start_year).iloc[:(GAMES_PER_GAMEWEEK * (gameweek - 1))]
            fixture_results_df = pd.concat([fixture_results_df, current_season_results])

        teams = np.sort(fixture_results_df['HomeTeam'].unique())
        # check for no weirdness in fixture_results_df
        away_teams = np.sort(fixture_results_df['AwayTeam'].unique())
        if not np.array_equal(teams, away_teams):
            raise ValueError("Something's not right")
        n_teams = len(teams)
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0,1,(n_teams)), # attack strength
                                    np.random.uniform(0,-1,(n_teams)), # defence strength
                                    np.array([0, 1.0]) # rho (score correction), gamma (home advantage)
        ))

        def estimate_paramters(params):
            score_coefs = dict(zip(teams, params[:n_teams]))
            defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
            rho, gamma = params[-2:]
            log_like = [self._dc_log_like(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                        score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam], rho, gamma) for row in fixture_results_df.itertuples()]
            return -sum(log_like)
        opt_output = minimize(
            fun=estimate_paramters, 
            x0=init_vals, 
            options={
            'disp': True, 
            'maxiter': 100
            }, 
            constraints = [{
            'type': 'eq', 
            'fun': lambda x: sum(x[:20]) - 20
            }]  
        )
        return dict(zip(["attack_"+team for team in teams] + 
                        ["defence_"+team for team in teams] +
                        ['rho', 'home_adv'],
                        opt_output.x))

    def simulate_match(self, homeTeam: str, awayTeam: str, params: Dict[str, float], max_goals=10):
        """
            - This function ties the Dixon-Coles prediction model together and returns the match score matrix based on the model parameters
            - Memoize the output matrix to prevent recalculation of parameters that maximize the likelihood function
        """
        def calc_means(param_dict, homeTeam, awayTeam):
            return [np.exp(param_dict['attack_'+homeTeam] + param_dict['defence_'+awayTeam] + param_dict['home_adv']),
                    np.exp(param_dict['defence_'+homeTeam] + param_dict['attack_'+awayTeam])]
        
        team_avgs = calc_means(params, homeTeam, awayTeam)
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
        output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        correction_matrix = np.array([[self._rho_correction(home_goals, away_goals, team_avgs[0],
                                                    team_avgs[1], params['rho']) for away_goals in range(2)]
                                    for home_goals in range(2)])
        output_matrix[:2,:2] = output_matrix[:2,:2] * correction_matrix
        print(DixonColesModel.solve_parameters.cache_info())
        return output_matrix
