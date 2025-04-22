import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from data_registry import DataRegistry
from utils import format_season_name
from dixon_coles import DixonColesModel
from player_ability import PlayerAbility
from position_minutes import PositionMinutesModel
from gameweek_simulator import GameweekSimulator
from team_optimizer import generate_multiple_teams


class FPLEnv(gym.Env):
    """
    Fantasy Premier League Environment for Reinforcement Learning
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        season_start_year: str,
        fixtures_data: pd.DataFrame,
        total_gameweeks: int = 38,
        current_gameweek: int = 1,
        budget: float = 100.0,
        team_size: int = 15,
        starting_size: int = 11,
        free_transfers_per_gw: int = 1,
        max_transfers: int = 2,  # Maximum accumulated free transfers
        transfer_penalty: int = 4,  # Points deducted for each additional transfer
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the FPL environment

        Args:
            season_start_year: Starting year of the season (e.g., "2022")
            fixtures_data: DataFrame containing fixture information for all gameweeks
            players_ability_data: DataFrame containing player abilities data
            position_minutes_data: DataFrame with position-specific minutes distribution
            total_gameweeks: Total number of gameweeks in the season
            current_gameweek: Starting gameweek
            budget: Total budget available
            team_size: Number of players in a team (15 in FPL)
            starting_size: Number of starting players (11 in FPL)
            free_transfers_per_gw: Number of free transfers allowed per gameweek
            max_transfers: Maximum accumulated free transfers
            transfer_penalty: Points deducted for each additional transfer
            render_mode: Rendering mode
        """
        super().__init__()

        self.render_mode = render_mode
        self.season_start_year = season_start_year
        self.fixtures_data = fixtures_data
        self.total_gameweeks = total_gameweeks
        self.current_gameweek = current_gameweek
        self.budget = budget
        self.team_size = team_size
        self.starting_size = starting_size
        self.free_transfers_per_gw = free_transfers_per_gw
        self.max_transfers = max_transfers
        self.transfer_penalty = transfer_penalty

        # Initialize simulator for getting player points
        self.gameweek_simulator = GameweekSimulator(season_start_year)

        # Bayesian model for players' abilities
        self.players_ability_model = PlayerAbility(season_start_year, current_gameweek)
        # Bayesian model for players' minutes based on their position
        self.position_minutes_model = PositionMinutesModel()
        # Initialize available free transfers
        self.available_transfers = free_transfers_per_gw

        # Initialize team and points
        self.current_team = None
        self.current_lineup = None
        self.captain_id = None
        self.vice_captain_id = None
        self.total_points = 0
        self.gameweek_points = 0

        # Define observation space
        players_ability_data = self.players_ability_model.player_ability
        self.observation_space = spaces.Dict(
            {
                "player_beliefs": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        len(players_ability_data),
                        7,
                    ),  # 7 belief parameters per player
                    dtype=np.float32,
                ),
                "current_team": spaces.MultiBinary(len(players_ability_data)),
                "current_lineup": spaces.MultiBinary(len(players_ability_data)),
                "captain": spaces.Discrete(
                    len(players_ability_data) + 1
                ),  # +1 for no captain
                "vice_captain": spaces.Discrete(
                    len(players_ability_data) + 1
                ),  # +1 for no vice captain
                "available_transfers": spaces.Discrete(max_transfers + 1),
                "current_gameweek": spaces.Discrete(total_gameweeks + 1),
                "budget": spaces.Box(
                    low=0, high=float(100), shape=(1,), dtype=np.float32
                ),
            }
        )

        # Action space (for Bayesian Q-learning, we'll generate actions using MKP)
        # For the gym interface, we'll define a simple action space that can be mapped to teams
        # The agent will select from a set of candidate actions
        self.action_space = spaces.Discrete(3)

        # Action generation cache
        self.candidate_actions = []

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to the beginning of a new episode

        Args:
            seed: Random seed
            options: Additional options for reset

        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Initialize the RNG
        super().reset(seed=seed)

        # Reset gameweek and points
        if options and "gameweek" in options:
            self.current_gameweek = options["gameweek"]
        else:
            self.current_gameweek = 1

        self.total_points = 0
        self.gameweek_points = 0
        self.available_transfers = self.free_transfers_per_gw

        # Initialize team using MKP if not provided
        if options and "initial_team" in options:
            self.current_team = options["initial_team"]
            self.current_lineup = options["initial_lineup"]
            self.captain_id = options["captain"]
            self.vice_captain_id = options["vice_captain"]
        else:
            # Generate player points for the first gameweek
            fixture_gw = self.fixtures_data[
                self.fixtures_data["GW"] == self.current_gameweek
            ]
            player_points_gw = self._simulate_gameweek_points(fixture_gw)

            # Initialize team using MKP approach
            teams = generate_multiple_teams(
                player_df=player_points_gw,
                num_teams=3,
                samples_per_team=20,
                budget=self.budget,
            )

            if teams:
                best_team = teams[0]
                self.current_team = best_team["squad"]
                self.current_lineup = best_team["lineup"]
                self.captain_id = best_team["captain"]
                self.vice_captain_id = best_team["vice_captain"]
            else:
                raise ValueError("Could not generate a valid initial team")

        # Get current observation
        observation = self._get_observation()

        # Additional info
        info = {
            "gameweek": self.current_gameweek,
            "total_points": self.total_points,
            "available_transfers": self.available_transfers,
        }

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment by selecting a team for the next gameweek

        Args:
            action: Index of the candidate team to select (0, 1, or 2)

        Returns:
            observation: New observation after taking the action
            reward: Reward obtained from the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        if not self.candidate_actions:
            # Generate candidate actions if none exist
            self._generate_candidate_actions()

        # Check if action is valid
        if action < 0 or action >= len(self.candidate_actions):
            # Invalid action, return a large negative reward
            reward = -1000
            observation = self._get_observation()
            terminated = False
            truncated = False
            info = {"error": "Invalid action index"}
            return observation, reward, terminated, truncated, info

        # Get selected team
        selected_team = self.candidate_actions[action]

        # Calculate transfers made
        transfers_made = 0
        if self.current_team:
            players_out = [
                p for p in self.current_team if p not in selected_team["squad"]
            ]
            transfers_made = len(players_out)

        # Calculate transfer penalties
        transfer_penalty = 0
        extra_transfers = max(0, transfers_made - self.available_transfers)
        if extra_transfers > 0:
            transfer_penalty = extra_transfers * self.transfer_penalty

        # Update team
        self.current_team = selected_team["squad"]
        self.current_lineup = selected_team["lineup"]
        self.captain_id = selected_team["captain"]
        self.vice_captain_id = selected_team["vice_captain"]

        # Move to next gameweek
        self.current_gameweek += 1

        # Calculate actual points for the team using historical data
        actual_points = self._calculate_actual_points()

        # Subtract transfer penalty
        actual_points -= transfer_penalty

        # Update total points
        self.gameweek_points = actual_points
        self.total_points += actual_points

        # Update available transfers for next gameweek
        if transfers_made >= self.available_transfers:
            # Used all available transfers, reset to 1
            self.available_transfers = self.free_transfers_per_gw
        else:
            # Carry over unused transfer (max 2)
            self.available_transfers = min(
                self.max_transfers,
                self.available_transfers - transfers_made + self.free_transfers_per_gw,
            )

        # Update player ability beliefs based on actual performance
        self.players_ability_model.update_model()

        # Clear candidate actions for next step
        self.candidate_actions = []

        # Check if season is complete
        done = self.current_gameweek > self.total_gameweeks

        # Get new observation
        observation = self._get_observation()

        # Prepare info dictionary
        info = {
            "gameweek": self.current_gameweek,
            "gameweek_points": self.gameweek_points,
            "total_points": self.total_points,
            "transfers_made": transfers_made,
            "transfer_penalty": transfer_penalty,
            "available_transfers": self.available_transfers,
        }

        return observation, actual_points, done, False, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construct the observation from the current state

        Returns:
            observation: Dictionary containing observation data
        """
        # Construct the observation dictionary
        return {
            "current_team": self.current_team,
            "current_lineup": self.current_lineup,
            "captain": self.captain_id,
            "vice_captain": self.vice_captain_id,
            "available_transfers": self.available_transfers,
            "current_gameweek": self.current_gameweek,
            "budget": np.array([self.budget], dtype=np.float32),
        }

    def _simulate_gameweek_points(self, fixtures_gw: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate player points for a gameweek

        Args:
            fixtures_gw: DataFrame containing fixtures for the gameweek

        Returns:
            player_points: DataFrame with simulated points for each player
        """
        # Use GameweekSimulator to get points
        player_points = self.gameweek_simulator.simulate_gameweek(
            gameweek=str(self.current_gameweek),
            fixtures_df=fixtures_gw,
            position_minutes_df=self.position_minutes_model.model_df,
            players_ability_df=self.players_ability_model.player_ability,
        )

        return player_points

    def _calculate_actual_points(self) -> int:
        """
        Calculate the actual points scored by the team for the current gameweek
        using the real historical data

        Returns:
            points: Total points scored by the team
        """
        # Add points for the rest of the starting lineup
        player_abilities = self.players_ability_model.player_ability
        total_points = 0
        for player_name in self.current_lineup:
            player_points = player_abilities.loc[player_name, "real_points"]
            # Skip captain if we've already counted them
            if player_name == self.captain_id or (
                not self.captain_id and player_name == self.vice_captain_id
            ):
                total_points += player_points * 2
            else:
                total_points += player_points
        print(f"Actual team points: {total_points}")
        return total_points

    def _generate_candidate_actions(self) -> None:
        """
        Generate candidate team selections using MKP approach

        """
        # Get fixtures for the next gameweek
        next_gameweek = self.current_gameweek
        fixtures_gw = self.fixtures_data[self.fixtures_data["GW"] == next_gameweek]

        # Simulate player points for the next gameweek
        player_points_df = self._simulate_gameweek_points(fixtures_gw)

        # Generate multiple teams using MKP
        teams = generate_multiple_teams(
            player_df=player_points_df,
            num_teams=3,
            samples_per_team=20,
            budget=self.budget,
            current_team=self.current_team,
            free_transfers=self.available_transfers,
        )

        # Store the candidate actions
        self.candidate_actions = teams

    def render(self) -> None:
        """
        Render the current state of the environment
        """
        if self.render_mode == "human":
            print(f"\nGameweek: {self.current_gameweek}")
            print(f"Total Points: {self.total_points}")
            print(f"Last Gameweek Points: {self.gameweek_points}")
            print(f"Available Transfers: {self.available_transfers}")

            if self.current_team:
                print("\nCurrent Team:")
                for player_name in self.current_team:
                    player_data = self.players_ability_data[
                        self.players_ability_data["name"] == player_name
                    ]
                    captain_mark = (
                        " (C)"
                        if player_name == self.captain_id
                        else " (V)"
                        if player_name == self.vice_captain_id
                        else ""
                    )
                    lineup_mark = (
                        " (Starting)"
                        if player_name in self.current_lineup
                        else " (Bench)"
                    )
                    print(f"  {player_name}{captain_mark}{lineup_mark}")
