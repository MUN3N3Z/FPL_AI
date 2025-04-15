import gymnasium as gym
import pandas as pd
from constants import (
    POSITIONS,
    NUM_PLAYERS_IN_FPL,
    GAMEWEEK_COUNT,
    RANDOM_SEED,
    TEAM_COUNT_LIMIT,
    NUM_BENCHED_PLAYERS,
)
from utils import position_num_players
import numpy as np
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict
import scipy.stats as stats

# Bayesian Q-learning normal-gamma hyperparameters
ALPHA = 2.0
LAMBDA = 1.0
THETA = 0.1


class FPLEnv(gym.Env):
    """
    - Fantasy Premier League Environment for Reinforcement Learning
        - state: current_team, budget, gameweek information e.g. unavailable players, player performance predictions
        - actions: transfer decisions (players to buy/sell)
        - Reward: points earned after each gameweek
    """

    def __init__(
        self,
        player_performance_samples: pd.DataFrame,
        initial_budget: float = 100.0,
        max_transfers_per_gw: int = 1,
        transfer_penalty: int = 4,
    ):
        super().__init__()
        # environment data
        self._sampled_gameweek_player_performance = player_performance_samples
        self._sampled_gameweek_player_performance.index = player_performance_samples[
            "name"
        ]

        # environment parameters
        self._initial_budget = initial_budget
        self._max_transfers_per_gw = max_transfers_per_gw
        self._tranfer_penalty = transfer_penalty

        # gym.Env variables
        self._action_subspace_size = 3
        self.action_space = gym.spaces.Discrete(self._action_subspace_size)
        self.observation_space = gym.spaces.Dict(
            {
                "budget": gym.spaces.Box(low=0, high=initial_budget, shape=(1,)),
                "team": gym.spaces.MultiDiscrete(
                    [len(player_performance_samples)] * NUM_PLAYERS_IN_FPL
                ),
                "gameweek": gym.spaces.Discrete(GAMEWEEK_COUNT),
                "player_performance_prediction": gym.spaces.Box(
                    low=0,
                    high=100,  # Reasonable upperbound for player's points in a single gameweek
                    shape=(
                        len(player_performance_samples),
                        3,
                    ),  # Predicted points, mean_real_past_points, fixture_difficulty
                ),
            }
        )

        self.reset(seed=RANDOM_SEED)

    def reset(self, *, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed, options=options)
        self._current_gw = 0
        self._budget = self._initial_budget
        self._total_points = 0
        self._free_transfers = 1

        self._team = self._initialize_team()
        self._bench = self._select_bench()
        self._playing = self._team[~self._team["name"].isin(self._bench["name"])]
        self._captain = self._select_captain()
        self._vice_captain = self._select_vice_captain()

        self._action_subset = self._initialize_actions()
        self._q_values = self._initialize_q_values()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(
        self, action_index: int
    ) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """
        - Perform a state transition in the Markov Decision Problem (MDP) modeling of FPL
        - action: index into self._action_subset
        """
        selected_action = self._action_subset[action_index]
        transfer_cost = 0
        if selected_action["transfer"]:
            # Possibility of having no good transfer to make
            # 1 transfer restriction per step
            transfer_cost = self._make_transfer(selected_action)

        # Update captain and bench based on predictions
        self._captain = self._select_captain()
        self._vice_captain = self._select_vice_captain()
        self._bench = self._select_bench()
        self._playing = self._team[~self._team["name"].isin(self._bench["name"])]

        gw_points = self._calculate_gw_points()
        adjusted_points = gw_points - transfer_cost
        self._total_points += adjusted_points

        self._current_gw += 1

        # Use Bayesian Q-learning and Value of Perfect Information to update action subset
        self._update_action_subset(action_index, adjusted_points)

        done = self._current_gw >= GAMEWEEK_COUNT
        observation = self._get_observation()
        info = {
            "team": self._team,
            "budget": self._budget,
            "points": self._total_points,
            "transfers": selected_action["transfer"],
        }

        return observation, adjusted_points, done, info

    def _initialize_q_values(self) -> DefaultDict[Tuple[str, str], Dict[str, Any]]:
        """Initialize Bayesian Q-values for the action subset"""
        q_values = defaultdict(dict)
        for action in self._action_subset:
            # action: (sell_player, buy_player)
            sell_player, buy_player = action["transfer"]
            sampled_reward = self._sampled_gameweek_player_performance.loc[
                buy_player, "sampled_points"
            ]
            M2 = sampled_reward**2  # Second moment
            q_values[(sell_player, buy_player)] = {
                "μ": sampled_reward,
                "λ": LAMBDA,
                "β": THETA**2 * M2,
                "α": ALPHA,
                "vpi": 0.0,  # Value of perfect information
            }

        return q_values

    def _initialize_team(self) -> pd.DataFrame:
        """
        - Select a team of 15 players with the highest sampled points per unit cost in the
         first gameweek based on FPL's rules:
            - 2 goalkeepers
            - 5 defenders
            - 5 midfielders
            - 3 forwards
        - Other constrainst include:
            - 3 players per Premier league team
            - budget - 100 million Euros
        """
        # Greedily sort players using their expected points per unit cost
        self._sampled_gameweek_player_performance["value"] = (
            self._sampled_gameweek_player_performance["sampled_points"]
            / self._sampled_gameweek_player_performance["price"]
        )
        sorted_sampled_gameweek_player_performance = (
            self._sampled_gameweek_player_performance.sort_values(
                by="value", ascending=False, inplace=False
            )
        )

        team_counts = {team_id: 0 for team_id in range(1, 21)}
        position_counts = {position: 0 for position in POSITIONS}
        budget = self._initial_budget
        sampled_team = []
        for _, player_row in sorted_sampled_gameweek_player_performance.iterrows():
            if (
                player_row["price"] <= budget
                and position_counts[player_row["position"]]
                < position_num_players(player_row["position"])
                and team_counts[player_row["team"]] < TEAM_COUNT_LIMIT
            ):
                team_counts[player_row["team"]] += 1
                position_counts[player_row["position"]] += 1
                budget -= player_row["price"]
                sampled_team.append(player_row.copy())
            if len(sampled_team) == NUM_PLAYERS_IN_FPL:
                break

        sampled_team_df = pd.DataFrame(sampled_team)  # Retain "name" as index
        return sampled_team_df

    def _select_bench(self) -> pd.DataFrame:
        """
        - Bench 4 players with the lowest expected points while respecting FPL team formation rules:
            - 1 GK
            - At least 3 DEF
            - At least 3 MID
            - At least 1 FWD
        """
        sorted_team_descending = self._team.sort_values(
            by="sampled_points", ascending=True, inplace=False
        )
        bench = []
        benched_positions_count = defaultdict(int)
        # Bench one GK by default
        goal_keepers = self._team[self._team["position"] == "GK"].sort_values(
            by="sampled_points", inplace=False, ascending=True
        )
        bench.append(goal_keepers.iloc[0].copy())
        for _, player_row in sorted_team_descending.iterrows():
            if len(bench) == NUM_BENCHED_PLAYERS:
                break
            else:
                if benched_positions_count[player_row["position"]] < 2:
                    # Every position, apart from that of the GK allows a max of 2 players from that position to be benched\
                    bench.append(player_row.copy())
                    benched_positions_count[player_row["position"]] += 1

        benched_players = pd.DataFrame(bench)
        return benched_players

    def _get_observation(self) -> Dict[str, Any]:
        """Return current state of the observation state"""
        return {
            "budget": self._budget,
            "team": self._team,
            "player_performance_prediction": self._sampled_gameweek_player_performance.filter(
                ["name", "team", "sampled_points"]
            ),
        }

    def _make_transfer(self, action: Dict) -> None:
        """
        - Replace players stipulated in actions["transfer"]
        - See shape of actions in self._initialize_actions()
        - The list of actions must be feasible before calling self._make_transfer()
        - Return: The cost of making the transfer
        """
        sell_player_name, buy_player_name = action["transfer"]
        self._team = self._team.drop(index=sell_player_name)
        self._team[len(self._team)] = self._sampled_gameweek_player_performance.loc[
            buy_player_name, :
        ]
        self._budget += self._sampled_gameweek_player_performance.loc[
            sell_player_name, "price"
        ]
        self.budget -= self._sampled_gameweek_player_performance.loc[
            buy_player_name, "price"
        ]

        transfer_cost = self._tranfer_penalty if self._free_transfers == 0 else 0
        self._free_transfers = max(self._free_transfers - 1, 0)

        return transfer_cost

    def _get_player_points(self):
        pass

    def _select_captain(self):
        """Select player with highest expected points as captain"""
        return self._team["sampled_points"].idxmax()

    def _select_vice_captain(self):
        """Select player with second highest expected points as vice captain"""
        team_without_captain = self._team.drop(index=self._captain, inplace=False)
        vice_captain = team_without_captain["sampled_points"].idxmax()

        return vice_captain

    def _initialize_actions(self) -> List[Dict[str, Any]]:
        """Generate promising actions for the current game state"""
        actions = [
            {
                "transfer": self._generate_transfer(),
                "captain": None,  # Determined dynamically
                "vice_captain": None,  # Determined dynamically
                "bench": None,  # Determined dynamically
            }
            for _ in range(self._action_subspace_size)
        ]

        return actions

    def _generate_transfer(self) -> Tuple[str, str]:
        """Generate a new promising transfer based on scores per unit price"""
        unselected_players = self._sampled_gameweek_player_performance[
            ~self._sampled_gameweek_player_performance["name"].isin(self._team["name"])
        ]
        unselected_players["value"] = (
            unselected_players["cumulative_real_points"] / unselected_players["price"]
        )
        unselected_players.sort_values(by="value", inplace=True, ascending=False)

        team_with_player_value = self._team.copy()
        team_with_player_value["value"] = (
            team_with_player_value["cumulative_real_points"]
            / team_with_player_value["price"]
        )
        # Drop Goalkeepers - naturally have lower points/cost
        team_with_player_value = team_with_player_value[
            team_with_player_value["position"] != "GK"
        ]
        weakest_player = team_with_player_value.sort_values(
            by="value", inplace="False", ascending=True
        ).iloc[:0]
        for _, player_row in unselected_players.iterrows():
            if (
                player_row["position"] == weakest_player["position"].values[0]
                and player_row["price"] <= weakest_player["price"].values[0]
            ):
                # (sell_player, buy_player)
                return weakest_player["name"].values[0], player_row["name"]

        return None

    def _calculate_gw_points(self) -> int:
        """Sum sampled points for self._playing and double captain's points according to FPL rules"""
        return (
            self._playing["sampled_points"].sum()
            + self._playing.loc[self._captain, "sampled_points"]
        )

    def _update_q_value(self, action_index: int, reward: int) -> None:
        """
        - Update the Bayesian Q-value associated with the action index using
        normal-gamma moment updating
        """
        action = self._action_subset[action_index]["transfer"]
        λ = self._q_values[action]["λ"]
        μ = self._q_values[action]["μ"]
        β = self._q_values[action]["β"]

        self._q_values[action]["λ"] += 1
        self._q_values[action]["α"] += 0.5
        # Update μ and β using moment updating
        self._q_values[action]["μ"] += ((λ * μ) + reward) / self._q_values[action]["λ"]
        self._q_values[action]["β"] += β + (
            (0.5 * λ * (reward - μ) ** 2) / self._q_values[action]["λ"]
        )

        return None

    def _calculate_vpi(self, action: Tuple[str, str]) -> float:
        """
        - Calculate the Value of Perfect Information for an action
        - Action: (sell_player_name, buy_player_name)
        """
        best_action = max(
            self._q_values, key=lambda action: self._q_values[action]["μ"]
        )
        best_q_value = self._q_values[best_action]["μ"]
        second_best_q_value = max(
            [
                self._q_values[action]["μ"]
                for action in self._q_values.keys()
                if action != best_action
            ],
            default=0,
        )
        # Parameters for t-distribution
        ν = 2 * self._q_values[action]["α"]
        σ = np.sqrt(
            self._q_values[action]["β"]
            * (1 + 1 / self._q_values[action]["λ"])
            / self._q_values[action]["α"]
        )

        # Calculate VPI
        t_distribution = stats.t(df=ν, loc=self._q_values[action]["μ"], scale=σ)
        if action == best_action:
            #  For the best action, learning only helps if its true value turns out worse than estimate
            #  of second-best action
            vpi = σ * t_distribution.pdf(
                (second_best_q_value - self._q_values[action]["μ"]) / σ
            ) - (self._q_values[action]["μ"] - second_best_q_value)
        else:
            # Other actions' vpi only matter is their true value turns out to be better than estimated
            # value of best action
            vpi = σ * t_distribution.pdf(
                (best_q_value - self._q_values[action]["μ"]) / σ
            ) - (self._q_values[action]["μ"] - best_q_value)

        # VPi should be non-negative
        return max(0, vpi)

    def _update_action_subset(self, selected_action_index: int, reward):
        """
        - Update the action subset in place based on Bayesian Q-learning, replacing low-value actions with new, promising ones
        using the Value of Perfect Information (VPI) approach
        """
        self._update_q_value(selected_action_index, reward)

        # Calculate VPI for all actions
        for action_index in range(self._action_subspace_size):
            action = self._action_subset[action_index]["transfer"]
            self._q_values[action]["vpi"] = self._calculate_vpi(action)

        best_action = max(
            self._q_values, key=lambda action: self._q_values[action]["μ"]
        )
        best_q_value = self._q_values[best_action]["μ"]

        #  Replace actions where q_hat + VPI < best_q with new actions
        for action, q_value_dict in self._q_values.items():
            if action != best_action:
                q_hat_vpi = q_value_dict["μ"] + q_value_dict["vpi"]
                if q_hat_vpi < best_q_value:
                    #  Replace this action
                    new_action = self._generate_transfer()
                    old_action_index = next(
                        (
                            index
                            for index, action_dict in enumerate(self._action_subset)
                            if action_dict["transfer"] == action
                        ),
                        None,
                    )
                    self._action_subset[old_action_index] = {
                        "transfer": new_action,
                        "captain": None,  # Determined dynamically
                        "vice_captain": None,  # Determined dynamically
                        "bench": None,  # Determined dynamically
                    }

                    _, buy_player = new_action
                    sampled_reward = self._sampled_gameweek_player_performance.loc[
                        buy_player, "sampled_points"
                    ]
                    M2 = sampled_reward**2  # Second moment
                    self._q_values[new_action] = {
                        "μ": sampled_reward,
                        "λ": LAMBDA,
                        "β": THETA**2 * M2,
                        "α": ALPHA,
                        "vpi": 0.0,
                    }
                    self._q_values.pop(action)

        return None
