from typing import Dict, List, Tuple, Any
import math
from scipy.stats import norm
from collections import defaultdict
from fpl_env import FPLEnv
from utils import get_logger
import numpy as np
from constants import GAMEWEEK_COUNT

# Module logger
logger = get_logger(__name__)


class BayesianQLearningAgent:
    """
    Bayesian Q-Learning Agent for Fantasy Premier League

    This agent implements the Bayesian Q-learning approach described in the paper
    "Competing with Humans at Fantasy Football: Team Formation in Large Partially-Observable Domains"

    It maintains normal-gamma distributions over Q-values and uses Value of Perfect Information (VPI)
    to balance exploration and exploitation.
    """

    def __init__(
        self,
        discount_factor: float = 0.5,
        init_variance_ratio: float = 0.1,
        episode_limit: int = 50,
        num_actions: int = 3,
    ):
        """
        Initialize the Bayesian Q-Learning Agent

        Args:
            discount_factor: Discount factor for future rewards (γ)
            init_variance_ratio: Proportion of mean used to set initial variance (θ)
            episode_limit: Maximum number of episodes to run
            num_actions: Number of candidate actions to maintain
        """
        self.gamma = discount_factor
        self.theta = init_variance_ratio
        self.episode_limit = episode_limit
        self.num_actions = num_actions

        # Q-value distributions for each state-action pair
        # Each distribution is a normal-gamma with parameters (μ, λ, α, β)
        self.q_distributions = defaultdict(dict)

        # Current episode
        self.current_episode = 0

        # Performance tracking
        self.episode_rewards = []
        self.cumulative_reward = 0

    def select_action(self, state: Any, candidate_actions: List[Any]) -> int:
        """
        Select the best action according to Q-value means and VPI

        Args:
            state: Current state (will be serialized as key)
            candidate_actions: List of available actions

        Returns:
            action_idx: Index of the selected action
        """
        state_key = self._hash_state(state)

        # Initialize Q-distributions for any new state-action pairs
        for action_idx, action in enumerate(candidate_actions):
            if action_idx not in self.q_distributions[state_key]:
                # Initialize with estimate of immediate reward
                expected_reward = action["expected_points"]

                # Initialize normal-gamma parameters
                # µ is initial Q-value (set to immediate reward estimate)
                # λ is precision parameter (set to 1 as per paper)
                # α is shape parameter (set to 2 as per paper)
                # β is scale parameter (set to θ²M₂ as per paper)
                self.q_distributions[state_key][action_idx] = {
                    "mu": expected_reward,
                    "lambda": 1.0,
                    "alpha": 2.0,
                    "beta": self.theta**2 * expected_reward**2,
                }

        # If we have fewer than num_actions, initialize the rest
        if len(self.q_distributions[state_key]) < self.num_actions:
            for action_idx in range(
                len(self.q_distributions[state_key]), self.num_actions
            ):
                # Initialize with a default value
                self.q_distributions[state_key][action_idx] = {
                    "mu": 0.0,
                    "lambda": 1.0,
                    "alpha": 2.0,
                    "beta": self.theta**2,
                }

        # Get Q-value means for each action
        q_means = {
            action_idx: self.q_distributions[state_key][action_idx]["mu"]
            for action_idx in range(len(candidate_actions))
        }

        # Find the best and second best actions based on means
        best_action_idx = max(q_means, key=q_means.get)
        best_q_mean = q_means[best_action_idx]

        # Find second best Q-value (used for VPI calculation)
        second_best_q_mean = float("-inf")
        for action_idx, q_mean in q_means.items():
            if action_idx != best_action_idx and q_mean > second_best_q_mean:
                second_best_q_mean = q_mean

        if second_best_q_mean == float("-inf"):
            second_best_q_mean = 0  # Default if only one action

        # Calculate VPI for each action
        vpi_values = {}
        for action_idx in range(len(candidate_actions)):
            vpi = self._calculate_vpi(
                state_key, action_idx, best_action_idx, second_best_q_mean, best_q_mean
            )
            vpi_values[action_idx] = vpi

        # Select action with highest Q-value + VPI
        # This balances exploitation (high Q-value) with exploration (high VPI)
        action_scores = {
            action_idx: q_means[action_idx] + vpi_values[action_idx]
            for action_idx in range(len(candidate_actions))
        }
        selected_action = max(action_scores, key=action_scores.get)

        # Ensure the selected action is valid
        if selected_action >= len(candidate_actions):
            # Fallback to best mean if invalid
            selected_action = best_action_idx

        return selected_action

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_candidate_actions: List[Any],
        done: bool,
    ) -> None:
        """
        Update Q-value distribution based on observed reward

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            next_candidate_actions: List of available actions in next state
            done: Whether the episode is complete
        """
        state_key = self._hash_state(state)
        next_state_key = self._hash_state(next_state)

        # If this is a terminal state, only use immediate reward
        if done:
            target = reward
        else:
            # Get maximum Q-value for next state (bootstrapping)
            next_q_values = []
            for action_idx in range(len(next_candidate_actions)):
                if action_idx in self.q_distributions[next_state_key]:
                    next_q_values.append(
                        self.q_distributions[next_state_key][action_idx]["mu"]
                    )
                else:
                    # Initialize with expected immediate reward
                    expected_reward = next_candidate_actions[action_idx][
                        "expected_points"
                    ]
                    next_q_values.append(expected_reward)

            # Use the maximum Q-value for bootstrapping
            max_next_q = max(next_q_values) if next_q_values else 0

            # Calculate target Q-value using Bellman equation
            target = reward + self.gamma * max_next_q

        # Get current distribution parameters
        params = self.q_distributions[state_key][action]
        mu = params["mu"]
        lambda_param = params["lambda"]
        alpha = params["alpha"]
        beta = params["beta"]

        # Update the normal-gamma distribution parameters
        # These update equations follow the moment-matching procedure described in Dearden et al. 1998
        lambda_new = lambda_param + 1
        alpha_new = alpha + 0.5

        # Calculate moments
        m1 = target  # First moment (mean)
        m2 = target**2  # Second moment

        # Update mean
        mu_new = (lambda_param * mu + m1) / lambda_new

        # Update beta
        beta_new = beta + 0.5 * (
            lambda_param * (mu**2) + m2 - lambda_new * (mu_new**2)
        )

        # Store updated parameters
        self.q_distributions[state_key][action] = {
            "mu": mu_new,
            "lambda": lambda_new,
            "alpha": alpha_new,
            "beta": beta_new,
        }

    def train(self, env: FPLEnv, num_episodes: int) -> List[float]:
        """
        Train the agent over multiple episodes

        Args:
            env: FPL environment
            num_episodes: Number of episodes to train

        Returns:
            episode_rewards: List of total rewards per episode
        """
        episode_rewards = []
        gameweek_rewards = np.zeros(shape=(GAMEWEEK_COUNT,))

        for episode in range(num_episodes):
            self.current_episode = episode
            total_reward = 0
            state, _ = env.reset()
            done = False
            episode_gameweek_rewards = []

            while not done:
                # Generate candidate actions if needed
                if not env.candidate_actions:
                    env._generate_candidate_actions()

                # Select action
                action = self.select_action(state, env.candidate_actions)

                # Take action
                next_state, reward, done, _, info = env.step(action)

                # Generate next candidate actions (if not done)
                if not done:
                    env._generate_candidate_actions()
                    next_candidate_actions = env.candidate_actions
                else:
                    next_candidate_actions = []

                # Update Q-values
                self.update(
                    state, action, reward, next_state, next_candidate_actions, done
                )

                # Update state and accumulate reward
                state = next_state
                total_reward += reward
                episode_gameweek_rewards.append(reward)

            # Record episode performance
            episode_rewards.append(total_reward)
            self.episode_rewards.append(total_reward)
            self.cumulative_reward += total_reward
            gameweek_rewards += np.array(episode_gameweek_rewards)

            logger.info(
                f"Episode {episode+1}/{num_episodes} - Total points: {total_reward} - Cumulative points: {self.cumulative_reward}"
            )
        gameweek_rewards = list((gameweek_rewards / num_episodes).astype(int))

        return episode_rewards, gameweek_rewards

    def _calculate_vpi(
        self,
        state_key: str,
        action_idx: int,
        best_action_idx: int,
        second_best_q_mean: float,
        best_q_mean: float,
    ) -> float:
        """
        Calculate the Value of Perfect Information (VPI) for an action

        Args:
            state_key: Hashed state representation
            action_idx: Index of the action to evaluate
            best_action_idx: Index of the action with highest Q-value mean
            second_best_q_mean: Q-value mean of the second-best action
            best_q_mean: Q-value mean of the best action
        """
        # Get distribution parameters
        params = self.q_distributions[state_key][action_idx]
        mu = params["mu"]
        lambda_param = params["lambda"]
        alpha = params["alpha"]
        beta = params["beta"]

        # Calculate standard deviation from the normal-gamma distribution
        if alpha > 1:  # Ensure variance exists
            variance = beta / (lambda_param * (alpha - 1))
            std_dev = math.sqrt(variance)
        else:
            # Default to a small but non-zero std_dev if variance is undefined
            std_dev = 0.1

        # Calculate VPI differently based on whether this is the best action or not
        if action_idx == best_action_idx:
            # For the best action, we learn something only if its true value
            # is less than the second best action's estimated value

            # Calculate the probability that q* < second_best_q_mean
            prob_threshold = (second_best_q_mean - mu) / std_dev if std_dev > 0 else 0
            prob_less = norm.cdf(prob_threshold)

            # Calculate the expected gain if q* < second_best_q_mean
            # E[max(second_best_q_mean - q*, 0)]
            # For normal distribution: E[max(c-X, 0)] = (c-μ)Φ((c-μ)/σ) + σφ((c-μ)/σ)
            expected_gain = (second_best_q_mean - mu) * prob_less
            if std_dev > 0:
                expected_gain += std_dev * norm.pdf(prob_threshold)

            return expected_gain
        else:
            # For non-best actions, we learn something only if its true value
            # is greater than the best action's estimated value

            # Calculate the probability that q* > best_q_mean
            prob_threshold = (best_q_mean - mu) / std_dev if std_dev > 0 else 0
            prob_greater = 1 - norm.cdf(prob_threshold)

            # Calculate the expected gain if q* > best_q_mean
            # E[max(q* - best_q_mean, 0)]
            # For normal distribution: E[max(X-c, 0)] = (μ-c)Φ((μ-c)/σ) + σφ((μ-c)/σ)
            expected_gain = (mu - best_q_mean) * prob_greater
            if std_dev > 0:
                expected_gain += std_dev * norm.pdf(prob_threshold)

            return expected_gain

    def _hash_state(self, state: Any) -> str:
        """
        Convert a state to a hashable representation

        Args:
            state: State observation

        Returns:
            state_key: Hashable representation of the state
        """
        # For the FPL environment, we can use the current gameweek as a simple hash
        return (
            f"gw_{state['current_gameweek']}_transfers_{state['available_transfers']}"
        )

    def should_replace_action(
        self, state_key: str, action_idx: int, best_action_idx: int
    ) -> bool:
        """
        Determine whether to replace a candidate action based on VPI

        Only replace an action when its expected value plus VPI is less than the best action's expected value.

        """
        if action_idx == best_action_idx:
            return False

        # Get Q-values
        best_q_mean = self.q_distributions[state_key][best_action_idx]["mu"]
        action_q_mean = self.q_distributions[state_key][action_idx]["mu"]

        # Calculate VPI
        vpi = self._calculate_vpi(
            state_key,
            action_idx,
            best_action_idx,
            action_q_mean,  # Not used for non-best actions
            best_q_mean,
        )

        # Replace if expected value + VPI is less than best action's expected value
        return (action_q_mean + vpi) < best_q_mean

    def evaluate(self, env: FPLEnv) -> Tuple[float, List[Dict]]:
        """
        Evaluate the trained agent on a season

        Returns:
            total_points: Total points scored
            decisions: List of decisions made at each gameweek
        """
        state, _ = env.reset()
        done = False
        total_points = 0
        decisions = []
        gameweek_rewards = []

        while not done:
            # Generate candidate actions
            if not env.candidate_actions:
                env._generate_candidate_actions()

            # Select best action (no exploration)
            state_key = self._hash_state(state)
            q_means = {}
            for action_idx in range(len(env.candidate_actions)):
                if action_idx in self.q_distributions[state_key]:
                    q_means[action_idx] = self.q_distributions[state_key][action_idx][
                        "mu"
                    ]
                else:
                    q_means[action_idx] = env.candidate_actions[action_idx][
                        "expected_points"
                    ]

            # Select the action with highest Q-value mean
            action = max(q_means, key=q_means.get)

            # Record decision
            decision = {
                "gameweek": env.current_gameweek,
                "selected_team": env.candidate_actions[action],
                "q_value": q_means[action],
            }
            decisions.append(decision)

            # Take action
            next_state, reward, done, _, info = env.step(action)

            # Update state and accumulate reward
            state = next_state
            total_points += reward
            gameweek_rewards.append(reward)

            logger.info(
                f"Gameweek {info['gameweek']-1} - Points: {reward} - Total: {total_points}"
            )

        return total_points, decisions, gameweek_rewards
