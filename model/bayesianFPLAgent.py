import random
from constants import RANDOM_SEED
from fpl_env import FPLEnv
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np


class BayesianFPLAgent:
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        random.seed(RANDOM_SEED)

    def _select_action(self, env: FPLEnv, training: bool = True) -> int:
        """
        Select action based on Bayesian Q-learning principles

        If training, use epsilon-greedy with VPI
        If evaluating, just select best action

        Return: Index of action in env.action_subset
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, env.action_space.n - 1)
        else:
            # Exploitation: select action with highest Q + VPI
            best_action = max(
                env._q_values,
                key=lambda action: env._q_values[action]["μ"]
                + env._q_values[action]["vpi"],
            )
            return next(
                (
                    index
                    for index, action_dict in enumerate(env._action_subset)
                    if action_dict["transfer"] == best_action
                ),
                None,
            )

    def _update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: FPLEnv, num_episodes: int) -> Tuple[List[int], Dict[str, Any]]:
        """Train the Bayesian Q-learning agent over num_episodes"""
        total_rewards, info = [], None

        for episode in tqdm(range(num_episodes)):
            state, _ = env.reset(seed=RANDOM_SEED)
            done = False
            episode_reward = 0

            while not done:
                # Select action
                action_index = self._select_action(env)
                # Take action
                next_state, reward, done, info = env.step(action_index)
                #  Perform updates
                state = next_state
                episode_reward += reward

                if done:
                    break
            # Update exploration rate
            self._update_epsilon()
            # Record results
            total_rewards.append(episode_reward)

            # Print progress
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Avg Last 10: {np.mean(total_rewards[-10:]):.2f}"
            )

        return total_rewards, info
