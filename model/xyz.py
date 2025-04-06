import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
import random
from collections import deque, namedtuple
import scipy.stats as stats
from typing import Dict, List, Tuple, Any

# Constants for FPL
MAX_PLAYERS = 15
MAX_BUDGET = 100.0
POSITIONS = ["GK", "DEF", "MID", "FWD"]
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
TEAM_LIMIT = 3  # Max players from same team

class FPLEnv(gym.Env):
    """
    Fantasy Premier League Environment
    
    State: Current team, budget, gameweek information, player performance predictions
    Action: Transfer decisions
    Reward: Points earned after each gameweek
    """
    
    def __init__(self, player_samples, real_player_points, initial_budget=100.0, 
                 max_transfers_per_gw=2, transfer_penalty=4):
        super().__init__()
        
        # Store player performance samples and real player points
        self.player_samples = player_samples  # Dictionary: {player_id: {gw: [samples]}}
        self.real_player_points = real_player_points  # Dictionary: {player_id: {gw: points}}
        
        # Player information
        self.player_positions = self._get_player_positions()  # You would implement this
        self.player_teams = self._get_player_teams()  # You would implement this
        self.player_prices = self._get_player_prices()  # You would implement this
        
        # Environment parameters
        self.initial_budget = initial_budget
        self.max_transfers_per_gw = max_transfers_per_gw
        self.transfer_penalty = transfer_penalty
        self.total_players = len(player_samples)
        self.num_gameweeks = 38
        
        # Define the action space
        # For simplicity, we'll use a discrete action space for now
        # Each action represents a subset of possible transfer combinations
        self.action_subset_size = 3  # As mentioned in your approach
        self.action_space = spaces.Discrete(self.action_subset_size)
        
        # Define the observation space (simplified)
        self.observation_space = spaces.Dict({
            'team': spaces.MultiDiscrete([self.total_players] * MAX_PLAYERS),
            'budget': spaces.Box(low=0, high=initial_budget, shape=(1,)),
            'gameweek': spaces.Discrete(self.num_gameweeks),
            'player_predictions': spaces.Box(
                low=-float('inf'), high=float('inf'), 
                shape=(self.total_players, 4)  # Mean, variance, form, fixture difficulty
            ),
        })
        
        # Initialize state
        self.reset()
    
    def reset(self, **kwargs):
        # Reset environment for new season
        self.current_gw = 0
        self.budget = self.initial_budget
        self.team = self._initialize_team()
        self.captain = self._select_captain(self.team)
        self.vice_captain = self._select_vice_captain(self.team, self.captain)
        self.bench = self._select_bench(self.team)
        self.playing = [p for p in self.team if p not in self.bench]
        self.total_points = 0
        self.free_transfers = 1
        
        # Initialize action subset
        self.current_action_subset = self._generate_new_actions(self.action_subset_size)
        
        # Initialize Bayesian Q-values
        self.q_values = self._initialize_q_values()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step within the environment
        Action: Index into the current_action_subset
        """
        # Get the actual transfer decisions from the action subset
        selected_action = self.current_action_subset[action]
        
        # Process transfers
        old_team = self.team.copy()
        transfer_cost = self._make_transfers(selected_action)
        
        # Update captain and bench based on predictions
        self.captain = self._select_captain(self.team)
        self.vice_captain = self._select_vice_captain(self.team, self.captain)
        self.bench = self._select_bench(self.team)
        self.playing = [p for p in self.team if p not in self.bench]
        
        # Simulate gameweek and get points
        points = self._simulate_gameweek()
        adjusted_points = points - transfer_cost
        self.total_points += adjusted_points
        
        # Update free transfers
        if len(selected_action['transfers']) == 0:
            self.free_transfers = min(2, self.free_transfers + 1)
        else:
            self.free_transfers = 1
        
        # Update state
        self.current_gw += 1
        
        # Update action subset using Bayesian Q-learning approach
        self._update_action_subset(selected_action, adjusted_points)
        
        # Check if season is over
        done = self.current_gw >= self.num_gameweeks
        
        # Get observation
        observation = self._get_observation()
        reward = adjusted_points
        info = {
            'team': self.team,
            'budget': self.budget,
            'points': points,
            'total_points': self.total_points,
            'transfers': selected_action['transfers']
        }
        
        return observation, reward, done, False, info
    
    def _initialize_team(self):
        """Initialize a valid team respecting all FPL constraints"""
        team = []
        remaining_budget = self.initial_budget
        position_counts = {pos: 0 for pos in POSITIONS}
        team_counts = {}
        
        # Greedy approach to initialize team (can be improved)
        # Sort players by expected points per cost
        player_values = {}
        for player_id in self.player_samples:
            expected_points = np.mean(self.player_samples[player_id][0])  # GW1 samples
            cost = self.player_prices[player_id]
            position = self.player_positions[player_id]
            team = self.player_teams[player_id]
            
            if cost <= remaining_budget and position_counts[position] < POSITION_LIMITS[position]:
                player_values[player_id] = expected_points / cost
        
        sorted_players = sorted(player_values.keys(), key=lambda p: player_values[p], reverse=True)
        
        for player_id in sorted_players:
            position = self.player_positions[player_id]
            cost = self.player_prices[player_id]
            player_team = self.player_teams[player_id]
            
            if (len(team) < MAX_PLAYERS and 
                position_counts[position] < POSITION_LIMITS[position] and 
                team_counts.get(player_team, 0) < TEAM_LIMIT and
                cost <= remaining_budget):
                
                team.append(player_id)
                remaining_budget -= cost
                position_counts[position] += 1
                team_counts[player_team] = team_counts.get(player_team, 0) + 1
            
            if len(team) == MAX_PLAYERS:
                break
        
        self.budget = remaining_budget
        return team
    
    def _generate_new_actions(self, num_actions):
        """Generate promising actions for the current game state"""
        actions = []
        
        for _ in range(num_actions):
            # Each action represents a set of transfer decisions
            action = {
                'transfers': self._generate_random_transfers(),
                'captain': None,  # Will be determined dynamically
                'vice_captain': None,  # Will be determined dynamically
                'bench': []  # Will be determined dynamically
            }
            actions.append(action)
        
        return actions
    
    def _initialize_q_values(self):
        """Initialize the Bayesian Q-values for the action subset"""
        q_values = {}
        
        for i, action in enumerate(self.current_action_subset):
            # Initialize hyperparameters (μ, λ, α, β) for normal-gamma prior
            # Sample approximation of reward for this action
            sampled_reward = self._estimate_action_value(action)
            
            # As per your approach:
            alpha = 2.0
            lambda_val = 1.0
            mu = sampled_reward
            theta = 0.1  # Trade-off parameter
            M2 = mu**2   # Second moment
            beta = theta**2 * M2
            
            q_values[i] = {
                'mu': mu,
                'lambda': lambda_val,
                'alpha': alpha,
                'beta': beta,
                'vpi': 0.0  # Value of Perfect Information
            }
            
        return q_values
    
    def _make_transfers(self, action):
        """
        Execute transfers and return the transfer cost
        """
        transfers = action['transfers']
        cost = 0
        
        if len(transfers) > self.free_transfers:
            cost = (len(transfers) - self.free_transfers) * self.transfer_penalty
        
        # Execute transfers
        for out_player, in_player in transfers:
            if out_player in self.team:
                self.team.remove(out_player)
                self.team.append(in_player)
                
                # Update budget
                self.budget += self.player_prices[out_player]
                self.budget -= self.player_prices[in_player]
        
        return cost
    
    def _select_captain(self, team):
        """Select the player with highest expected points as captain"""
        best_player = None
        best_score = -1
        
        for player_id in team:
            expected_points = np.mean(self.player_samples[player_id][self.current_gw])
            if expected_points > best_score:
                best_score = expected_points
                best_player = player_id
        
        return best_player
    
    def _select_vice_captain(self, team, captain):
        """Select the player with second highest expected points as vice captain"""
        best_player = None
        best_score = -1
        
        for player_id in team:
            if player_id != captain:
                expected_points = np.mean(self.player_samples[player_id][self.current_gw])
                if expected_points > best_score:
                    best_score = expected_points
                    best_player = player_id
        
        return best_player
    
    def _select_bench(self, team):
        """Select the 4 players with lowest expected points for bench"""
        player_scores = []
        
        for player_id in team:
            expected_points = np.mean(self.player_samples[player_id][self.current_gw])
            player_scores.append((player_id, expected_points))
        
        # Sort by expected points (ascending)
        player_scores.sort(key=lambda x: x[1])
        
        # Select bench players
        bench = [p[0] for p in player_scores[:4]]
        
        return bench
    
    def _simulate_gameweek(self):
        """
        Simulate a gameweek and return points
        For training, use sampled points; for evaluation, use real points
        """
        total_points = 0
        
        # Process playing team (11 players)
        for player_id in self.playing:
            # Use real points if available (for evaluation), otherwise sample
            if player_id in self.real_player_points and self.current_gw in self.real_player_points[player_id]:
                points = self.real_player_points[player_id][self.current_gw]
            else:
                # Sample from player's performance distribution
                points = np.random.choice(self.player_samples[player_id][self.current_gw])
            
            # Apply captain bonus
            if player_id == self.captain:
                points *= 2
            
            total_points += points
        
        return total_points
    
    def _get_observation(self):
        """Get the current state observation"""
        # Get predictions for all players
        predictions = np.zeros((self.total_players, 4))
        
        for player_id in range(self.total_players):
            if player_id in self.player_samples and self.current_gw in self.player_samples[player_id]:
                samples = self.player_samples[player_id][self.current_gw]
                predictions[player_id, 0] = np.mean(samples)  # Mean
                predictions[player_id, 1] = np.var(samples)   # Variance
                
                # Add form (average of last 3 gameweeks)
                form = 0
                form_count = 0
                for i in range(1, 4):
                    if self.current_gw - i >= 0 and self.current_gw - i in self.player_samples[player_id]:
                        form += np.mean(self.player_samples[player_id][self.current_gw - i])
                        form_count += 1
                
                if form_count > 0:
                    predictions[player_id, 2] = form / form_count
                
                # Add fixture difficulty (could be obtained from external data)
                predictions[player_id, 3] = 0  # Placeholder
        
        return {
            'team': np.array(self.team + [0] * (MAX_PLAYERS - len(self.team))),
            'budget': np.array([self.budget]),
            'gameweek': self.current_gw,
            'player_predictions': predictions
        }
    
    def _update_action_subset(self, selected_action, reward):
        """
        Update the action subset based on Bayesian Q-learning principles
        Replace low-value actions with new ones as per the VPI approach
        """
        # Update Q-value for the selected action
        action_idx = self.current_action_subset.index(selected_action)
        self._update_q_value(action_idx, reward)
        
        # Calculate VPI for all actions
        for idx in range(len(self.current_action_subset)):
            self.q_values[idx]['vpi'] = self._calculate_vpi(idx)
        
        # Find best action
        best_idx = max(self.q_values, key=lambda x: self.q_values[x]['mu'])
        best_q = self.q_values[best_idx]['mu']
        
        # Replace actions where q_hat + VPI < best_q with new actions
        for idx in range(len(self.current_action_subset)):
            if idx != best_idx:
                q_vpi = self.q_values[idx]['mu'] + self.q_values[idx]['vpi']
                if q_vpi < best_q:
                    # Replace this action
                    self.current_action_subset[idx] = self._generate_random_transfers()
                    
                    # Initialize new Q-value
                    sampled_reward = self._estimate_action_value(self.current_action_subset[idx])
                    alpha = 2.0
                    lambda_val = 1.0
                    mu = sampled_reward
                    theta = 0.1
                    M2 = mu**2
                    beta = theta**2 * M2
                    
                    self.q_values[idx] = {
                        'mu': mu,
                        'lambda': lambda_val,
                        'alpha': alpha,
                        'beta': beta,
                        'vpi': 0.0
                    }
    
    def _update_q_value(self, action_idx, reward):
        """
        Update the Bayesian Q-value using normal-gamma moment updating
        """
        # Get current hyperparameters
        mu = self.q_values[action_idx]['mu']
        lambda_val = self.q_values[action_idx]['lambda']
        alpha = self.q_values[action_idx]['alpha']
        beta = self.q_values[action_idx]['beta']
        
        # Update hyperparameters
        lambda_new = lambda_val + 1
        alpha_new = alpha + 0.5
        
        # Update μ and β using moment updating
        mu_new = (lambda_val * mu + reward) / lambda_new
        beta_new = beta + 0.5 * lambda_val * (reward - mu)**2 / lambda_new
        
        # Store updated hyperparameters
        self.q_values[action_idx]['mu'] = mu_new
        self.q_values[action_idx]['lambda'] = lambda_new
        self.q_values[action_idx]['alpha'] = alpha_new
        self.q_values[action_idx]['beta'] = beta_new
    
    def _calculate_vpi(self, action_idx):
        """
        Calculate the Value of Perfect Information for an action
        """
        # Get the best action's Q-value
        best_idx = max(self.q_values, key=lambda x: self.q_values[x]['mu'])
        best_q = self.q_values[best_idx]['mu']
        second_best_q = max([self.q_values[i]['mu'] for i in self.q_values if i != best_idx], default=0)
        
        # Get the current action's hyperparameters
        mu = self.q_values[action_idx]['mu']
        lambda_val = self.q_values[action_idx]['lambda']
        alpha = self.q_values[action_idx]['alpha']
        beta = self.q_values[action_idx]['beta']
        
        # Calculate parameters for Student's t-distribution
        nu = 2 * alpha
        sigma = np.sqrt(beta * (1 + 1/lambda_val) / alpha)
        
        # Calculate VPI
        if action_idx == best_idx:
            # Best action's VPI (how much we might learn if it's worse than we thought)
            t_distribution = stats.t(df=nu, loc=mu, scale=sigma)
            # VPI is expected gain if we learn q* < second_best_q
            vpi = sigma * t_distribution.pdf((second_best_q - mu) / sigma) - (mu - second_best_q) * (1 - t_distribution.cdf((second_best_q - mu) / sigma))
        else:
            # Other actions' VPI (how much we might learn if they're better than we thought)
            t_distribution = stats.t(df=nu, loc=mu, scale=sigma)
            # VPI is expected gain if we learn q* > best_q
            vpi = sigma * t_distribution.pdf((best_q - mu) / sigma) + (mu - best_q) * t_distribution.cdf((best_q - mu) / sigma)
        
        return max(0, vpi)  # VPI should be non-negative
    
    def _generate_random_transfers(self):
        """Generate random but valid transfers"""
        # Simplified implementation
        num_transfers = random.randint(0, self.max_transfers_per_gw)
        transfers = []
        
        if num_transfers > 0:
            # Randomly select players to transfer out
            out_candidates = random.sample(self.team, min(num_transfers, len(self.team)))
            
            for out_player in out_candidates:
                # Find a valid replacement
                valid_replacements = []
                position = self.player_positions[out_player]
                
                for in_player in range(self.total_players):
                    if (in_player not in self.team and 
                        self.player_positions[in_player] == position and
                        self.player_prices[in_player] <= self.budget + self.player_prices[out_player]):
                        valid_replacements.append(in_player)
                
                if valid_replacements:
                    in_player = random.choice(valid_replacements)
                    transfers.append((out_player, in_player))
        
        return {'transfers': transfers}
    
    def _estimate_action_value(self, action):
        """
        Estimate the value of an action by simulating its outcome
        This is used for initializing Q-values
        """
        # Clone current state
        original_team = self.team.copy()
        original_budget = self.budget
        original_free_transfers = self.free_transfers
        
        # Apply action
        transfer_cost = self._make_transfers(action)
        
        # Simulate outcome (simplified)
        # For a more accurate estimate, you could simulate multiple gameweeks ahead
        points = self._simulate_gameweek()
        adjusted_points = points - transfer_cost
        
        # Restore original state
        self.team = original_team
        self.budget = original_budget
        self.free_transfers = original_free_transfers
        
        return adjusted_points
    
    def _get_player_positions(self):
        """Get player positions (implement based on your data structure)"""
        # Placeholder
        return {player_id: random.choice(POSITIONS) for player_id in range(self.total_players)}
    
    def _get_player_teams(self):
        """Get player teams (implement based on your data structure)"""
        # Placeholder
        return {player_id: random.randint(1, 20) for player_id in range(self.total_players)}
    
    def _get_player_prices(self):
        """Get player prices (implement based on your data structure)"""
        # Placeholder
        return {player_id: random.uniform(4.0, 12.0) for player_id in range(self.total_players)}


class BayesianFPLAgent:
    """
    Bayesian Q-Learning agent for FPL using Value of Perfect Information
    """
    
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, env, training=True):
        """
        Select action based on Bayesian Q-learning principles
        
        If training, use epsilon-greedy with VPI
        If evaluating, just select best action
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, env.action_space.n - 1)
        else:
            # Exploitation: select action with highest Q + VPI
            q_values = env.q_values
            best_action = max(q_values, key=lambda idx: q_values[idx]['mu'] + q_values[idx]['vpi'])
            return best_action
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, num_episodes=100):
    """Train the Bayesian Q-learning agent"""
    total_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        # Reset environment
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action
            action = agent.select_action(env)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Environment updates Q-values internally
            
            # Update state
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record results
        total_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Avg Last 10: {np.mean(total_rewards[-10:]):.2f}")
    
    return total_rewards


def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate the trained agent"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select best action without exploration
            action = agent.select_action(env, training=False)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward}")
    
    print(f"Average Evaluation Reward: {np.mean(total_rewards):.2f}")
    return total_rewards


def main():
    # Load player performance samples (you would implement this)
    player_samples = {}  # Dictionary: {player_id: {gameweek: [samples]}}
    
    # Load real player points (for evaluation)
    real_player_points = {}  # Dictionary: {player_id: {gameweek: points}}
    
    # Create FPL environment
    env = FPLEnv(player_samples, real_player_points)
    
    # Create Bayesian Q-learning agent
    agent = BayesianFPLAgent()
    
    # Train agent
    print("Training agent...")
    training_rewards = train_agent(env, agent, num_episodes=100)
    
    # Evaluate agent
    print("\nEvaluating agent...")
    eval_rewards = evaluate_agent(env, agent)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Points')
    plt.grid(True)
    plt.savefig('fpl_training_rewards.png')
    plt.show()


if __name__ == "__main__":
    main()