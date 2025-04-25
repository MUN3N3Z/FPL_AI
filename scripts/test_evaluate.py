from bayesianFPLAgent import BayesianQLearningAgent
from fpl_env import FPLEnv
from datetime import datetime
import matplotlib.pyplot as plt
from constants import DATA_FOLDER
import pandas as pd
import numpy as np
import pytensor
import logging
import pickle
import os
from utils import format_season_name, setup_logging

pytensor.config.cxx = "/usr/bin/clang++"

# Setup logging
logger = setup_logging(log_level=logging.INFO, log_dir="../model/logs")


def train_agent(env: FPLEnv, num_episodes: int) -> None:
    """Use the env provided to train the Bayesian Q-Learning agent on 2022-23 data"""

    start_time = datetime.now()
    fantasy_agent = BayesianQLearningAgent()
    episode_rewards = fantasy_agent.train(env, num_episodes)
    logger.info(f"Training completed in {datetime.now() - start_time}")

    return episode_rewards


def evaluate_agent(env: FPLEnv, agent: BayesianQLearningAgent):
    """
    Evaluate the trained agent against 2023-24 data
    """
    total_points, decisions = agent.evaluate(env)

    logger.info(f"Evaluation complete - Total points: {total_points}")

    return total_points, decisions


def plot_learning_curve(episode_rewards):
    """
    Plot the learning curve of the agent
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Points)")
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.close()

    # Also plot cumulative rewards
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(episode_rewards))
    plt.title("Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Points")
    plt.grid(True)
    plt.savefig("cumulative_reward.png")
    plt.close()


if __name__ == "__main__":
    # Configuration
    train_year = "2022"
    test_year = "2023"
    num_episodes = 50
    search_depth = 3
    discount_factor = 0.5

    logger.info("Loading 2022-23 fixtures (training) ...")
    formatted_season_name = format_season_name(train_year)
    # Load 2022-23 season fixtures
    fixtures_2022_23 = pd.read_csv(
        filepath_or_buffer=f"{DATA_FOLDER}/{formatted_season_name}/fixtures.csv"
    )

    # Load pickled trained agent if present
    pickled_agent_path = os.path.join(
        DATA_FOLDER, formatted_season_name, "model_data", "trained_agent.pkl"
    )
    if os.path.exists(pickled_agent_path):
        with open(file=pickled_agent_path, mode="rb") as f:
            agent = pickle.load(f)
    else:
        logger.info("Initializing FPLEnv ...")
        # Create environment
        train_env = FPLEnv(
            season_start_year=train_year,
            fixtures_data=fixtures_2022_23,
            total_gameweeks=38,
            current_gameweek=1,
            budget=100.0,
            render_mode="human",
        )
        logger.info("Initializing BayesianQLearningAgent ...")
        # Create agent
        agent = BayesianQLearningAgent(
            discount_factor=discount_factor,
            search_depth=search_depth,
            init_variance_ratio=0.1,
            episode_limit=num_episodes,
            num_actions=3,
        )
        logger.info(f"Training BayesianQLearningAgent for {num_episodes} episodes ...")
        # Train agent
        episode_rewards = train_agent(env=train_env, num_episodes=num_episodes)
        logger.info("Completed training BayesianQLearningAgent")
        with open(file=pickled_agent_path, mode="wb") as f:
            pickle.dump(agent, file=f)
        logger.info(f"Saved trained agent as a pickle file in {pickled_agent_path}")
        # Plot learning curve
        plot_learning_curve(episode_rewards)

    logger.info("Loading 2023-24 fixtures (evaluation) ...")

    # Load 2022-23 season fixtures
    fixtures_2023_24 = pd.read_csv(
        filepath_or_buffer=f"{DATA_FOLDER}/{format_season_name(test_year)}/fixtures.csv"
    )

    logging.info("Initializing FPLEnv ...")
    # Create environment
    test_env = FPLEnv(
        season_start_year=test_year,
        fixtures_data=fixtures_2023_24,
        total_gameweeks=38,
        current_gameweek=1,
        budget=100.0,
        render_mode="human",
    )
    logging.info("Evaluating agent ...")
    # Evaluate agent
    total_points, decisions = evaluate_agent(env=test_env, agent=agent)

    # Save results
    results = {
        "total_points": total_points,
        "episode_rewards": episode_rewards,
        "decisions": decisions,
    }

    # Display summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Number of episodes: {num_episodes}")
    logger.info(f"Discount factor: {discount_factor}")
    logger.info(f"Search depth: {search_depth}")
    logger.info(f"Average points per episode: {np.mean(episode_rewards):.2f}")
    logger.info(f"Maximum points in an episode: {np.max(episode_rewards):.2f}")

    logger.info("\n=== Evaluation ===")
    logger.info(f"Total FPL points: {total_points}")

    logger.info(results)
