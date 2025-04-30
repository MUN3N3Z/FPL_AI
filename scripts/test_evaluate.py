from bayesianFPLAgent import BayesianQLearningAgent
from fpl_env import FPLEnv
from datetime import datetime
import matplotlib.pyplot as plt
from constants import DATA_FOLDER, GAMEWEEK_COUNT
import pandas as pd
import numpy as np
import pytensor
import logging
import pickle
import os
from utils import format_season_name, setup_logging
import pprint
from typing import List
from benchmark_evaluation import FPLEvaluator

pytensor.config.cxx = "/usr/bin/clang++"

# Setup logging
logger = setup_logging(log_level=logging.INFO, log_dir="../model/logs")


def train_agent(env: FPLEnv, num_episodes: int) -> None:
    """Use the env provided to train the Bayesian Q-Learning agent on 2022-23 data"""

    start_time = datetime.now()
    fantasy_agent = BayesianQLearningAgent()
    episode_rewards, gameweek_average_rewards = fantasy_agent.train(env, num_episodes)
    logger.info(f"Training completed in {datetime.now() - start_time}")

    return episode_rewards, gameweek_average_rewards


def evaluate_agent(env: FPLEnv, agent: BayesianQLearningAgent, iterations: int = 30):
    """
    Evaluate the trained agent against 2023-24 data
    """
    start_time = datetime.now()
    total_points, _ = [], None
    total_gameweek_rewards = np.zeros(shape=(GAMEWEEK_COUNT,))

    for _ in range(iterations):
        points, _, gameweek_rewards = agent.evaluate(env)
        total_points.append(points)
        total_gameweek_rewards += np.array(gameweek_rewards)

    logger.info(
        f"Evaluation complete - Average points: {round(sum(total_points) / len(total_points))}"
    )
    logger.info(f"Evaluation completed in {datetime.now() - start_time}")

    return total_points, _, list((total_gameweek_rewards / num_episodes).astype(int))


def plot_training_stats(
    episode_rewards: List[int], gameweek_average_rewards: List[int]
):
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

    plt.figure(figsize=(10, 6))
    plt.plot(gameweek_average_rewards)
    plt.title("Average Gameweek Performance in Training")
    plt.xlabel("Gameweek")
    plt.ylabel("Points")
    plt.grid(True)
    plt.savefig("train_average_gameweek_performance.png")
    plt.close()


def plot_test_stats(gameweek_average_rewards: List[int]):
    plt.figure(figsize=(10, 6))
    plt.plot(gameweek_average_rewards)
    plt.title("Average Gameweek Performance in Testing")
    plt.xlabel("Gameweek")
    plt.ylabel("Points")
    plt.grid(True)
    plt.savefig("test_average_gameweek_performance.png")
    plt.close()


if __name__ == "__main__":
    # Configuration
    train_year = "2022"
    test_year = "2023"
    num_episodes = 30
    discount_factor = 0.5
    num_actions = 3

    trained = False
    formatted_season_name = format_season_name(train_year)
    # Load pickled trained agent if present
    pickled_agent_path = os.path.join(
        DATA_FOLDER, formatted_season_name, "model_data", "trained_agent_0.5.pkl"
    )
    if os.path.exists(pickled_agent_path):
        with open(file=pickled_agent_path, mode="rb") as f:
            agent = pickle.load(f)
    else:
        logger.info("Loading 2022-23 fixtures (training) ...")
        # Load 2022-23 season fixtures
        fixtures_2022_23 = pd.read_csv(
            filepath_or_buffer=f"{DATA_FOLDER}/{formatted_season_name}/fixtures.csv"
        )
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
            init_variance_ratio=0.1,
            episode_limit=num_episodes,
            num_actions=num_actions,
        )
        logger.info(f"Training BayesianQLearningAgent for {num_episodes} episodes ...")
        # Train agent
        episode_rewards, gameweek_average_rewards = train_agent(
            env=train_env, num_episodes=num_episodes
        )
        trained = True
        logger.info("Completed training BayesianQLearningAgent")
        with open(file=pickled_agent_path, mode="wb") as f:
            pickle.dump(agent, file=f)
        logger.info(f"Saved trained agent as a pickle file in {pickled_agent_path}")
        # Plot learning curve
        plot_training_stats(episode_rewards, gameweek_average_rewards)

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
    average_points, decisions, average_gameweek_rewards = evaluate_agent(
        env=test_env, agent=agent
    )

    # Save results
    results = {
        "average_points": average_points,
        # "decisions": decisions,
    }

    if trained:
        logger.info("\n=== Training Summary ===")
        logger.info(f"Number of episodes: {num_episodes}")
        logger.info(f"Discount factor: {discount_factor}")
        logger.info(f"Average points per episode: {np.mean(episode_rewards):.2f}")
        logger.info(f"Maximum points in an episode: {np.max(episode_rewards):.2f}")

    # Display summary
    logger.info("\n=== Evaluation ===")
    logger.info(f"Results: {pprint.pformat(results)}")
    average_points = round(sum(average_points) / len(average_points))
    plot_test_stats(average_gameweek_rewards)
    FPLEvaluator(average_points).print_evaluation_summary()
