import requests
import numpy as np
from typing import Dict
from constants import GAMEWEEK_COUNT
import math


class FPLEvaluator:
    """Evaluate FPL agent performance."""

    BASE_URL = "https://fantasy.premierleague.com/api"
    SEASON = "2023/24"
    TOTAL_GAMEWEEKS = GAMEWEEK_COUNT

    def __init__(self, fpl_agent_score: int):
        """Initialize with bootstrap-static data which contains essential FPL information."""
        self.bootstrap_data = self._fetch_bootstrap_static()
        self._fpl_agent_score = fpl_agent_score

    def _fetch_bootstrap_static(self) -> Dict:
        """Fetch the bootstrap-static data from the FPL API."""
        url = f"{self.BASE_URL}/bootstrap-static/"
        response = requests.get(url)
        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to fetch data from {url}: {response.status_code}"
            )
        return response.json()

    def get_average_weekly_manager_points(self) -> int:
        """
        Get the total points for the average weekly manager across all gameweeks.

        Returns:
            int: The cumulative points of an average FPL manager for the 2023/24 season.
        """
        total_points = 0
        # Extract gameweek data from bootstrap-static which contains average scores
        events = self.bootstrap_data.get("events", [])

        for event in events:
            # The 'average_entry_score' field contains the average score for the gameweek
            avg_score = event.get("average_entry_score", 0)
            total_points += avg_score

        return total_points

    def get_percentile_ranking(self) -> float:
        """
        Calculate the percentile ranking for the agent's total points by estimating using the
        distribution of scores typically observed
        Args:
            agent_points (int): The total points scored by your FPL agent.

        Returns:
            float: The percentile ranking (0-100) of your agent compared to all managers.
        """
        # Get the average score for the season
        events = self.bootstrap_data.get("events", [])
        average_total_score = sum(
            event.get("average_entry_score", 0) for event in events
        )

        # Get the highest overall score from top manager
        # This would normally come from leaderboard data
        # For estimation, we'll use a typical ratio from average to top score
        estimated_top_score = int(
            average_total_score * 1.5
        )  # Top managers typically score ~50% higher

        # Using a simplified normal distribution to estimate percentile
        # Standard deviation is typically around 15-20% of the average score
        std_dev = average_total_score * 0.18
        # Calculate Z-score
        z_score = (self._fpl_agent_score - average_total_score) / std_dev
        # Convert Z-score to percentile (using cumulative distribution function)
        percentile = 100 * (0.5 + 0.5 * math.erf(z_score / np.sqrt(2)))

        return percentile

    def print_evaluation_summary(self) -> None:
        """
        Get a comprehensive evaluation summary for the agent.

        Args:
            agent_points (int): The total points scored by your FPL agent.

        Returns:
            None
        """
        avg_manager_points = self.get_average_weekly_manager_points()
        percentile = self.get_percentile_ranking()

        print(f"===== FPL Agent Evaluation for {self.SEASON} Season =====")
        print(f"Agent total points: {self._fpl_agent_score}")
        print(f"Average manager total points: {avg_manager_points}")
        print(f"Agent percentile ranking: {percentile}")
        print(f"Points vs average: {self._fpl_agent_score - avg_manager_points}")


if __name__ == "__main__":
    evaluator = FPLEvaluator(1083)
    print(evaluator.get_percentile_ranking())
