import pandas as pd
import numpy as np
import pulp
from typing import Dict, Any, List, Optional
from utils import get_logger

logger = get_logger(__name__)


def generate_fpl_team(
    player_df: pd.DataFrame,
    budget=100.0,
    current_team=None,
    free_transfers=1,
    team_value=None,
):
    """
    Generate an optimal FPL team using PuLP (Linear Programming)

    Parameters:
    -----------
    player_df : DataFrame
        DataFrame containing player information with columns:
        - id: unique player ID (name)
        - position: player position (GK, DEF, MID, FWD)
        - team: player's team
        - price: player's price
        - sampled_points: expected points for the gameweek
    budget : float
        Total budget available (default: 100.0)
    current_team : list
        List of player IDs in the current team (if None, builds from scratch)
    free_transfers : int
        Number of free transfers available (default: 1)
    team_value : float
        Current team value (important if it's greater than budget)

    Returns:
    --------
    dict
        Dictionary containing selected team information
    """
    # First, handle the case where players in current_team are not in player_df
    missing_players = []
    valid_current_team = None

    if current_team is not None:
        # Check if all current team players are in the dataframe
        missing_players = [p for p in current_team if p not in player_df.index]

        if missing_players:
            logger.warning(
                f"The following players from your current team are not in the available players: {missing_players}"
            )

            # Filter out missing players from current_team
            valid_current_team = [p for p in current_team if p in player_df.index]

            # If all players are missing, treat as new team
            if len(valid_current_team) == 0:
                logger.warning(
                    "No valid players remain in current team. Treating as a new team selection."
                )
                valid_current_team = None
        else:
            valid_current_team = (
                current_team.copy()
                if isinstance(current_team, list)
                else list(current_team)
            )

    # Enforce unique players
    player_df = player_df.sort_values(by="sampled_points", ascending=False)
    player_df = (
        player_df.reset_index().drop_duplicates(subset=["index"]).set_index("index")
    )

    # Create the PuLP model
    model = pulp.LpProblem(name="FPL_Team_Selection", sense=pulp.LpMaximize)

    # Create decision variables for each player (binary: 0 or 1)
    x = {
        player_id: pulp.LpVariable(name=f"x_{player_id}", cat="Binary")
        for player_id in player_df.index
    }

    # Create variables for starting 11 players (binary: 0 or 1)
    starter = {
        player_id: pulp.LpVariable(name=f"starter_{player_id}", cat="Binary")
        for player_id in player_df.index
    }

    # Create variables for transfers (only for valid current team players)
    if valid_current_team is not None:
        # Variable for discretionary transfers (limited by free_transfers)
        discretionary_transfers = pulp.LpVariable(
            name="discretionary_transfers", lowBound=0, cat="Integer"
        )

        # Variable for actual transfers made (counting only valid players, not missing ones)
        actual_transfers = pulp.LpVariable(
            name="actual_transfers", lowBound=0, cat="Integer"
        )

        # Variable for extra transfers beyond free allowance
        extra_transfers = pulp.LpVariable(
            name="extra_transfers", lowBound=0, cat="Integer"
        )

    # Set objective: Maximize expected points
    if valid_current_team is not None:
        # Consider transfer penalties (-4 points per extra transfer)
        model += (
            pulp.lpSum(
                [
                    player_df.loc[player_id, "sampled_points"] * starter[player_id]
                    for player_id in player_df.index
                ]
            )
            - 4 * extra_transfers
        )
    else:
        # Just maximize expected points for initial team
        model += pulp.lpSum(
            [
                player_df.loc[player_id, "sampled_points"] * starter[player_id]
                for player_id in player_df.index
            ]
        )

    # CONSTRAINTS
    # 1. Squad size: Exactly 15 players
    model += (
        pulp.lpSum([x[player_id] for player_id in player_df.index]) == 15,
        "squad_size",
    )

    # 2. Budget constraint
    model += (
        pulp.lpSum(
            [
                player_df.loc[player_id, "price"] * x[player_id]
                for player_id in player_df.index
            ]
        )
        <= budget,
        "budget",
    )

    # 3. Position constraints
    # 2 Goalkeepers
    model += (
        pulp.lpSum(
            [
                x[player_id]
                for player_id in player_df[player_df["position"] == "GK"].index
            ]
        )
        == 2,
        "gk_count",
    )

    # 5 Defenders
    model += (
        pulp.lpSum(
            [
                x[player_id]
                for player_id in player_df[player_df["position"] == "DEF"].index
            ]
        )
        == 5,
        "def_count",
    )

    # 5 Midfielders
    model += (
        pulp.lpSum(
            [
                x[player_id]
                for player_id in player_df[player_df["position"] == "MID"].index
            ]
        )
        == 5,
        "mid_count",
    )

    # 3 Forwards
    model += (
        pulp.lpSum(
            [
                x[player_id]
                for player_id in player_df[player_df["position"] == "FWD"].index
            ]
        )
        == 3,
        "fwd_count",
    )

    # 4. Team constraint: Maximum 3 players from each team
    for team in player_df["team"].unique():
        model += (
            pulp.lpSum(
                [
                    x[player_id]
                    for player_id in player_df[player_df["team"] == team].index
                ]
            )
            <= 3,
            f"max_players_from_{team}",
        )

    # 5. Starting 11 constraints
    # Only selected players can be starters
    for player_id in player_df.index:
        model += (starter[player_id] <= x[player_id], f"starter_selected_{player_id}")

    # Exactly 11 starters
    model += (
        pulp.lpSum([starter[player_id] for player_id in player_df.index]) == 11,
        "starting_11_count",
    )

    # Valid formation constraints
    # At least 1 goalkeeper
    model += (
        pulp.lpSum(
            [
                starter[player_id]
                for player_id in player_df[player_df["position"] == "GK"].index
            ]
        )
        == 1,
        "starting_gk_count",
    )

    # At least 3 defenders
    model += (
        pulp.lpSum(
            [
                starter[player_id]
                for player_id in player_df[player_df["position"] == "DEF"].index
            ]
        )
        >= 3,
        "min_starting_def",
    )

    # At least 2 midfielders
    model += (
        pulp.lpSum(
            [
                starter[player_id]
                for player_id in player_df[player_df["position"] == "MID"].index
            ]
        )
        >= 2,
        "min_starting_mid",
    )

    # At least 1 forward
    model += (
        pulp.lpSum(
            [
                starter[player_id]
                for player_id in player_df[player_df["position"] == "FWD"].index
            ]
        )
        >= 1,
        "min_starting_fwd",
    )

    # 6. Transfer constraints (only for valid current team players)
    if valid_current_team is not None:
        # Count actual transfers (players dropping out of valid_current_team)
        model += (
            actual_transfers
            == pulp.lpSum([1 - x[player_id] for player_id in valid_current_team]),
            "actual_transfers_count",
        )

        # Only one discretionary transfer (the rest are forced due to missing players)
        # Note: We use '<=1' instead of '==1' to allow for no discretionary transfers if needed
        model += (
            discretionary_transfers <= 1,
            "max_discretionary_transfers",
        )

        # Discretionary transfers can't exceed actual transfers
        model += (
            discretionary_transfers <= actual_transfers,
            "discretionary_vs_actual",
        )

        # Extra transfers = actual transfers - free transfers - forced transfers
        # Where forced transfers = number of missing players
        forced_transfers = len(missing_players)
        available_free_transfers = max(0, free_transfers - forced_transfers)

        model += (
            extra_transfers
            >= actual_transfers - discretionary_transfers - available_free_transfers,
            "extra_transfers_count",
        )

    # Solve the model using the default CBC solver
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if solution was found
    if model.status != pulp.LpStatusOptimal:
        logger.warning(
            "No optimal solution found. Attempting to solve with relaxed constraints."
        )

        # Try again with relaxed constraints if we have a current team
        if valid_current_team is not None:
            # Start fresh
            return generate_fpl_team(
                player_df=player_df,
                budget=budget,
                current_team=None,  # Treat as a new team
                free_transfers=free_transfers,
                team_value=team_value,
            )
        else:
            logger.error("Failed to find a solution even with relaxed constraints.")
            return None

    # Extract results
    selected_players = [
        player_id for player_id in player_df.index if pulp.value(x[player_id]) > 0.5
    ]
    starting_players = [
        player_id
        for player_id in player_df.index
        if pulp.value(starter[player_id]) > 0.5
    ]

    # Order starting lineup in a reasonable formation
    # GK, DEF, MID, FWD
    ordered_lineup = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_players = player_df[
            (player_df.index.isin(starting_players)) & (player_df["position"] == pos)
        ].sort_values("sampled_points", ascending=False)
        ordered_lineup.extend(pos_players.index.tolist())

    # Order bench
    bench = [p for p in selected_players if p not in starting_players]
    ordered_bench = []
    # First GK
    gk_bench = player_df[
        (player_df.index.isin(bench)) & (player_df["position"] == "GK")
    ].index.tolist()
    ordered_bench.extend(gk_bench)

    # Then outfield by expected points
    outfield_bench = player_df[
        (player_df.index.isin(bench)) & (player_df["position"] != "GK")
    ].sort_values("sampled_points", ascending=False)
    ordered_bench.extend(outfield_bench.index.tolist())

    # Calculate team stats
    team_cost = sum(player_df.loc[player_id, "price"] for player_id in selected_players)
    expected_points = sum(
        player_df.loc[player_id, "sampled_points"] for player_id in starting_players
    )

    # Transfer statistics
    if valid_current_team is not None:
        # Get actual number of transfers made
        actual_transfers_made = int(pulp.value(actual_transfers))

        # Get discretionary transfers (the ones we chose to make)
        discretionary_transfers_made = int(pulp.value(discretionary_transfers))

        # Get forced transfers (due to missing players)
        forced_transfers_made = len(missing_players)

        # Calculate extra transfers beyond free allowance
        extra_transfers_made = int(pulp.value(extra_transfers))

        # Calculate transfer penalty (only for discretionary transfers beyond allowance)
        transfer_penalty = 4 * extra_transfers_made
        expected_points -= transfer_penalty

        # Total transfers including missing players
        total_transfers = actual_transfers_made + forced_transfers_made
    else:
        actual_transfers_made = 0
        discretionary_transfers_made = 0
        forced_transfers_made = 0
        extra_transfers_made = 0
        transfer_penalty = 0
        total_transfers = 0

    # Identify best captain (highest expected points)
    captain_id = (
        player_df.loc[player_df.index.isin(starting_players)]
        .sort_values("sampled_points", ascending=False)
        .index[0]
    )
    vice_captain_id = (
        player_df.loc[player_df.index.isin(starting_players)]
        .sort_values("sampled_points", ascending=False)
        .index[1]
    )

    result = {
        "squad": selected_players,
        "lineup": ordered_lineup,
        "bench": ordered_bench,
        "captain": captain_id,
        "vice_captain": vice_captain_id,
        "team_cost": team_cost,
        "expected_points": expected_points,
    }

    if current_team is not None:
        result[
            "transfers"
        ] = discretionary_transfers_made  # Only count discretionary transfers
        result[
            "forced_transfers"
        ] = forced_transfers_made  # Count forced transfers separately
        result[
            "total_transfers"
        ] = total_transfers  # Total transfers including forced ones
        result["transfer_penalty"] = transfer_penalty

        # Determine which players are coming in
        players_in = [p for p in selected_players if p not in current_team]

        # Determine which players are going out (including missing players)
        players_out = [p for p in current_team if p not in selected_players]

        result["players_in"] = players_in
        result["players_out"] = players_out

    return result


def generate_multiple_teams(
    player_df,
    num_teams=3,
    samples_per_team=20,
    budget=100.0,
    current_team=None,
    free_transfers=1,
) -> List[Dict[str, Any]]:
    """
    Generate multiple candidate teams

    Parameters:
    -----------
    player_df : DataFrame
        DataFrame with all player data
    num_teams : int
        Number of candidate teams to generate
    samples_per_team : int
        Number of samples to use for each team generation
    budget : float
        Total budget available
    current_team : list
        List of player IDs in the current team
    free_transfers : int
        Number of free transfers available

    Returns:
    --------
    list
        List of team dictionaries sorted by expected points
    """
    teams = []

    # Safety check for current_team - make sure it's a list if provided
    if current_team is not None and not isinstance(current_team, list):
        try:
            current_team = list(current_team)
        except Exception as e:
            logger.warning(
                f"Failed to convert current_team to list: {e}. Treating as a new team selection."
            )
            current_team = None

    # First attempt: try with current team
    attempts_with_current = 0
    max_attempts_with_current = 3

    while attempts_with_current < max_attempts_with_current and len(teams) < num_teams:
        try:
            # For each team, resample the expected points
            # This introduces variability as suggested in the paper
            if samples_per_team > 1:
                # Resample points based on uncertainty in player performance
                sampled_df = player_df.copy()
                # Add noise to expected points to simulate different sampling outcomes
                noise = np.random.normal(0, 0.5, len(sampled_df))
                sampled_df["sampled_points"] = sampled_df["sampled_points"] + noise
                # Ensure no negative expected points
                sampled_df["sampled_points"] = sampled_df["sampled_points"].clip(
                    lower=0
                )
            else:
                sampled_df = player_df

            # Generate team with this sample
            team = generate_fpl_team(
                sampled_df,
                budget=budget,
                current_team=current_team,
                free_transfers=free_transfers,
            )

            if team is not None:
                teams.append(team)

        except Exception as e:
            logger.warning(f"Error generating team with current squad: {e}")

        attempts_with_current += 1

    # If we couldn't generate enough teams with the current squad, try without it
    if len(teams) < num_teams:
        logger.info(
            f"Only generated {len(teams)} teams with current squad. Trying without current squad constraints."
        )

        remaining_teams = num_teams - len(teams)
        for i in range(remaining_teams):
            try:
                # Resample points
                if samples_per_team > 1:
                    sampled_df = player_df.copy()
                    noise = np.random.normal(0, 0.5, len(sampled_df))
                    sampled_df["sampled_points"] = sampled_df["sampled_points"] + noise
                    sampled_df["sampled_points"] = sampled_df["sampled_points"].clip(
                        lower=0
                    )
                else:
                    sampled_df = player_df

                # Generate team without current squad constraints
                team = generate_fpl_team(
                    sampled_df,
                    budget=budget,
                    current_team=None,  # No current team constraint
                    free_transfers=free_transfers,
                )

                if team is not None:
                    team["note"] = "Generated without current team constraints"
                    teams.append(team)
            except Exception as e:
                logger.warning(f"Error generating team without current squad: {e}")

    # If we still have no teams, try one last time with a different approach
    if len(teams) == 0:
        logger.warning(
            "Failed to generate any teams. Making one final attempt with simplified constraints."
        )
        try:
            # Try one more time with simplified constraints
            team = generate_fpl_team(
                player_df,
                budget=budget + 5,  # Slightly relaxed budget
                current_team=None,  # No current team constraint
                free_transfers=free_transfers,
            )

            if team is not None:
                team["note"] = "Generated with relaxed constraints"
                teams.append(team)
        except Exception as e:
            logger.error(f"Failed to generate any teams: {e}")

    # Sort teams by expected points
    teams.sort(key=lambda x: x["expected_points"], reverse=True)

    if len(teams) == 0:
        logger.error("Could not generate any valid teams")
    else:
        logger.info(f"Successfully generated {len(teams)} teams")

    return teams


def display_team(team: Dict[str, Any], player_df: pd.DataFrame):
    """Display the selected team in a readable format"""
    logger.info("\n==== SELECTED TEAM ====")
    if "note" in team:
        logger.info(f"Note: {team['note']}")
    logger.info(f"Expected Points: {team['expected_points']:.2f}")
    logger.info(f"Team Cost: £{team['team_cost']:.1f}m")

    if "transfers" in team:
        if "forced_transfers" in team and team["forced_transfers"] > 0:
            logger.info(f"Discretionary Transfers: {team['transfers']}")
            logger.info(
                f"Forced Transfers (missing players): {team['forced_transfers']}"
            )
            logger.info(f"Total Transfers: {team['total_transfers']}")
        else:
            logger.info(f"Transfers Made: {team['transfers']}")

        if team["transfer_penalty"] > 0:
            logger.info(f"Transfer Penalty: {team['transfer_penalty']} points")

    logger.info("\n--- Starting Lineup ---")
    logger.info("Captain: " + team["captain"])
    logger.info("Vice Captain: " + team["vice_captain"])

    positions = ["GK", "DEF", "MID", "FWD"]
    for pos in positions:
        pos_players = player_df[
            (player_df.index.isin(team["lineup"])) & (player_df["position"] == pos)
        ]
        logger.info(f"\n{pos}:")
        for player_name, player in pos_players.iterrows():
            captain_mark = (
                " (C)"
                if player_name == team["captain"]
                else " (V)"
                if player_name == team["vice_captain"]
                else ""
            )
            logger.info(
                f"  {player_name} - {player['team']} - £{player['price']}m - {player['sampled_points']:.2f}pts{captain_mark}"
            )

    logger.info("\n--- Bench ---")
    for player_id in team["bench"]:
        if player_id in player_df.index:
            player = player_df.loc[player_id]
            logger.info(
                f"  {player_id} - {player['position']} - {player['team']} - £{player['price']}m - {player['sampled_points']:.2f}pts"
            )
        else:
            logger.warning(
                f"  {player_id} - DATA MISSING (player not in current dataframe)"
            )

    if "players_in" in team and team["players_in"]:
        logger.info("\n--- Transfers In ---")
        for player_id in team["players_in"]:
            if player_id in player_df.index:
                player = player_df.loc[player_id]
                logger.info(f"  {player_id} - {player['position']} - {player['team']}")
            else:
                logger.warning(
                    f"  {player_id} - DATA MISSING (player not in current dataframe)"
                )

    if "players_out" in team and team["players_out"]:
        logger.info("\n--- Transfers Out ---")
        for player_id in team["players_out"]:
            if player_id in player_df.index:
                player = player_df.loc[player_id]
                logger.info(f"  {player_id} - {player['position']} - {player['team']}")
            else:
                logger.warning(
                    f"  {player_id} - DATA MISSING (player not available in dataset)"
                )
