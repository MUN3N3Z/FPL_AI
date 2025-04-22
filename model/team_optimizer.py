import pandas as pd
import numpy as np
import pulp
from typing import Dict, Any


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

    # Create variables for transfers
    if current_team is not None:
        transfers_in = pulp.LpVariable(name="transfers_in", lowBound=0, cat="Integer")
        transfers_out = pulp.LpVariable(name="transfers_out", lowBound=0, cat="Integer")
        extra_transfers = pulp.LpVariable(
            name="extra_transfers", lowBound=0, cat="Integer"
        )

    # Set objective: Maximize expected points
    if current_team is not None:
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

    # 6. Transfer constraints (if updating an existing team)
    if current_team is not None:
        # Count players transferred out
        model += (
            transfers_out
            == pulp.lpSum([1 - x[player_id] for player_id in current_team]),
            "transfers_out_count",
        )

        # Count players transferred in (must equal transfers out)
        model += (transfers_in == transfers_out, "transfers_balance")

        # Calculate extra transfers
        model += (
            extra_transfers >= transfers_in - free_transfers,
            "extra_transfers_count",
        )

    # Solve the model using the default CBC solver
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if solution was found
    if model.status != pulp.LpStatusOptimal:
        print("No solution found!")
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

    if current_team is not None:
        transfers_made = int(pulp.value(transfers_in))
        extra_transfers_made = int(pulp.value(extra_transfers))
        transfer_penalty = 4 * extra_transfers_made
        expected_points -= transfer_penalty

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
        result["transfers"] = transfers_made
        result["transfer_penalty"] = transfer_penalty
        result["players_in"] = [p for p in selected_players if p not in current_team]
        result["players_out"] = [p for p in current_team if p not in selected_players]

    return result


def generate_multiple_teams(
    player_df,
    num_teams=3,
    samples_per_team=20,
    budget=100.0,
    current_team=None,
    free_transfers=1,
):
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

    for i in range(num_teams):
        # For each team, resample the expected points
        # This introduces variability as suggested in the paper
        if samples_per_team > 1:
            # Resample points based on uncertainty in player performance
            # This is a simplified version - in reality, you'd use your Bayesian model
            sampled_df = player_df.copy()
            # Add noise to expected points to simulate different sampling outcomes
            noise = np.random.normal(0, 0.5, len(sampled_df))
            sampled_df["sampled_points"] = sampled_df["sampled_points"] + noise
            # Ensure no negative expected points
            sampled_df["sampled_points"] = sampled_df["sampled_points"].clip(lower=0)
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

    # Sort teams by expected points
    teams.sort(key=lambda x: x["expected_points"], reverse=True)
    return teams


def display_team(team: Dict[str, Any], player_df: pd.DataFrame):
    """Display the selected team in a readable format"""
    print("\n==== SELECTED TEAM ====")
    print(f"Expected Points: {team['expected_points']:.2f}")
    print(f"Team Cost: £{team['team_cost']:.1f}m")

    if "transfers" in team:
        print(f"Transfers Made: {team['transfers']}")
        if team["transfer_penalty"] > 0:
            print(f"Transfer Penalty: {team['transfer_penalty']} points")

    print("\n--- Starting Lineup ---")
    print("Captain: " + team["captain"])
    print("Vice Captain: " + team["vice_captain"])

    positions = ["GK", "DEF", "MID", "FWD"]
    for pos in positions:
        pos_players = player_df[
            (player_df.index.isin(team["lineup"])) & (player_df["position"] == pos)
        ]
        print(f"\n{pos}:")
        for player_name, player in pos_players.iterrows():
            captain_mark = (
                " (C)"
                if player_name == team["captain"]
                else " (V)"
                if player_name == team["vice_captain"]
                else ""
            )
            print(
                f"  {player_name} - {player['team']} - £{player['price']}m - {player['sampled_points']:.2f}pts{captain_mark}"
            )

    print("\n--- Bench ---")
    for player_id in team["bench"]:
        player = player_df.loc[player_id].iloc[0]
        print(
            f"  {player_id} - {player['position']} - {player['team']} - £{player['price']}m - {player['sampled_points']:.2f}pts"
        )

    if "players_in" in team and team["players_in"]:
        print("\n--- Transfers In ---")
        for player_id in team["players_in"]:
            player = player_df.loc[player_id].iloc[0]
            print(f"  {player_id} - {player['position']} - {player['team']}")

    if "players_out" in team and team["players_out"]:
        print("\n--- Transfers Out ---")
        for player_id in team["players_out"]:
            player = player_df.loc[player_id].iloc[0]
            print(f"  {player_id} - {player['position']} - {player['team']}")
