import numpy as np
import pandas as pd
from data_registry import DataRegistry
from utils import team_to_id_converter_2023_24, format_season_name


class PlayerAbility:
    """
    - Represents beliefs on player abilities modelled as Bayesian priors
    - Priors are defined as conjugates as follows:
        - ρ (rho): Playing status probabilities (Dirichlet-Multinomial conjugate prior) - (played, subbed, not_used)
        - ω (omega): Goal scoring ability (Beta-Binomial conjugate prior)
        - ψ (psi): Assist making ability (Beta-Binomial conjugate prior)
    """

    def __init__(self, season_start_year: str, gameweek: str):
        self._season = season_start_year
        self._gameweek = int(gameweek)
        self._seasonal_gameweek_player_data = DataRegistry(
            gw_data_columns=[
                "name",
                "minutes",
                "starts",
                "goals_scored",
                "assists",
                "position",
                "team",
                "GW",
                "value",
                "total_points",
            ]
        )
        self.player_ability = self._create_model()

    def _create_model(self) -> pd.DataFrame:
        """
        Creates ability priors for players in the 'self._season' season using historical gameweek data.

        This function iteratively updates priors for player abilities (playing time,
        goal scoring, and assist making) using Bayesian inference and historical
        data from the 2016/17 season to the last completed gameweek as passed in 'self.gameweek'. The priors are updated sequentially for each
        season, using Beta-Binomial (goal scoring and assist making) and Dirichlet-Multinomial (probability of: starting, substitution, not playing) conjugates.

        Returns: A Pandas DataFrame of the trained priors
        """

        current_season = format_season_name(self._season)
        current_season_gw_data = self._seasonal_gameweek_player_data.gameweek_data[
            current_season
        ]
        preceeding_seasons_data = (
            self._seasonal_gameweek_player_data.gameweek_data.copy()
        )
        preceeding_seasons_data.pop(current_season)
        compiled_gw_data = list(preceeding_seasons_data.values())
        if self._gameweek:
            for gameweeks in range(1, self._gameweek):
                # Include preceeding gameweeks' data  from current season
                compiled_gw_data.append(
                    current_season_gw_data[
                        current_season_gw_data["GW"] == int(gameweeks)
                    ]
                )

        compiled_gw_data = pd.concat(compiled_gw_data)
        current_season_players = current_season_gw_data.drop_duplicates(
            subset=["name"], inplace=False
        )
        current_season_players.index = current_season_players["name"]
        # Priors obtained from https://cdn.aaai.org/ojs/8259/8259-13-11787-1-2-20201228.pdf
        player_priors = {
            player_row["name"]: {
                "team": team_to_id_converter_2023_24(player_row["team"]),
                "position": player_row["position"],
                "ρ_β": np.array([0.25, 0.25, 0.5]),
                "ω_α": 0.0,
                "ω_β": 5.0,
                "ψ_α": 0.0,
                "ψ_β": 5.0,
            }
            for _, player_row in current_season_players.iterrows()
        }

        # Filter out players not present in the 2023/24 season
        compiled_gw_data = compiled_gw_data[
            compiled_gw_data["name"].isin(current_season_players["name"])
        ]
        grouped_compiled_data = compiled_gw_data.groupby(["name"])
        print(
            f"Player abilities matched players: {len(grouped_compiled_data['name'].unique())}"
        )

        for (player_name,), player_data in grouped_compiled_data:
            ρ_observed = np.array(
                player_data.apply(
                    lambda row: (
                        [1, 0, 0]
                        if row["starts"] == 1
                        else [0, 1, 0]
                        if row["minutes"] > 0
                        else [0, 0, 1]
                    ),
                    axis=1,
                ).tolist()
            ).sum(axis=0)
            ω_observed = player_data["goals_scored"].sum()
            ψ_observed = player_data["assists"].sum()
            total_gameweeks_played = (
                ρ_observed[0] + ρ_observed[1]
            )  # Games where player started or got subbed in

            # Conjugate prior updates
            # Dirichlet prior update for ρ
            player_priors[player_name]["ρ_β"] += ρ_observed

            # Beta prior update for ω
            player_priors[player_name]["ω_α"] += ω_observed
            player_priors[player_name]["ω_β"] += total_gameweeks_played - ω_observed
            # Beta prior update for ψ
            player_priors[player_name]["ψ_α"] += ψ_observed
            player_priors[player_name]["ψ_β"] += total_gameweeks_played - ψ_observed

        # Save player_ability_df as a csv file
        player_ability_df = pd.DataFrame(player_priors)
        player_ability_df = player_ability_df.T.reset_index()
        player_ability_df.rename(columns={"index": "name"}, inplace=True)

        return player_ability_df

    def update_model(self) -> None:
        """Updates self.player_ability with data from the next gameweek (self._gameweek + 1)"""
        self._gameweek += 1
        current_season_data = self._seasonal_gameweek_player_data.gameweek_data[
            self._season
        ]
        current_gameweek_data = current_season_data[
            current_season_data["GW"] == self._gameweek
        ]
        current_gameweek_data.index = current_gameweek_data["name"]
        for _, player_row in self.player_ability.iterrows():
            self.player_ability.loc[player_row["name"], "ρ_β"] += (
                [1, 0, 0]
                if player_row["starts"] == 1
                else [0, 1, 0]
                if player_row["minutes"] > 0
                else [0, 0, 1]
            )
            # Beta prior update for ω
            ω_observed = current_gameweek_data.loc[player_row["name"], "goals_scored"]
            self.player_ability[player_row["name"], "ω_α"] += ω_observed
            self.player_ability[player_row["name"], "ω_β"] += 1 - ω_observed
            # Beta prior update for ψ
            ψ_observed = current_gameweek_data.loc[player_row["name"], "assists"]
            self.player_ability[player_row["name"], "ψ_α"] += ψ_observed
            self.player_ability[player_row["name"], "ψ_β"] += 1 - ψ_observed

        return
