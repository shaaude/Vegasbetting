"""Situational context: home-field, rest, travel, weather, officiating."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_games

# Stadium time zone (hours from UTC, standard time) for travel adjustments
TZ = {"ARI": -7, "ATL": -5, "BAL": -5, "BUF": -5, "CAR": -5, "CHI": -6, "CIN": -5, "CLE": -5, "DAL": -6, "DEN": -7,
      "DET": -5, "GB": -6, "HOU": -6, "IND": -5, "JAX": -5, "KC": -6, "LA": -8, "LAC": -8, "LV": -8, "MIA": -5,
      "MIN": -6, "NE": -5, "NO": -6, "NYG": -5, "NYJ": -5, "PHI": -5, "PIT": -5, "SEA": -8, "SF": -8, "TB": -5,
      "TEN": -6, "WAS": -5}


def league_hfa(seasons=(2022, 2023, 2024, 2025)) -> float:
    g = load_games()
    g = g[g.season.isin(seasons) & (g.neutral == 0) & g.result.notna()]
    return float(g.result.mean())


def team_hfa(seasons=(2021, 2022, 2023, 2024, 2025), shrink: float = 0.4, cap: float = 1.0) -> pd.Series:
    """Team-specific home edge relative to league average: (mean home margin - mean away margin)/2 over several
    seasons (a team's own quality cancels), centred, shrunk toward zero and capped. Venue effects such as altitude,
    crowd noise and travel burden on visitors show up here."""
    g = load_games()
    g = g[g.season.isin(seasons) & (g.neutral == 0) & g.result.notna()]
    home = g.result.groupby(g.home_team).mean()
    away = (-g.result).groupby(g.away_team).mean()
    raw = (home - away) / 2.0
    raw = raw - raw.mean()
    return (raw * shrink).clip(-cap, cap)


def rest_adjustment(home_rest: float, away_rest: float, per_day: float = 0.12, cap: float = 1.5) -> float:
    """Points to the home side from the rest differential (bye = +, short week = -)."""
    if pd.isna(home_rest) or pd.isna(away_rest):
        return 0.0
    return float(np.clip((home_rest - away_rest) * per_day, -cap, cap))


def travel_adjustment(home: str, away: str, neutral: int, per_zone: float = 0.25) -> float:
    """Points to the home side: away team penalised 'per_zone' for each time zone crossed beyond the first."""
    if neutral:
        return 0.0
    zones = abs(TZ[home] - TZ[away])
    return per_zone * max(0, zones - 1)


def wind_pass_factor(wind, roof) -> float:
    """Scales the passing-rating differential when sustained wind is high (outdoor / open roof only)."""
    if roof in ("dome", "closed") or wind is None or pd.isna(wind):
        return 1.0
    return float(max(0.7, 1.0 - 0.02 * max(0.0, wind - 10.0)))


def referee_profile(pbp_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Penalty yards per game by referee (diagnostic only - used for totals / variance flags, not the spread)."""
    g = load_games()[["game_id", "referee", "season"]]
    rows = []
    for season, p in pbp_by_season.items():
        pen = p.groupby("game_id").penalty_yards.sum()
        rows.append(pen.rename("pen_yds").reset_index())
    pen = pd.concat(rows).merge(g, on="game_id")
    prof = pen.groupby("referee").agg(games=("game_id", "size"), pen_yds_pg=("pen_yds", "mean"))
    prof["pen_vs_avg"] = prof.pen_yds_pg - pen.pen_yds.mean()
    return prof
