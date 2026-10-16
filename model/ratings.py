"""Opponent-adjusted EPA ratings (Bayesian ridge) and market-implied power ratings."""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
         "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",
         "TB", "TEN", "WAS"]
TIDX = {t: i for i, t in enumerate(TEAMS)}
NT = len(TEAMS)


def adjusted_unit_ratings(units: pd.DataFrame, unit: str, prior_off: pd.Series, prior_def: pd.Series,
                          k_plays: float, home_eff: float = 0.0, league_mean: float | None = None):
    """Solve  y = mu + off[t] - def[o] + home_eff*home  by weighted ridge toward the prior.

    units: rows of (posteam, defteam, home, <unit>_epa, <unit>_n).
    k_plays: prior strength in plays; posterior ~ (n*sample + k*prior)/(n+k).
    Returns (off, def) Series in EPA/play units, centred so league mean off = league mean def = 0.
    def is 'EPA allowed above league average' (negative = good defence).
    """
    y = units[f"{unit}_epa"].to_numpy()
    w = units[f"{unit}_n"].to_numpy().astype(float)
    mu = league_mean if league_mean is not None else np.average(y, weights=w)
    y = y - mu - home_eff * (units.home.to_numpy() * 2 - 1) / 2.0
    n = len(units)
    X = np.zeros((n, 2 * NT))
    off_i = units.posteam.map(TIDX).to_numpy()
    def_i = units.defteam.map(TIDX).to_numpy()
    X[np.arange(n), off_i] = 1.0
    X[np.arange(n), NT + def_i] = 1.0  # +def means EPA allowed (so better defence = negative)
    prior = np.concatenate([prior_off.reindex(TEAMS).fillna(0).to_numpy(), prior_def.reindex(TEAMS).fillna(0).to_numpy()])
    XtW = X.T * w
    A = XtW @ X + k_plays * np.eye(2 * NT)
    b = XtW @ y + k_plays * prior
    # sum-to-zero on offence and defence blocks keeps the system identified
    beta = np.linalg.solve(A, b)
    beta[:NT] -= beta[:NT].mean()
    beta[NT:] -= beta[NT:].mean()
    return pd.Series(beta[:NT], index=TEAMS), pd.Series(beta[NT:], index=TEAMS)


def market_implied_ratings(games: pd.DataFrame, hfa: float, ridge: float = 1.0, weights=None) -> pd.Series:
    """Fit team ratings R so that spread_line ~ hfa*(1-neutral) + R[home] - R[away], ridge toward 0.

    Returns points-scale ratings (mean 0). ridge is in 'games' of shrinkage toward zero; optional per-game weights
    (e.g. recency decay) make it a rolling market power rating.
    """
    g = games[games.spread_line.notna()]
    n = len(g)
    X = np.zeros((n, NT))
    X[np.arange(n), g.home_team.map(TIDX).to_numpy()] = 1.0
    X[np.arange(n), g.away_team.map(TIDX).to_numpy()] = -1.0
    y = g.spread_line.to_numpy() - hfa * (1 - g.neutral.to_numpy())
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    XtW = X.T * w
    A = XtW @ X + ridge * np.eye(NT)
    r = np.linalg.solve(A, XtW @ y)
    r -= r.mean()
    return pd.Series(r, index=TEAMS)


def season_unit_ratings(units_all: pd.DataFrame, k_plays_pass: float, k_plays_rush: float,
                        home_eff_pass: float = 0.0, home_eff_rush: float = 0.0):
    """Full-season adjusted ratings with a flat (zero) prior; used to build next season's prior."""
    zero = pd.Series(0.0, index=TEAMS)
    op, dp = adjusted_unit_ratings(units_all, "db", zero, zero, k_plays_pass, home_eff_pass)
    orr, dr = adjusted_unit_ratings(units_all, "rush", zero, zero, k_plays_rush, home_eff_rush)
    return pd.DataFrame({"off_pass": op, "def_pass": dp, "off_rush": orr, "def_rush": dr})
