"""Quarterback value estimates and starter-change adjustments.

Each QB gets a shrunken EPA-per-dropback value relative to league average, pooled across
seasons with recency decay and shrunk toward replacement level (REPL below average).
A team's 'sample mix' is the dropback-weighted average of the QBs who actually produced the
team's in-season passing numbers, so a starter change is valued as
    (value of this week's starter) - (value of the sample mix)   [EPA per dropback]
and scaled to points downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPL = -0.10          # replacement-level QB: EPA/dropback below league average
K_QB = 200.0          # shrinkage in dropbacks
DECAY = 0.65          # weight multiplier per season back
DB_PER_GAME = 34.0    # typical dropbacks per team-game (post garbage-time filter)


def qb_dropbacks(x: pd.DataFrame) -> pd.DataFrame:
    """Per-dropback rows with the QB id (passer, or rusher on scrambles)."""
    d = x[x.is_db == 1].copy()
    d["qb_id"] = d.passer_player_id.fillna(d.rusher_player_id)
    d = d[d.qb_id.notna()]
    return d


def qb_values(dropbacks_by_season: dict[int, pd.DataFrame], asof_season: int, asof_week: int,
              k: float = K_QB, repl: float = REPL, decay: float = DECAY) -> pd.DataFrame:
    """Shrunken relative EPA/dropback and CPOE for every QB, using data strictly before (asof_season, asof_week)."""
    rows = []
    for season, d in dropbacks_by_season.items():
        if season > asof_season:
            continue
        if season == asof_season:
            d = d[d.week < asof_week]
        if len(d) == 0:
            continue
        lm = d.epa.mean()
        wgt = decay ** (asof_season - season)
        g = d.groupby("qb_id").agg(n=("epa", "size"), epa=("epa", "mean"), cpoe=("cpoe", "mean"),
                                   name=("passer_player_name", lambda s: s.dropna().iloc[0] if s.notna().any() else None))
        g["rel"] = g.epa - lm
        g["w"] = g.n * wgt
        g["season"] = season
        rows.append(g.reset_index())
    if not rows:
        return pd.DataFrame(columns=["qb_id", "name", "n_eff", "rel_raw", "value", "cpoe"]).set_index("qb_id")
    a = pd.concat(rows).rename(columns={"name": "qb_name"})
    a["wrel"] = a.w * a.rel
    a["wcpoe"] = a.w * a.cpoe.fillna(0)
    a = a.sort_values("season")
    out = a.groupby("qb_id").agg(qb_name=("qb_name", "last"), n_eff=("w", "sum"), wrel=("wrel", "sum"), wcpoe=("wcpoe", "sum"))
    out["rel_raw"] = out.wrel / out.n_eff
    out["cpoe"] = out.wcpoe / out.n_eff
    out["value"] = (out.n_eff * out.rel_raw + k * repl) / (out.n_eff + k)
    return out[["qb_name", "n_eff", "rel_raw", "cpoe", "value"]]


def qb_value(values: pd.DataFrame, qb_id, repl: float = REPL) -> float:
    if qb_id is None or (isinstance(qb_id, float) and np.isnan(qb_id)):
        return repl
    if qb_id in values.index:
        return float(values.loc[qb_id, "value"])
    return repl  # never seen: replacement level


def team_sample_mix(dropbacks: pd.DataFrame, values: pd.DataFrame, repl: float = REPL) -> pd.Series:
    """Dropback-weighted QB value of the passers who generated each team's in-season sample."""
    out = {}
    for team, d in dropbacks.groupby("posteam"):
        cnt = d.qb_id.value_counts()
        out[team] = sum(qb_value(values, q, repl) * n for q, n in cnt.items()) / cnt.sum()
    return pd.Series(out)
