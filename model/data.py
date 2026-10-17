"""Data loading for the NFL spread model (nflverse sources).

All raw files live in DATA_DIR (downloaded from the nflverse GitHub releases):
  games.csv                      schedule, scores, closing lines, QB starters, rest, roof, weather, referee
  play_by_play_<season>.parquet  nflfastR play-by-play with EPA / success / win probability
  injuries_<season>.csv          weekly official injury reports
  depth_charts_<season>.csv      ESPN daily depth-chart snapshots (2025+) or NFL weekly charts (2024)
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("NFL_DATA_DIR", "/tmp/nfl")
HERE = os.path.dirname(os.path.abspath(__file__))

PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "home_team", "away_team",
    "play_type", "epa", "success", "qb_dropback", "rush", "pass", "qb_kneel", "qb_spike",
    "two_point_attempt", "qtr", "wp", "vegas_wp", "interception", "fumble_lost",
    "passer_player_id", "passer_player_name", "rusher_player_id", "cpoe", "sack", "qb_hit", "air_yards",
    "complete_pass", "pass_attempt", "yards_gained", "penalty_yards", "penalty",
    "season_type",
]


@lru_cache(maxsize=None)
def load_games() -> pd.DataFrame:
    g = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    g = g[g.game_type == "REG"].copy()
    g["neutral"] = (g.location == "Neutral").astype(int)
    g["gameday"] = pd.to_datetime(g.gameday)
    # late-week starting-QB overrides (starter_overrides.csv), resolved to gsis ids through the weekly roster
    ov = load_starter_overrides()
    for r in ov.itertuples():
        ros = load_roster(int(r.season))
        hit = ros[(ros.team == r.team) & (ros.full_name == r.qb_name)]
        if len(hit) == 0:
            continue
        qid = hit.gsis_id.iloc[0]
        for side in ("home", "away"):
            m = (g.season == r.season) & (g.week == r.week) & (g[f"{side}_team"] == r.team)
            g.loc[m, f"{side}_qb_id"] = qid
            g.loc[m, f"{side}_qb_name"] = r.qb_name
    return g


@lru_cache(maxsize=None)
def load_pbp(season: int) -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(DATA_DIR, f"play_by_play_{season}.parquet"), columns=PBP_COLS)
    p = p[p.season_type == "REG"]
    return p


@lru_cache(maxsize=None)
def load_injuries(season: int) -> pd.DataFrame:
    """Official reports, with manual late-week overrides (injury_overrides.csv: team, full_name, report_status)
    applied to the latest week so a Friday ruling can be used before nflverse refreshes."""
    i = pd.read_csv(os.path.join(DATA_DIR, f"injuries_{season}.csv"))
    i = i[i.game_type == "REG"].copy()
    path = os.path.join(HERE, "injury_overrides.csv")
    if os.path.exists(path) and season == int(pd.read_csv(os.path.join(DATA_DIR, "games.csv"), usecols=["season"]).season.max()):
        ov = pd.read_csv(path)
        wk = i.week.max()
        for r in ov.itertuples():
            m = (i.week == wk) & (i.team == r.team) & (i.full_name == r.full_name)
            if m.any():
                i.loc[m, "report_status"] = r.report_status
            else:  # player not on the nflverse report yet: add a stub row
                ros = load_roster(season)
                hit = ros[(ros.team == r.team) & (ros.full_name == r.full_name)]
                i = pd.concat([i, pd.DataFrame([{"season": season, "game_type": "REG", "team": r.team, "week": wk,
                                                 "gsis_id": hit.gsis_id.iloc[0] if len(hit) else None,
                                                 "position": hit.position.iloc[0] if len(hit) else None,
                                                 "full_name": r.full_name, "report_status": r.report_status}])], ignore_index=True)
    return i


@lru_cache(maxsize=None)
def load_roster(season: int) -> pd.DataFrame:
    r = pd.read_csv(os.path.join(DATA_DIR, f"roster_weekly_{season}.csv"), usecols=["season", "week", "team", "position", "full_name", "gsis_id"])
    return r.sort_values("week").drop_duplicates(["team", "gsis_id"], keep="last")


def load_starter_overrides() -> pd.DataFrame:
    path = os.path.join(HERE, "starter_overrides.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["season", "week", "team", "qb_name", "source"])


@lru_cache(maxsize=None)
def load_depth(season: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"depth_charts_{season}.csv")
    d = pd.read_csv(path)
    if "dt" in d.columns:  # ESPN daily snapshots
        d["dt"] = pd.to_datetime(d.dt, utc=True).dt.tz_convert(None)
        d = d[d.pos_rank <= 2][["dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"]]
        d = d.rename(columns={"pos_abb": "pos", "pos_rank": "rank"})
        d["fmt"] = "espn"
    else:  # NFL weekly charts
        d = d[(d.game_type == "REG") & (d.depth_team <= 2)][["week", "club_code", "gsis_id", "full_name", "depth_position", "position", "depth_team"]]
        d = d.rename(columns={"club_code": "team", "full_name": "player_name", "depth_position": "pos", "depth_team": "rank"})
        d["fmt"] = "nfl"
    return d


def load_opening_lines() -> pd.DataFrame:
    return pd.read_csv(os.path.join(HERE, "opening_lines.csv"))


def filter_plays(p: pd.DataFrame, garbage_lo: float = 0.04, garbage_hi: float = 0.96) -> pd.DataFrame:
    """Keep real offensive snaps (pass or run), drop kneels/spikes/2-pt tries and 4th-quarter garbage time."""
    x = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna()].copy()
    x = x[(x.two_point_attempt.fillna(0) == 0)]
    wp = x.vegas_wp.fillna(x.wp)
    garbage = (x.qtr >= 4) & ((wp < garbage_lo) | (wp > garbage_hi))
    x = x[~garbage]
    x["turnover"] = ((x.interception.fillna(0) == 1) | (x.fumble_lost.fillna(0) == 1)).astype(int)
    x["is_db"] = (x.qb_dropback.fillna(0) == 1).astype(int)
    return x


def unit_game_stats(x: pd.DataFrame, to_weight: float = 1.0, to_fill: float = -0.7) -> pd.DataFrame:
    """One row per (game, offense team): dropback & rush EPA/play, success rate, play counts.

    to_weight dampens turnover luck: EPA on interception / lost-fumble plays becomes
    to_weight * epa + (1 - to_weight) * to_fill  (to_fill ~ an ordinary failed play).
    """
    x = x.copy()
    x["epa_t"] = np.where(x.turnover == 1, to_weight * x.epa + (1 - to_weight) * to_fill, x.epa)
    x["home"] = (x.posteam == x.home_team).astype(int)
    grp = x.groupby(["game_id", "season", "week", "posteam", "defteam", "home"])
    out = grp.apply(lambda d: pd.Series({
        "db_epa": d.loc[d.is_db == 1, "epa_t"].mean(),
        "db_n": int((d.is_db == 1).sum()),
        "rush_epa": d.loc[d.is_db == 0, "epa_t"].mean(),
        "rush_n": int((d.is_db == 0).sum()),
        "success": d.success.mean(),
        "epa_raw": d.epa.mean(),
        "plays": len(d),
    }), include_groups=False).reset_index()
    return out
