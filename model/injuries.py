"""Position-weighted injury index for non-QB starters.

For every team-game we build an index = sum over starters of (position weight x probability out),
where starters come from the depth chart in force the week before the game (so a player who was
already replaced on the chart is not double-counted) and 'out' comes from the official report.
Only the DIFFERENCE between this week's index and the average index across the team's sample games
is applied, so a long-term absence that is already baked into the team's EPA sample is not charged twice.
Point weights are priors (approximate points per game lost when the starter is out); they are
scaled by a single tuned factor downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ESPN depth-chart slot -> (group, weight in points)
ESPN_POS = {
    "LT": ("OL", 1.0), "RT": ("OL", 0.6), "LG": ("OL", 0.35), "RG": ("OL", 0.35), "C": ("OL", 0.4),
    "WR": ("WR", 0.55), "TE": ("TE", 0.4), "RB": ("RB", 0.4), "FB": ("FB", 0.05),
    "LDE": ("EDGE", 0.7), "RDE": ("EDGE", 0.7), "LDT": ("DL", 0.4), "RDT": ("DL", 0.4), "NT": ("DL", 0.4),
    "WLB": ("LB", 0.3), "SLB": ("LB", 0.3), "MLB": ("LB", 0.35), "LILB": ("LB", 0.3), "RILB": ("LB", 0.3),
    "LCB": ("CB", 0.6), "RCB": ("CB", 0.6), "NB": ("CB", 0.35), "FS": ("S", 0.4), "SS": ("S", 0.4),
    "PK": ("K", 0.4), "P": ("P", 0.15),
}
# NFL weekly chart depth_position (2024 format) -> same scheme
NFL_POS = {
    "LT": ("OL", 1.0), "RT": ("OL", 0.6), "LG": ("OL", 0.35), "RG": ("OL", 0.35), "C": ("OL", 0.4),
    "WR": ("WR", 0.55), "TE": ("TE", 0.4), "RB": ("RB", 0.4), "FB": ("FB", 0.05),
    "DE": ("EDGE", 0.7), "LDE": ("EDGE", 0.7), "RDE": ("EDGE", 0.7), "OLB": ("EDGE", 0.5), "LOLB": ("EDGE", 0.5), "ROLB": ("EDGE", 0.5),
    "DT": ("DL", 0.4), "NT": ("DL", 0.4), "LDT": ("DL", 0.4), "RDT": ("DL", 0.4),
    "LB": ("LB", 0.3), "MLB": ("LB", 0.35), "ILB": ("LB", 0.3), "LILB": ("LB", 0.3), "RILB": ("LB", 0.3), "WLB": ("LB", 0.3), "SLB": ("LB", 0.3),
    "CB": ("CB", 0.6), "LCB": ("CB", 0.6), "RCB": ("CB", 0.6), "NB": ("CB", 0.35), "NCB": ("CB", 0.35),
    "FS": ("S", 0.4), "SS": ("S", 0.4), "S": ("S", 0.4), "K": ("K", 0.4), "PK": ("K", 0.4), "P": ("P", 0.15),
}
STATUS_P_OUT = {"Out": 1.0, "Doubtful": 0.8, "Questionable": 0.3}
# before the final (Friday) report is posted, fall back to the latest practice participation
PRACTICE_P_OUT = {"Did Not Participate In Practice": 0.5, "Limited Participation in Practice": 0.15,
                  "Full Participation in Practice": 0.0}


def p_out(report_status, practice_status) -> float:
    if isinstance(report_status, str) and report_status in STATUS_P_OUT:
        return STATUS_P_OUT[report_status]
    if isinstance(practice_status, str):
        return PRACTICE_P_OUT.get(practice_status, 0.0)
    return 0.0


def chart_for_game(depth: pd.DataFrame, team: str, week: int, gameday: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Depth chart (ranks 1-2) in force ~6 days before kickoff, plus the position map for its format."""
    if depth.fmt.iloc[0] == "espn":
        cutoff = gameday - pd.Timedelta(days=6)
        d = depth[(depth.team == team) & (depth.dt <= cutoff + pd.Timedelta(hours=23))]
        if len(d) == 0:
            d = depth[(depth.team == team)]
            d = d[d.dt == d.dt.min()]
        else:
            d = d[d.dt == d.dt.max()]
        return d, ESPN_POS
    d = depth[(depth.team == team) & (depth.week == week)]
    if len(d) == 0:
        d = depth[(depth.team == team) & (depth.week < week)]
        if len(d):
            d = d[d.week == d.week.max()]
    return d, NFL_POS


def backup_qb(depth: pd.DataFrame, team: str, week: int, gameday: pd.Timestamp):
    """gsis_id and name of the rank-2 quarterback on the chart in force for the game."""
    d, _ = chart_for_game(depth, team, week, gameday)
    q = d[(d.pos == "QB") & (d["rank"] == 2)]
    if len(q) == 0:
        return None, None
    return q.gsis_id.iloc[0], q.player_name.iloc[0]


def starters_for_game(depth: pd.DataFrame, team: str, week: int, gameday: pd.Timestamp) -> pd.DataFrame:
    """Rank-1 non-QB starters from the chart in force ~6 days before kickoff (week-of chart for NFL format)."""
    d, posmap = chart_for_game(depth, team, week, gameday)
    d = d[d["rank"] == 1].copy()
    d["grp"] = d.pos.map(lambda p: posmap.get(p, (None, 0.0))[0])
    d["wt"] = d.pos.map(lambda p: posmap.get(p, (None, 0.0))[1])
    d = d[d.wt > 0]
    # a player listed in several formations counts once at his highest weight
    d = d.sort_values("wt", ascending=False).drop_duplicates("gsis_id")
    return d[["gsis_id", "player_name", "grp", "wt"]]


def injury_index(depth: pd.DataFrame, inj: pd.DataFrame, team: str, season: int, week: int,
                 gameday: pd.Timestamp) -> tuple[float, list]:
    st = starters_for_game(depth, team, week, gameday)
    rep = inj[(inj.team == team) & (inj.week == week) & (inj.position != "QB")].copy()
    rep["p_out"] = [p_out(a, b) for a, b in zip(rep.report_status, rep.practice_status)]
    rep = rep[rep.p_out > 0].drop_duplicates("gsis_id")
    m = st.merge(rep[["gsis_id", "report_status", "practice_status", "p_out"]], on="gsis_id", how="inner")
    if len(m) == 0:
        return 0.0, []
    m["report_status"] = m.report_status.fillna(m.practice_status.str.replace(" Participation in Practice", "").str.replace("Did Not Participate In Practice", "DNP"))
    m["pts"] = m.wt * m.p_out
    # diminishing returns inside a position group (3rd WR out hurts less than the 1st)
    m = m.sort_values("pts", ascending=False)
    m["rank_in_grp"] = m.groupby("grp").cumcount()
    m["pts"] = m.pts * (0.65 ** m.rank_in_grp)
    detail = [f"{r.player_name} ({r.grp}, {r.report_status})" for r in m.itertuples()]
    return float(m.pts.sum()), detail
