"""Spread model: assembles ratings + adjustments into a predicted margin, win/cover probabilities, edge and Kelly stake.

Pipeline for a given (season, week):
  1. static game features that do not depend on tuneable weights (home field, rest, travel, wind,
     QB starter-change delta in EPA/dropback, injury-index delta)                -> SeasonCache.static(week)
  2. posterior opponent-adjusted EPA ratings from games before `week`, with a prior that blends
     last season's EPA ratings (QB-adjusted) and market-implied ratings from Weeks 1-3 lines
                                                                                   -> ratings_asof(...)
  3. assemble: margin = HFA + rating gap + QB adj + injury adj + rest + travel     -> assemble(...)
  4. probabilities, edge vs market, fractional Kelly                               -> add_probabilities(...)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import load_games, load_pbp, load_injuries, load_depth, filter_plays, unit_game_stats
from ratings import adjusted_unit_ratings, market_implied_ratings, season_unit_ratings, TEAMS
from qb import qb_dropbacks, qb_values, qb_value, team_sample_mix, DB_PER_GAME, REPL
from injuries import injury_index, backup_qb, p_out
from context import team_hfa, rest_adjustment, travel_adjustment, wind_pass_factor

PLAYS = 54.0  # offensive snaps per team-game after the garbage-time filter (2025 average 53.6)
TEAM_HFA = team_hfa()


@dataclass
class Params:
    k_pass: float = 400.0      # prior strength for passing units, in dropbacks (~12 games)
    k_rush: float = 500.0      # prior strength for rushing units, in rushes (~23 games)
    w_pass: float = 0.70       # weight of passing vs rushing in the net rating
    beta: float = 0.85         # EPA/play -> points scale (regression of true strength on sample)
    rho_prev: float = 0.40     # prior mix: previous-season EPA (rho) vs market-implied (1-rho)
    to_weight: float = 0.75    # turnover EPA kept (1 = full EPA, 0 = treat turnovers as ordinary failed plays)
    qb_scale: float = 0.80     # scale on QB starter-change adjustment
    inj_scale: float = 0.50    # scale on non-QB injury index
    rest_per_day: float = 0.12
    travel_per_zone: float = 0.25
    hfa: float = 2.0           # league home-field advantage, points
    use_team_hfa: bool = True
    use_wind: bool = True
    alpha_market: float = 0.60 # weight on the market line in the blended (probability-grade) margin
    prior_mode: str = "early"  # "early": prior from Weeks 1-3 lines only; "rolling": recency-weighted lines through week-1
    prior_decay: float = 0.85  # per-week decay of older lines in the rolling market prior
    sigma: float = 12.8        # SD of margin around the expected margin
    edge_threshold: float = 0.03  # minimum blended cover-probability edge for a stake to be sized
    play_pts: float = 4.0      # |model - market| in points for a PLAY
    lean_pts: float = 2.5      # |model - market| in points for a LEAN
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.03


class SeasonCache:
    """Everything about a season that does not depend on tuneable parameters (except to_weight)."""

    def __init__(self, season: int, prev_k=(400.0, 500.0)):
        self.season = season
        g = load_games()
        self.games = g[g.season == season].copy()
        self.plays = filter_plays(load_pbp(season))
        self.units = {}
        self.db = qb_dropbacks(self.plays)
        prev = filter_plays(load_pbp(season - 1))
        self.prev_db = qb_dropbacks(prev)
        self.prev_units = unit_game_stats(prev, to_weight=1.0)
        self.prev_ratings = season_unit_ratings(self.prev_units, *prev_k)
        self.db_by_season = {}
        for s in range(season - 3, season + 1):
            try:
                self.db_by_season[s] = qb_dropbacks(filter_plays(load_pbp(s)))
            except FileNotFoundError:
                pass
        self._qbv = {}
        self._inj = None
        self._static = {}
        self._market_prior = {}

    # ---- parameter-free pieces -------------------------------------------------
    def units_for(self, to_weight: float) -> pd.DataFrame:
        key = round(to_weight, 3)
        if key not in self.units:
            self.units[key] = unit_game_stats(self.plays, to_weight=to_weight)
        return self.units[key]

    def qb_values_asof(self, week: int) -> pd.DataFrame:
        if week not in self._qbv:
            self._qbv[week] = qb_values(self.db_by_season, self.season, week)
        return self._qbv[week]

    def market_prior(self, hfa: float, week: int | None = None, decay: float = 0.85) -> pd.Series:
        """Market-implied power ratings: from Weeks 1-3 lines (week=None) or recency-weighted lines before `week`."""
        key = (round(hfa, 2), week, round(decay, 3))
        if key not in self._market_prior:
            g = self.games[self.games.spread_line.notna()]
            if week is None:
                sub = g[g.week <= 3]
                self._market_prior[key] = market_implied_ratings(sub, hfa)
            else:
                sub = g[g.week < week]
                wts = decay ** (week - 1 - sub.week.to_numpy())
                self._market_prior[key] = market_implied_ratings(sub, hfa, ridge=1.0, weights=wts)
        return self._market_prior[key]

    def prev_ratings_qb_adjusted(self) -> pd.DataFrame:
        """Last season's unit ratings with passing offence shifted for an offseason QB change."""
        if not hasattr(self, "_prev_adj"):
            prev = self.prev_ratings.copy()
            qbv = self.qb_values_asof(1)
            prev_mix = team_sample_mix(self.prev_db, qbv)
            wk1 = self.games[self.games.week == 1]
            starters = {}
            for r in wk1.itertuples():
                starters[r.home_team] = r.home_qb_id
                starters[r.away_team] = r.away_qb_id
            prev["qb_shift"] = 0.0
            for t in TEAMS:
                if t in starters and t in prev_mix.index:
                    prev.loc[t, "qb_shift"] = qb_value(qbv, starters[t]) - prev_mix[t]
            prev["off_pass"] += prev.qb_shift
            self._prev_adj = prev
        return self._prev_adj

    def injury_table(self) -> pd.DataFrame:
        """Injury index for both teams in every game of the season (where a depth chart / report exists)."""
        if self._inj is None:
            inj = load_injuries(self.season)
            depth = load_depth(self.season)
            rows = []
            for r in self.games.itertuples():
                for side, team in (("home", r.home_team), ("away", r.away_team)):
                    idx, detail = injury_index(depth, inj, team, self.season, r.week, r.gameday)
                    rows.append({"game_id": r.game_id, "week": r.week, "team": team, "side": side,
                                 "inj_index": idx, "inj_detail": "; ".join(detail)})
            self._inj = pd.DataFrame(rows)
        return self._inj

    def static(self, week: int) -> pd.DataFrame:
        """Game-level features for `week` that do not depend on tuneable weights."""
        if week in self._static:
            return self._static[week]
        g = self.games[self.games.week == week].copy()
        qbv = self.qb_values_asof(week)
        db_sample = self.db[self.db.week < week]
        mix = team_sample_mix(db_sample, qbv) if len(db_sample) else pd.Series(dtype=float)
        inj = self.injury_table()
        inj_sample = inj[inj.week < week].groupby("team").inj_index.mean()
        inj_this = inj[inj.week == week].set_index(["game_id", "team"])
        rep = load_injuries(self.season)
        rep = rep[rep.week == week].drop_duplicates(["team", "gsis_id"]).set_index(["team", "gsis_id"])
        depth = load_depth(self.season)
        rows = []
        for r in g.itertuples():
            row = dict(game_id=r.game_id, season=r.season, week=r.week, gameday=r.gameday, away=r.away_team,
                       home=r.home_team, neutral=int(r.neutral), home_qb=r.home_qb_name, away_qb=r.away_qb_name,
                       home_rest=r.home_rest, away_rest=r.away_rest, roof=r.roof, wind=getattr(r, "wind", np.nan),
                       market_spread=r.spread_line, total_line=r.total_line, result=r.result,
                       team_hfa=TEAM_HFA.get(r.home_team, 0.0),
                       travel_zones=travel_adjustment(r.home_team, r.away_team, r.neutral, 1.0),
                       rest_diff=0.0 if pd.isna(r.home_rest) or pd.isna(r.away_rest) else float(r.home_rest - r.away_rest))
            for side, team, qid in (("home", r.home_team, r.home_qb_id), ("away", r.away_team, r.away_qb_id)):
                base = mix.get(team, np.nan)
                if pd.isna(base) or qid is None or (isinstance(qid, float) and np.isnan(qid)):
                    row[f"qb_delta_{side}"] = 0.0
                    row[f"qb_value_{side}"] = np.nan
                else:
                    row[f"qb_delta_{side}"] = qb_value(qbv, qid) - base
                    row[f"qb_value_{side}"] = qb_value(qbv, qid)
                # is the listed starter on this week's report? what would the backup be worth?
                st = rep.loc[(team, qid)] if (team, qid) in rep.index else None
                row[f"qb_status_{side}"] = "" if st is None else (st.report_status if isinstance(st.report_status, str) else
                                                               (st.practice_status or "").replace(" Participation in Practice", "").replace("Did Not Participate In Practice", "DNP"))
                row[f"qb_p_out_{side}"] = 0.0 if st is None else p_out(st.report_status, st.practice_status)
                bid, bname = backup_qb(depth, team, r.week, r.gameday)
                row[f"backup_qb_{side}"] = bname or ""
                row[f"backup_delta_{side}"] = (qb_value(qbv, bid) - base) if (bid is not None and not pd.isna(base)) else row[f"qb_delta_{side}"]
                key = (r.game_id, team)
                cur = inj_this.inj_index.get(key, np.nan) if len(inj_this) else np.nan
                row[f"inj_delta_{side}"] = (0.0 if pd.isna(cur) else cur) - inj_sample.get(team, 0.0)
                row[f"inj_detail_{side}"] = inj_this.inj_detail.get(key, "") if len(inj_this) else ""
            rows.append(row)
        self._static[week] = pd.DataFrame(rows)
        return self._static[week]


def ratings_asof(cache: SeasonCache, week: int, p: Params) -> pd.DataFrame:
    units = cache.units_for(p.to_weight)
    sample = units[units.week < week]
    M = cache.market_prior(p.hfa) if p.prior_mode == "early" else cache.market_prior(p.hfa, max(week, 4), p.prior_decay)
    m_epa = M / (p.beta * PLAYS)                  # net EPA/play equivalent
    prev = cache.prev_ratings_qb_adjusted()
    prior = pd.DataFrame(index=TEAMS)
    prior["off_pass"] = p.rho_prev * prev.off_pass + (1 - p.rho_prev) * m_epa / 2
    prior["off_rush"] = p.rho_prev * prev.off_rush + (1 - p.rho_prev) * m_epa / 2
    prior["def_pass"] = p.rho_prev * prev.def_pass - (1 - p.rho_prev) * m_epa / 2
    prior["def_rush"] = p.rho_prev * prev.def_rush - (1 - p.rho_prev) * m_epa / 2
    if len(sample) == 0:
        post = prior.copy()
    else:
        op, dp = adjusted_unit_ratings(sample, "db", prior.off_pass, prior.def_pass, p.k_pass)
        orr, dr = adjusted_unit_ratings(sample, "rush", prior.off_rush, prior.def_rush, p.k_rush)
        post = pd.DataFrame({"off_pass": op, "def_pass": dp, "off_rush": orr, "def_rush": dr})
    post["net_pass"] = post.off_pass - post.def_pass
    post["net_rush"] = post.off_rush - post.def_rush
    post["pass_pts"] = p.beta * PLAYS * p.w_pass * post.net_pass
    post["rush_pts"] = p.beta * PLAYS * (1 - p.w_pass) * post.net_rush
    post["rating"] = post.pass_pts + post.rush_pts
    post["market_prior"] = M
    post["games"] = sample.groupby("posteam").size().reindex(TEAMS).fillna(0).astype(int)
    return post


def assemble(static: pd.DataFrame, post: pd.DataFrame, p: Params, line_override: pd.Series | None = None) -> pd.DataFrame:
    d = static.copy()
    d["rating_home"] = d.home.map(post.rating)
    d["rating_away"] = d.away.map(post.rating)
    d["hfa"] = np.where(d.neutral == 1, 0.0, p.hfa + (d.team_hfa if p.use_team_hfa else 0.0))
    d["qb_adj_home"] = p.qb_scale * p.beta * DB_PER_GAME * d.qb_delta_home
    d["qb_adj_away"] = p.qb_scale * p.beta * DB_PER_GAME * d.qb_delta_away
    d["inj_adj_home"] = -p.inj_scale * d.inj_delta_home
    d["inj_adj_away"] = -p.inj_scale * d.inj_delta_away
    d["rest_adj"] = (d.rest_diff * p.rest_per_day).clip(-1.5, 1.5)
    d["travel_adj"] = d.travel_zones * p.travel_per_zone
    if p.use_wind:
        d["wind_factor"] = [wind_pass_factor(w, r) for w, r in zip(d.wind, d.roof)]
    else:
        d["wind_factor"] = 1.0
    pass_gap = d.home.map(post.pass_pts) - d.away.map(post.pass_pts)
    d["rating_gap"] = (d.rating_home - d.rating_away) - (1 - d.wind_factor) * pass_gap
    d["model_spread"] = (d.hfa + d.rating_gap + (d.qb_adj_home - d.qb_adj_away)
                         + (d.inj_adj_home - d.inj_adj_away) + d.rest_adj + d.travel_adj)
    # alternate margin if a starter who is Out/Doubtful/DNP is replaced by the chart's #2 QB
    alt_h = p.qb_scale * p.beta * DB_PER_GAME * d.backup_delta_home
    alt_a = p.qb_scale * p.beta * DB_PER_GAME * d.backup_delta_away
    risk_h = d.qb_p_out_home >= 0.5
    risk_a = d.qb_p_out_away >= 0.5
    d["qb_risk"] = np.select([risk_h & risk_a, risk_h, risk_a], ["BOTH", "HOME", "AWAY"], "")
    d["model_spread_if_backup"] = d.model_spread + np.where(risk_h, alt_h - d.qb_adj_home, 0.0) - np.where(risk_a, alt_a - d.qb_adj_away, 0.0)
    if line_override is not None:
        d["market_spread"] = d.game_id.map(line_override).fillna(d.market_spread)
    d["blend_spread"] = np.where(d.market_spread.isna(), d.model_spread,
                                 p.alpha_market * d.market_spread + (1 - p.alpha_market) * d.model_spread)
    return add_probabilities(d, p)


def predict_week(cache: SeasonCache, week: int, p: Params, line_override: pd.Series | None = None) -> pd.DataFrame:
    return assemble(cache.static(week), ratings_asof(cache, week, p), p, line_override)


def add_probabilities(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    s = p.sigma
    df["p_home_win_model"] = norm.cdf(df.model_spread / s)
    df["p_home_win_blend"] = norm.cdf(df.blend_spread / s)
    df["edge_pts"] = df.model_spread - df.market_spread
    df["p_home_cover_model"] = norm.cdf((df.model_spread - df.market_spread) / s)
    df["p_home_cover_blend"] = norm.cdf((df.blend_spread - df.market_spread) / s)
    df["pick"] = np.where(df.edge_pts > 0, df.home, df.away)
    df["pick_cover_prob"] = np.where(df.edge_pts > 0, df.p_home_cover_blend, 1 - df.p_home_cover_blend)
    df["edge_prob"] = df.pick_cover_prob - 110 / 210
    df["kelly_stake"] = [kelly(e + 110 / 210, p.kelly_fraction, p.kelly_cap) if e >= p.edge_threshold else 0.0 for e in df.edge_prob]
    # tiers on the raw model-vs-market gap; the probability/Kelly columns use the market-blended margin (conservative)
    df["recommend"] = df.edge_pts.abs() >= p.play_pts
    df["tier"] = np.where(df.recommend, "PLAY", np.where(df.edge_pts.abs() >= p.lean_pts, "LEAN", "PASS"))
    return df


def kelly(p_win: float, fraction: float = 0.25, cap: float = 0.03, odds: float = -110) -> float:
    b = 100 / abs(odds) if odds < 0 else odds / 100
    f = (b * p_win - (1 - p_win)) / b
    return float(np.clip(fraction * max(f, 0.0), 0.0, cap))


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ATS outcomes for completed games."""
    d = df[df.result.notna() & df.market_spread.notna()].copy()
    d["home_cover"] = np.sign(d.result - d.market_spread)   # +1 home covers, -1 away, 0 push
    d["pick_is_home"] = d.pick == d.home
    d["pick_cover"] = np.where(d.pick_is_home, d.home_cover, -d.home_cover)
    d["model_err"] = (d.model_spread - d.result).abs()
    d["blend_err"] = (d.blend_spread - d.result).abs()
    d["market_err"] = (d.market_spread - d.result).abs()
    d["home_won"] = (d.result > 0).astype(float)
    return d


def summarize(d: pd.DataFrame) -> dict:
    rec = d[d.recommend]
    w, l, push = int((rec.pick_cover == 1).sum()), int((rec.pick_cover == -1).sum()), int((rec.pick_cover == 0).sum())
    allw, alll = int((d.pick_cover == 1).sum()), int((d.pick_cover == -1).sum())
    return dict(games=len(d), mae_model=round(d.model_err.mean(), 3), mae_blend=round(d.blend_err.mean(), 3),
                mae_market=round(d.market_err.mean(), 3),
                ats_all=f"{allw}-{alll}", ats_all_pct=round(allw / max(allw + alll, 1), 3),
                ats_rec=f"{w}-{l}-{push}", ats_rec_pct=round(w / max(w + l, 1), 3), n_rec=len(rec),
                brier_model=round(((d.p_home_win_model - d.home_won) ** 2).mean(), 4),
                brier_blend=round(((d.p_home_win_blend - d.home_won) ** 2).mean(), 4),
                corr_model=round(np.corrcoef(d.model_spread, d.result)[0, 1], 3) if len(d) > 2 else np.nan,
                corr_market=round(np.corrcoef(d.market_spread, d.result)[0, 1], 3) if len(d) > 2 else np.nan)
