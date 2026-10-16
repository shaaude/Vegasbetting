"""NCAA (FBS) spread model — the NFL pipeline re-used on ESPN college play-by-play.

Same architecture: opponent-adjusted pass/rush EPA ratings (Bayesian ridge) shrunk toward a rolling
market-implied power rating; HFA; blend with the market; win/cover probabilities; tiers.
No college injury reports or QB-change feed are freely available, so those layers are off here.

Data: sportsdataverse `espn_cfb_pbp` 2026 (EPA, success, win prob and the ESPN/DraftKings spread for
every game) and cfbfastR `cfb_schedules_2026` (divisions, Elo, Saturday slate).
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
CFB = os.environ.get("CFB_DATA_DIR", "/tmp/cfb")
OUT = os.path.join(HERE, "output")
PLAYS = 65.0          # scrimmage snaps per team-game (ESPN pbp averages ~130 per game)
AP_TOP25 = ["Georgia", "Notre Dame", "Miami", "Ohio State", "Indiana", "Alabama", "Texas", "BYU", "Ole Miss", "Texas Tech",
            "Utah", "UCLA", "Missouri", "Tennessee", "Kentucky", "Mississippi State", "Iowa", "Oklahoma State", "USC",
            "Houston", "Wake Forest", "LSU", "Boise State", "SMU", "Northwestern"]   # AP poll of Oct 11, 2026
AP_RANK = {t: i + 1 for i, t in enumerate(AP_TOP25)}


@dataclass
class CParams:
    k_pass: float = 600.0
    k_rush: float = 600.0
    w_pass: float = 0.65
    beta: float = 1.0
    prior_decay: float = 0.8
    market_ridge: float = 1.0
    hfa: float = 2.5
    alpha_market: float = 0.7
    sigma: float = 14.8
    play_pts: float = 5.0
    lean_pts: float = 3.0
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.03
    edge_threshold: float = 0.03


def load():
    cols = ["game_id", "week", "homeTeamName", "awayTeamName", "homeTeamSpread", "pos_team_id", "homeTeamId", "awayTeamId", "EPA",
            "rush", "pass", "EPA_success", "period", "home_wp_before", "homeScore", "awayScore"]
    e = pd.read_parquet(os.path.join(CFB, "espn_pbp_2026.parquet"), columns=cols)
    s = pd.read_parquet(os.path.join(CFB, "schedules_2026.parquet"))
    fbs = set(s.loc[s.home_division == "fbs", "home_team"]) | set(s.loc[s.away_division == "fbs", "away_team"])
    return e, s, fbs


def bucket(name, fbs):
    return name if name in fbs else "FCS"


def games_table(e, fbs):
    g = e.groupby("game_id").agg(week=("week", "first"), home=("homeTeamName", "first"), away=("awayTeamName", "first"),
                                 spread=("homeTeamSpread", "first"), hs=("homeScore", "max"), as_=("awayScore", "max")).reset_index()
    g["margin"] = g.hs - g.as_
    g["home_b"] = g.home.map(lambda t: bucket(t, fbs))
    g["away_b"] = g.away.map(lambda t: bucket(t, fbs))
    return g


def unit_stats(e, fbs):
    x = e[(e.rush == True) | (e["pass"] == True)].copy()
    x = x[x.EPA.notna()]
    garbage = (x.period >= 4) & ((x.home_wp_before < 0.04) | (x.home_wp_before > 0.96))
    x = x[~garbage]
    x["home"] = (x.pos_team_id == x.homeTeamId).astype(int)
    x["is_db"] = (x["pass"] == True).astype(int)
    pos_name = np.where(x.home == 1, x.homeTeamName, x.awayTeamName)
    def_name = np.where(x.home == 1, x.awayTeamName, x.homeTeamName)
    x["posteam"] = [bucket(t, fbs) for t in pos_name]
    x["defteam"] = [bucket(t, fbs) for t in def_name]
    grp = x.groupby(["game_id", "week", "posteam", "defteam", "home"])
    u = grp.apply(lambda d: pd.Series({
        "db_epa": d.loc[d.is_db == 1, "EPA"].mean(), "db_n": int((d.is_db == 1).sum()),
        "rush_epa": d.loc[d.is_db == 0, "EPA"].mean(), "rush_n": int((d.is_db == 0).sum()),
        "success": d.EPA_success.mean(), "plays": len(d)}), include_groups=False).reset_index()
    return u.dropna(subset=["db_epa", "rush_epa"])


def ridge_ratings(units, unit, teams, prior_off, prior_def, k):
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    y = units[f"{unit}_epa"].to_numpy()
    w = units[f"{unit}_n"].to_numpy().astype(float)
    mu = np.average(y, weights=w)
    y = y - mu
    n = len(units)
    X = np.zeros((n, 2 * n_t))
    X[np.arange(n), units.posteam.map(idx).to_numpy()] = 1.0
    X[np.arange(n), n_t + units.defteam.map(idx).to_numpy()] = 1.0
    prior = np.concatenate([prior_off.reindex(teams).fillna(0).to_numpy(), prior_def.reindex(teams).fillna(0).to_numpy()])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + k * np.eye(2 * n_t), XtW @ y + k * prior)
    beta[:n_t] -= beta[:n_t].mean()
    beta[n_t:] -= beta[n_t:].mean()
    return pd.Series(beta[:n_t], index=teams), pd.Series(beta[n_t:], index=teams)


def elo_prior(s, teams, fbs, scale=None):
    """Preseason CFBD Elo converted to points (mean 0 over FBS); FCS bucket gets the FCS average."""
    first = pd.concat([s[["week", "home_team", "home_pregame_elo"]].rename(columns={"home_team": "t", "home_pregame_elo": "elo"}),
                       s[["week", "away_team", "away_pregame_elo"]].rename(columns={"away_team": "t", "away_pregame_elo": "elo"})])
    first = first.dropna().sort_values("week").drop_duplicates("t")
    elo = first.set_index("t").elo
    fbs_mean = elo[elo.index.isin(fbs)].mean()
    scale = scale or 27.0   # ~27 Elo points per point of spread (fit below)
    pts = (elo - fbs_mean) / scale
    out = pd.Series(0.0, index=teams)
    for t in teams:
        if t == "FCS":
            out[t] = pts[~pts.index.isin(fbs)].mean() if (~pts.index.isin(fbs)).any() else -25.0
        elif t in pts.index:
            out[t] = pts[t]
    return out


def market_ratings(g, teams, hfa, decay, week, ridge=1.0, prior=None):
    """Rolling market-implied ratings (points) from ESPN/DraftKings spreads of games before `week`,
    ridge-shrunk toward `prior` (preseason Elo in points) rather than toward zero."""
    idx = {t: i for i, t in enumerate(teams)}
    r0 = np.zeros(len(teams)) if prior is None else prior.reindex(teams).fillna(0).to_numpy()
    sub = g[g.spread.notna() & (g.week < week)]
    n = len(sub)
    X = np.zeros((n, len(teams)))
    X[np.arange(n), sub.home_b.map(idx).to_numpy()] = 1.0
    X[np.arange(n), sub.away_b.map(idx).to_numpy()] = -1.0
    y = sub.spread.to_numpy() - hfa
    w = decay ** (week - 1 - sub.week.to_numpy())
    XtW = X.T * w
    r = np.linalg.solve(XtW @ X + ridge * np.eye(len(teams)), XtW @ y + ridge * r0)
    r -= r[[idx[t] for t in teams if t != "FCS"]].mean()
    return pd.Series(r, index=teams)


def ratings_asof(units, g, teams, week, p: CParams, prior=None):
    M = market_ratings(g, teams, p.hfa, p.prior_decay, week, p.market_ridge, prior)
    m_epa = M / (p.beta * PLAYS)
    prior_off, prior_def = m_epa / 2, -m_epa / 2
    sample = units[units.week < week]
    op, dp = ridge_ratings(sample, "db", teams, prior_off, prior_def, p.k_pass)
    orr, dr = ridge_ratings(sample, "rush", teams, prior_off, prior_def, p.k_rush)
    post = pd.DataFrame({"off_pass": op, "def_pass": dp, "off_rush": orr, "def_rush": dr})
    post["net_pass"] = post.off_pass - post.def_pass
    post["net_rush"] = post.off_rush - post.def_rush
    post["pass_pts"] = p.beta * PLAYS * p.w_pass * post.net_pass
    post["rush_pts"] = p.beta * PLAYS * (1 - p.w_pass) * post.net_rush
    post["rating"] = post.pass_pts + post.rush_pts
    post["market_prior"] = M
    post["games"] = sample.groupby("posteam").size().reindex(teams).fillna(0).astype(int)
    return post


def assemble(df, post, p: CParams):
    d = df.copy()
    d["rating_home"] = d.home_b.map(post.rating)
    d["rating_away"] = d.away_b.map(post.rating)
    d["hfa"] = np.where(d.neutral == 1, 0.0, p.hfa)
    d["model_spread"] = d.hfa + d.rating_home - d.rating_away
    d["blend_spread"] = np.where(d.market_spread.isna(), d.model_spread, p.alpha_market * d.market_spread + (1 - p.alpha_market) * d.model_spread)
    s = p.sigma
    d["p_home_win_model"] = norm.cdf(d.model_spread / s)
    d["p_home_win_blend"] = norm.cdf(d.blend_spread / s)
    d["edge_pts"] = d.model_spread - d.market_spread
    d["p_home_cover_blend"] = norm.cdf((d.blend_spread - d.market_spread) / s)
    d["pick"] = np.where(d.edge_pts > 0, d.home, d.away)
    d["pick_cover_prob"] = np.where(d.edge_pts > 0, d.p_home_cover_blend, 1 - d.p_home_cover_blend)
    d["edge_prob"] = d.pick_cover_prob - 110 / 210
    d["tier"] = np.where(d.edge_pts.abs() >= p.play_pts, "PLAY", np.where(d.edge_pts.abs() >= p.lean_pts, "LEAN", "PASS"))
    d["tier"] = np.where(d.market_spread.isna(), "NO LINE", d.tier)
    return d


def backtest(units, g, teams, p: CParams, weeks=(4, 5, 6, 7), prior=None):
    rows = []
    for w in weeks:
        post = ratings_asof(units, g, teams, w, p, prior)
        gw = g[(g.week == w) & g.spread.notna() & g.margin.notna()].copy()
        gw["neutral"] = 0  # ESPN pbp does not carry the neutral flag; small effect on a handful of games
        gw["market_spread"] = gw.spread
        rows.append(assemble(gw, post, p))
    d = pd.concat(rows, ignore_index=True)
    d["err_model"] = (d.model_spread - d.margin).abs()
    d["err_blend"] = (d.blend_spread - d.margin).abs()
    d["err_market"] = (d.market_spread - d.margin).abs()
    d["home_cover"] = np.sign(d.margin - d.market_spread)
    d["pick_cover"] = np.where(d.pick == d.home, d.home_cover, -d.home_cover)
    return d


def summarize(d):
    w, l = int((d.pick_cover == 1).sum()), int((d.pick_cover == -1).sum())
    t = d[d.tier != "PASS"]
    tw, tl = int((t.pick_cover == 1).sum()), int((t.pick_cover == -1).sum())
    return dict(games=len(d), mae_model=round(d.err_model.mean(), 2), mae_blend=round(d.err_blend.mean(), 2),
                mae_market=round(d.err_market.mean(), 2), ats_all=f"{w}-{l}", ats_tiers=f"{tw}-{tl}", n_tiers=len(t))


def main():
    e, s, fbs = load()
    g = games_table(e, fbs)
    units = unit_stats(e, fbs)
    teams = sorted(set(units.posteam) | set(units.defteam) | set(g.home_b) | set(g.away_b))
    # Elo -> points scale from the lines themselves: spread ~ hfa + (elo_home - elo_away)/scale
    sg = s[s.home_pregame_elo.notna() & s.away_pregame_elo.notna()].merge(g[["home", "away", "spread"]], left_on=["home_team", "away_team"], right_on=["home", "away"])
    sg = sg[sg.spread.notna()]
    scale = float(((sg.home_pregame_elo - sg.away_pregame_elo) ** 2).sum() / ((sg.home_pregame_elo - sg.away_pregame_elo) * (sg.spread - 2.5)).sum())
    print(f"Elo scale: {scale:.1f} Elo points per spread point (n={len(sg)})")
    prior = elo_prior(s, teams, fbs, scale)
    base = CParams()
    # --- tune on a walk-forward over Weeks 4-7 (ratings from earlier weeks, ESPN/DK line as market) ---
    best, best_p, rows = None, base, []
    for k, b, dec, wp, rg in itertools.product([300, 600, 1200, 2400], [0.85, 1.0, 1.15], [0.6, 0.8, 1.0], [0.55, 0.65], [0.5, 1.0, 2.0]):
        p = replace(base, k_pass=k, k_rush=k, beta=b, prior_decay=dec, w_pass=wp, market_ridge=rg)
        sm = summarize(backtest(units, g, teams, p, prior=prior))
        rows.append({**dict(k=k, beta=b, decay=dec, w_pass=wp, ridge=rg), **sm})
        if best is None or sm["mae_model"] < best["mae_model"]:
            best, best_p = sm, p
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ncaa_tuning.csv"), index=False)
    alphas = [(summarize(backtest(units, g, teams, replace(best_p, alpha_market=a), prior=prior))["mae_blend"], a) for a in [0.5, 0.6, 0.7, 0.8, 0.9]]
    best_p = replace(best_p, alpha_market=min(alphas)[1])
    bt = backtest(units, g, teams, best_p, prior=prior)
    final = summarize(bt)
    bt.to_csv(os.path.join(OUT, "ncaa_backtest_weeks4_7.csv"), index=False)
    json.dump(asdict(best_p), open(os.path.join(OUT, "ncaa_params.json"), "w"), indent=2)
    print("NCAA tuned", asdict(best_p)); print("backtest", final)
    for lo, hi in [(0, 3), (3, 5), (5, 8), (8, 99)]:
        b = bt[(bt.edge_pts.abs() >= lo) & (bt.edge_pts.abs() < hi)]
        print(f"  |edge| {lo}-{hi}: n={len(b)} ATS {(b.pick_cover==1).sum()}-{(b.pick_cover==-1).sum()}")

    # --- Saturday slate: AP Top 25 + Big Ten ---
    post = ratings_asof(units, g, teams, 8, best_p, prior)   # all games through week 7 (midweek games included)
    tbl = post.copy(); tbl["ap_rank"] = tbl.index.map(AP_RANK)
    tbl.sort_values("rating", ascending=False).round(3).to_csv(os.path.join(OUT, "ncaa_power_ratings.csv"))
    s["start"] = pd.to_datetime(s.start_date)
    sat = s[(s.start >= "2026-10-17T00:00:00Z") & (s.start < "2026-10-18T06:00:00Z")].copy()
    keep = sat.home_team.isin(AP_TOP25) | sat.away_team.isin(AP_TOP25) | (sat.home_conference == "Big Ten") | (sat.away_conference == "Big Ten")
    slate = sat[keep].copy()
    lines = pd.read_csv(os.path.join(HERE, "ncaa_lines.csv"))
    slate = slate.merge(lines, left_on=["away_team", "home_team"], right_on=["away", "home"], how="left")
    slate["home_b"] = slate.home_team.map(lambda t: bucket(t, fbs)); slate["away_b"] = slate.away_team.map(lambda t: bucket(t, fbs))
    slate["home"] = slate.home_team; slate["away"] = slate.away_team
    slate["neutral"] = slate.neutral_site.astype(int)
    out = assemble(slate, post, best_p)
    out["kelly_stake"] = [max(0.0, min(best_p.kelly_cap, best_p.kelly_fraction * ((0.909 * (e_ + 110 / 210) - (1 - (e_ + 110 / 210))) / 0.909))) if e_ >= best_p.edge_threshold else 0.0 for e_ in out.edge_prob]
    out["ap_rank_home"] = out.home.map(AP_RANK); out["ap_rank_away"] = out.away.map(AP_RANK)
    out["elo_home"] = out.home_pregame_elo; out["elo_away"] = out.away_pregame_elo
    out["pass_pts_home"] = out.home_b.map(post.pass_pts); out["rush_pts_home"] = out.home_b.map(post.rush_pts)
    out["pass_pts_away"] = out.away_b.map(post.pass_pts); out["rush_pts_away"] = out.away_b.map(post.rush_pts)
    out["market_prior_home"] = out.home_b.map(post.market_prior); out["market_prior_away"] = out.away_b.map(post.market_prior)
    out["kickoff_utc"] = out.start_date
    out["result"] = out.home_points - out.away_points
    cols = ["kickoff_utc", "away", "home", "ap_rank_away", "ap_rank_home", "away_conference", "home_conference", "neutral",
            "rating_away", "rating_home", "pass_pts_away", "rush_pts_away", "pass_pts_home", "rush_pts_home",
            "market_prior_away", "market_prior_home", "elo_away", "elo_home", "hfa", "model_spread", "blend_spread",
            "market_spread", "total", "source", "note", "edge_pts", "p_home_win_model", "p_home_win_blend", "pick",
            "pick_cover_prob", "edge_prob", "kelly_stake", "tier", "result"]
    out = out.sort_values("kickoff_utc")
    out[cols].round(3).to_csv(os.path.join(OUT, "ncaa_saturday_board.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out[["away", "home", "rating_away", "rating_home", "model_spread", "market_spread", "edge_pts", "pick", "tier", "p_home_win_model"]].round(2).to_string())


if __name__ == "__main__":
    main()
