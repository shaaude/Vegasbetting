"""2026 season: train on Weeks 1-3, predict Weeks 4-5 against DraftKings opening lines (and closing lines),
grade the results, run refinement diagnostics, then publish the Week 6 board.

Usage:  python3 run_2026.py [--week N]   (default produces Weeks 4-5 evaluation + Week 6 board)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from model import SeasonCache, Params, predict_week, grade, summarize, ratings_asof, add_probabilities
from data import load_opening_lines, load_games
from ratings import TEAMS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


def load_params() -> Params:
    path = os.path.join(HERE, os.environ.get("NFL_PARAMS", "params_tuned_rolling.json"))
    if os.path.exists(path):
        return Params(**json.load(open(path)))
    return Params()


def eval_table(d: pd.DataFrame, open_lines: pd.Series) -> pd.DataFrame:
    """Add opening-line fields and ATS grades against both open and close."""
    d = d.copy()
    d["open_spread"] = d.game_id.map(open_lines)
    d["close_spread"] = d.market_spread
    d["edge_vs_open"] = d.model_spread - d.open_spread
    d["edge_vs_close"] = d.model_spread - d.close_spread
    d["line_move"] = d.close_spread - d.open_spread
    d["pick_open"] = np.where(d.edge_vs_open > 0, d.home, d.away)
    d["pick_close"] = np.where(d.edge_vs_close > 0, d.home, d.away)
    d["home_cover_open"] = np.sign(d.result - d.open_spread)
    d["home_cover_close"] = np.sign(d.result - d.close_spread)
    d["ats_open"] = np.where(d.pick_open == d.home, d.home_cover_open, -d.home_cover_open)
    d["ats_close"] = np.where(d.pick_close == d.home, d.home_cover_close, -d.home_cover_close)
    d["err_model"] = (d.model_spread - d.result).abs()
    d["err_open"] = (d.open_spread - d.result).abs()
    d["err_close"] = (d.close_spread - d.result).abs()
    d["err_blend"] = (d.blend_spread - d.result).abs()
    d["su_model"] = np.sign(d.model_spread) == np.sign(d.result)
    d["su_close"] = np.sign(d.close_spread) == np.sign(d.result)
    return d


def ats(x: pd.Series) -> str:
    w, l, p = int((x == 1).sum()), int((x == -1).sum()), int((x == 0).sum())
    return f"{w}-{l}" + (f"-{p}" if p else "") + f" ({w / max(w + l, 1):.1%})"


def main():
    p = load_params()
    p = replace(p, use_wind=False)   # no forecast feed; wind off so the board matches what you can know pre-game
    cache = SeasonCache(2026)
    ol = load_opening_lines().set_index("game_id")
    open_lines = ol.open_spread

    # ---- Weeks 4-5: walk-forward, opening line as the market reference for edge/picks ----
    frames = []
    for w in (4, 5):
        d = predict_week(cache, w, p, line_override=open_lines)   # edge/picks measured vs the OPENER
        d["market_spread"] = cache.static(w).market_spread          # restore the close for grading
        frames.append(d)
    ev = eval_table(pd.concat(frames, ignore_index=True), open_lines)
    ev = add_probabilities(ev.assign(market_spread=ev.open_spread), p).assign(market_spread=ev.close_spread)
    ev["open_note"] = ev.game_id.map(ol.open_note).fillna("")
    cols = ["week", "away", "home", "away_qb", "home_qb", "rating_away", "rating_home", "hfa", "qb_adj_home", "qb_adj_away",
            "inj_adj_home", "inj_adj_away", "rest_adj", "travel_adj", "model_spread", "blend_spread", "open_spread",
            "close_spread", "line_move", "result", "edge_vs_open", "edge_vs_close", "pick_open", "ats_open", "pick_close",
            "ats_close", "p_home_win_model", "pick_cover_prob", "edge_prob", "kelly_stake", "recommend", "tier",
            "err_model", "err_blend", "err_open", "err_close", "inj_detail_home", "inj_detail_away", "open_note"]
    ev[cols].round(3).to_csv(os.path.join(OUT, "predictions_weeks4_5.csv"), index=False)

    rep = []
    rep.append("## Weeks 4-5 scorecard (31 games, walk-forward)\n")
    rep.append(f"- Mean absolute error vs final margin: model {ev.err_model.mean():.2f} | blend {ev.err_blend.mean():.2f} | "
               f"opening line {ev.err_open.mean():.2f} | closing line {ev.err_close.mean():.2f}")
    rep.append(f"- Straight-up winners: model {ev.su_model.mean():.1%} | closing line {ev.su_close.mean():.1%}")
    rep.append(f"- ATS, every game, pick = model side vs opener: {ats(ev.ats_open)}; vs close: {ats(ev.ats_close)}")
    rec = ev[ev.recommend]
    rep.append(f"- ATS, PLAY tier only (|edge vs open| >= {p.play_pts:.0f} pts, n={len(rec)}): {ats(rec.ats_open)}; "
               f"same tickets graded at the close: {ats(rec.ats_close)}")
    lean = ev[ev.tier.isin(["PLAY", "LEAN"])]
    rep.append(f"- ATS, PLAY + LEAN tiers (|edge vs open| >= {p.lean_pts} pts, n={len(lean)}): {ats(lean.ats_open)}; at the close: {ats(lean.ats_close)}")
    for lo, hi in [(0, 1.5), (1.5, 3), (3, 99)]:
        b = ev[(ev.edge_vs_open.abs() >= lo) & (ev.edge_vs_open.abs() < hi)]
        rep.append(f"  - |edge vs open| {lo}-{hi} pts: n={len(b)}, ATS {ats(b.ats_open)}")
    # line-move agreement: did the market move toward the model?
    mv = ev[ev.line_move.abs() >= 0.5]
    agree = (np.sign(mv.line_move) == np.sign(mv.edge_vs_open)).mean() if len(mv) else np.nan
    rep.append(f"- Of {len(mv)} games whose line moved >= 0.5, the market moved toward the model side {agree:.0%} of the time")
    brier_m = ((ev.p_home_win_model - (ev.result > 0)) ** 2).mean()
    brier_mkt = ((norm.cdf(ev.close_spread / p.sigma) - (ev.result > 0)) ** 2).mean()
    rep.append(f"- Win-probability Brier: model {brier_m:.3f} | closing line {brier_mkt:.3f}")
    for w, dd in ev.groupby("week"):
        rep.append(f"- Week {w}: MAE model {dd.err_model.mean():.2f} vs open {dd.err_open.mean():.2f} vs close {dd.err_close.mean():.2f}; "
                   f"all-game ATS vs open {ats(dd.ats_open)}; PLAY+LEAN {ats(dd[dd.tier != 'PASS'].ats_open)}")

    # ---- refinement diagnostics on the 31 games ----
    rep.append("\n## Refinement diagnostics\n")
    X = np.column_stack([ev.model_spread, ev.close_spread, np.ones(len(ev))])
    coef, *_ = np.linalg.lstsq(X, ev.result, rcond=None)
    rep.append(f"- Regression result ~ a*model + b*close: a={coef[0]:.2f}, b={coef[1]:.2f} (intercept {coef[2]:.2f}); "
               f"tuned blend weight on market = {p.alpha_market:.2f}")
    resid = ev.result - ev.model_spread
    rep.append(f"- Model residual (result - model): mean {resid.mean():+.2f}, home favorites {resid[ev.model_spread > 0].mean():+.2f}, "
               f"home dogs {resid[ev.model_spread < 0].mean():+.2f}")
    qbg = ev[(ev.qb_adj_home.abs() > 1) | (ev.qb_adj_away.abs() > 1)]
    rep.append(f"- Games with a material QB adjustment (n={len(qbg)}): model MAE {qbg.err_model.mean():.2f} vs close {qbg.err_close.mean():.2f}, ATS vs open {ats(qbg.ats_open)}")
    big = ev[ev.edge_vs_open.abs() >= 3]
    rep.append(f"- Large-edge games (>=3 pts vs open, n={len(big)}): ATS {ats(big.ats_open)}; these are where a structural miss would show")

    open(os.path.join(OUT, "scorecard_weeks4_5.md"), "w").write("\n".join(rep))
    print("\n".join(rep))

    # ---- Week 6 board (ratings through Week 5) ----
    post6 = ratings_asof(cache, 6, p)
    qbv = cache.qb_values_asof(6)
    tbl = post6[["rating", "pass_pts", "rush_pts", "off_pass", "def_pass", "off_rush", "def_rush", "market_prior", "games"]].copy()
    tbl = tbl.sort_values("rating", ascending=False).round(3)
    tbl.index.name = "team"
    tbl.to_csv(os.path.join(OUT, "power_ratings_through_week5.csv"))
    w6 = predict_week(cache, 6, p)                       # tiers / edge / Kelly against the CURRENT line (what is bettable now)
    pure = Params(**json.load(open(os.path.join(HERE, "params_tuned.json"))))   # Weeks 1-3 market prior + heavier EPA weight
    w6["model_spread_pure"] = predict_week(cache, 6, replace(pure, use_wind=False)).model_spread.values
    w6["open_spread"] = w6.game_id.map(open_lines)
    w6["current_spread"] = cache.static(6).market_spread
    w6["edge_vs_open"] = w6.model_spread - w6.open_spread
    w6["edge_vs_current"] = w6.model_spread - w6.current_spread
    w6cols = ["gameday", "away", "home", "away_qb", "home_qb", "rating_away", "rating_home", "hfa", "qb_adj_home", "qb_adj_away",
              "inj_adj_home", "inj_adj_away", "rest_adj", "travel_adj", "model_spread", "blend_spread", "open_spread",
              "current_spread", "edge_vs_open", "edge_vs_current", "p_home_win_model", "pick", "pick_cover_prob",
              "edge_prob", "kelly_stake", "recommend", "tier", "qb_risk", "qb_status_home", "qb_status_away", "backup_qb_home", "backup_qb_away",
              "model_spread_if_backup", "model_spread_pure", "result", "inj_detail_home", "inj_detail_away"]
    w6[w6cols].round(3).to_csv(os.path.join(OUT, "week6_board.csv"), index=False)
    json.dump(asdict(p), open(os.path.join(OUT, "params_used.json"), "w"), indent=2)
    print(tbl.head(10))
    print(w6[["away", "home", "model_spread", "open_spread", "current_spread", "edge_vs_open", "pick", "edge_prob", "recommend", "result"]].round(2).to_string())


if __name__ == "__main__":
    main()
