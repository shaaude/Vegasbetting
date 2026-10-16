"""Walk-forward backtest and staged hyper-parameter tuning on prior seasons.

Every week w of a season is predicted with ratings built from weeks < w (prior = last season + that
season's Weeks 1-3 market lines), the official injury report / named starters for week w, and the
closing line as the market reference. Hyper-parameters are tuned on 2024-2025 so the 2026 Weeks 4-5
evaluation stays out of sample.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from model import SeasonCache, Params, predict_week, grade, summarize

WEEKS = range(4, 19)


def run_season(cache: SeasonCache, p: Params, weeks=WEEKS) -> pd.DataFrame:
    return pd.concat([predict_week(cache, w, p) for w in weeks], ignore_index=True)


def evaluate(caches: list[SeasonCache], p: Params, weeks=WEEKS) -> tuple[dict, pd.DataFrame]:
    d = grade(pd.concat([run_season(c, p, weeks) for c in caches], ignore_index=True))
    return summarize(d), d


def stage(caches, base: Params, grid: dict, metric="mae_model"):
    keys = list(grid)
    best, best_p, rows = None, base, []
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = replace(base, **dict(zip(keys, combo)))
        s, _ = evaluate(caches, p)
        rows.append({**dict(zip(keys, combo)), **s})
        if best is None or s[metric] < best[metric]:
            best, best_p = s, p
    return best_p, best, pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    caches = [SeasonCache(2024), SeasonCache(2025)]
    for c in caches:  # warm the parameter-free caches
        c.injury_table()
        for w in WEEKS:
            c.static(w)
    print(f"caches ready in {time.time()-t0:.0f}s", flush=True)

    p = Params(use_wind=False)   # wind is not known at line time; tune without it
    base_summary, _ = evaluate(caches, p)
    print("baseline", base_summary, flush=True)

    # Stage 1: core engine
    p, s, r1 = stage(caches, p, dict(k_pass=[250, 400, 600, 900], k_rush=[300, 500, 900],
                                     w_pass=[0.6, 0.7, 0.8], beta=[0.7, 0.85, 1.0, 1.15]))
    print("stage1", s, {k: getattr(p, k) for k in ["k_pass", "k_rush", "w_pass", "beta"]}, flush=True)
    r1.to_csv("tuning_stage1.csv", index=False)
    # Stage 2: prior mix and turnover damping
    p, s, r2 = stage(caches, p, dict(rho_prev=[0.0, 0.25, 0.5, 0.75, 1.0], to_weight=[0.4, 0.7, 1.0]))
    print("stage2", s, {k: getattr(p, k) for k in ["rho_prev", "to_weight"]}, flush=True)
    r2.to_csv("tuning_stage2.csv", index=False)
    # Stage 3: situational adjustments
    p, s, r3 = stage(caches, p, dict(qb_scale=[0.0, 0.5, 1.0, 1.5], inj_scale=[0.0, 0.5, 1.0, 1.5],
                                     hfa=[1.5, 2.0, 2.5], rest_per_day=[0.0, 0.12, 0.25]))
    print("stage3", s, {k: getattr(p, k) for k in ["qb_scale", "inj_scale", "hfa", "rest_per_day"]}, flush=True)
    r3.to_csv("tuning_stage3.csv", index=False)
    # Stage 4: re-check core scale after adjustments, then market blend for probability grade
    p, s, r4 = stage(caches, p, dict(beta=[p.beta - 0.15, p.beta, p.beta + 0.15], k_pass=[p.k_pass * 0.7, p.k_pass, p.k_pass * 1.4]))
    print("stage4", s, {k: getattr(p, k) for k in ["beta", "k_pass"]}, flush=True)
    p, s, r5 = stage(caches, p, dict(alpha_market=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), metric="mae_blend")
    print("stage5", s, {"alpha_market": p.alpha_market}, flush=True)
    r5.to_csv("tuning_stage5.csv", index=False)

    final, d = evaluate(caches, p)
    d.to_csv("backtest_2024_2025.csv", index=False)
    json.dump(asdict(p), open("params_tuned.json", "w"), indent=2)
    print("FINAL", final)
    # per-season breakdown
    for season, dd in d.groupby("season"):
        print(season, summarize(dd))
    # edge buckets
    d["abs_edge"] = d.edge_pts.abs()
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)]:
        b = d[(d.abs_edge >= lo) & (d.abs_edge < hi)]
        w, l = (b.pick_cover == 1).sum(), (b.pick_cover == -1).sum()
        print(f"edge {lo}-{hi}: n={len(b)} ATS {w}-{l} ({w/max(w+l,1):.3f})")
    print(f"done in {time.time()-t0:.0f}s")
