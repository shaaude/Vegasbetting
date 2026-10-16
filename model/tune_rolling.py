"""Second tuning pass: rolling (recency-weighted) market prior instead of Weeks 1-3 only."""
import json, time
from dataclasses import asdict, replace
import pandas as pd
from model import SeasonCache, Params
from backtest import evaluate, stage, WEEKS

t0 = time.time()
caches = [SeasonCache(2024), SeasonCache(2025)]
for c in caches:
    c.injury_table()
    for w in WEEKS:
        c.static(w)
base = Params(**json.load(open("params_tuned.json")))
p = replace(base, prior_mode="rolling", use_wind=False)
print("rolling baseline", evaluate(caches, p)[0], flush=True)
p, s, r1 = stage(caches, p, dict(prior_decay=[0.6, 0.8, 0.95], k_pass=[250, 400, 700, 1200], beta=[0.85, 1.0, 1.15], w_pass=[0.6, 0.7]))
print("r-stage1", s, {k: getattr(p, k) for k in ["prior_decay", "k_pass", "beta", "w_pass"]}, flush=True)
r1.to_csv("tuning_rolling1.csv", index=False)
p, s, r2 = stage(caches, p, dict(k_rush=[300, 600, 1200], qb_scale=[0.5, 1.0, 1.5], inj_scale=[0.5, 1.0, 1.5], hfa=[1.5, 2.0]))
print("r-stage2", s, {k: getattr(p, k) for k in ["k_rush", "qb_scale", "inj_scale", "hfa"]}, flush=True)
r2.to_csv("tuning_rolling2.csv", index=False)
p, s, r3 = stage(caches, p, dict(alpha_market=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8]), metric="mae_blend")
print("r-stage3", s, {"alpha_market": p.alpha_market}, flush=True)
final, d = evaluate(caches, p)
d.to_csv("backtest_2024_2025_rolling.csv", index=False)
json.dump(asdict(p), open("params_tuned_rolling.json", "w"), indent=2)
print("FINAL rolling", final, flush=True)
for season, dd in d.groupby("season"):
    from model import summarize
    print(season, summarize(dd))
d["abs_edge"] = d.edge_pts.abs()
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)]:
    b = d[(d.abs_edge >= lo) & (d.abs_edge < hi)]
    w, l = (b.pick_cover == 1).sum(), (b.pick_cover == -1).sum()
    print(f"edge {lo}-{hi}: n={len(b)} ATS {w}-{l} ({w/max(w+l,1):.3f})")
print(f"done in {time.time()-t0:.0f}s")
