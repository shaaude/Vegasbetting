"""Assemble output/app_data.json: everything the Edge Board page embeds (NFL Week 6 + NCAA Saturday)."""
import json, os, math
import numpy as np, pandas as pd
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "output")
def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v)) else round(float(v), 3)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    try:
        if pd.isna(v): return None
    except Exception: pass
    return v
def rows(df): return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]

nfl_p = json.load(open(os.path.join(OUT, "params_used.json")))
w6 = pd.read_csv(os.path.join(OUT, "week6_board.csv"))
w6["gameday"] = w6.gameday.astype(str)
ev = pd.read_csv(os.path.join(OUT, "predictions_weeks4_5.csv"))
pr = pd.read_csv(os.path.join(OUT, "power_ratings_through_week5.csv"))
ncaa = pd.read_csv(os.path.join(OUT, "ncaa_saturday_board.csv"))
ncaa_p = json.load(open(os.path.join(OUT, "ncaa_params.json")))
npr = pd.read_csv(os.path.join(OUT, "ncaa_power_ratings.csv")).rename(columns={"Unnamed: 0": "team"})
nbt = pd.read_csv(os.path.join(OUT, "ncaa_backtest_weeks4_7.csv"))
scorecard = open(os.path.join(OUT, "scorecard_weeks4_5.md")).read()

kalshi = json.load(open(os.path.join(HERE, "kalshi_snapshot.json")))
kalshi_asof = kalshi.pop("as_of", "")

# ---- model totals: each team's expected points from opponent-adjusted EPA (offense vs the other defense) ----
games_all = pd.read_csv(os.path.join("/tmp/nfl", "games.csv"))
gp = games_all[(games_all.season == 2026) & (games_all.week < 6) & games_all.home_score.notna()]
L_nfl = float(pd.concat([gp.home_score, gp.away_score]).mean())
prr = pr.set_index("team")
def nfl_pts(t, o):
    b, wp = nfl_p["beta"], nfl_p["w_pass"]
    return L_nfl + b * 54.0 * (wp * (prr.at[t, "off_pass"] + prr.at[o, "def_pass"]) + (1 - wp) * (prr.at[t, "off_rush"] + prr.at[o, "def_rush"]))
w6["model_total"] = [nfl_pts(r.home, r.away) + nfl_pts(r.away, r.home) + (r.qb_adj_home or 0) + (r.qb_adj_away or 0) for r in w6.itertuples()]
w6["model_total"] = w6.model_total.round(1)
gl = games_all[(games_all.season == 2026) & (games_all.week == 6)].set_index(["away_team", "home_team"]).total_line
w6["total"] = [gl.get((r.away, r.home)) for r in w6.itertuples()]
sched = pd.read_parquet(os.path.join("/tmp/cfb", "schedules_2026.parquet"))
fb = sched[(sched.home_division == "fbs") & (sched.away_division == "fbs") & sched.home_points.notna() & (sched.week < 7)]
L_cfb = float(pd.concat([fb.home_points, fb.away_points]).mean())
npi = npr.set_index("team")
def cfb_pts(t, o):
    b, wp = ncaa_p["beta"], ncaa_p["w_pass"]
    if t not in npi.index or o not in npi.index: return None
    return L_cfb + b * 65.0 * (wp * (npi.at[t, "off_pass"] + npi.at[o, "def_pass"]) + (1 - wp) * (npi.at[t, "off_rush"] + npi.at[o, "def_rush"]))
mt = []
for r in ncaa.itertuples():
    a_, b_ = cfb_pts(r.home, r.away), cfb_pts(r.away, r.home)
    mt.append(round(a_ + b_, 1) if a_ is not None and b_ is not None else None)
ncaa["model_total"] = mt
print("league avg points per team: NFL", round(L_nfl, 1), "CFB", round(L_cfb, 1))

# ---- server-side ESPN snapshot: fresher lines, moneylines, scores; kickoff-based "started" flags ----
esp = json.load(open(os.path.join(OUT, "espn_snapshot.json"))) if os.path.exists(os.path.join(OUT, "espn_snapshot.json")) else {"nfl": {}, "ncaa": {}}
now_utc = pd.Timestamp.now(tz="UTC")
gk = games_all[(games_all.season == 2026) & (games_all.week == 6)].copy()
gk["kick"] = pd.to_datetime(gk.gameday + " " + gk.gametime.fillna("13:00")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
kick = gk.set_index(["away_team", "home_team"]).kick
def apply_espn(df, lg, line_col):
    for c in ["started", "score_text", "ml_home", "ml_away"]:
        if c not in df.columns: df[c] = None
    df["started"] = df["started"].astype(object)
    for i, r in df.iterrows():
        rec = esp.get(lg, {}).get(f"{r.away}@{r.home}")
        if lg == "nfl":
            k = kick.get((r.away, r.home))
            if k is not None and k <= now_utc: df.at[i, "started"] = True
        if not rec: continue
        if rec.get("spread") is not None: df.at[i, line_col] = rec["spread"]
        if rec.get("total"): df.at[i, "total"] = rec["total"]
        if rec.get("ml_home") and rec.get("ml_away"): df.at[i, "ml_home"], df.at[i, "ml_away"] = rec["ml_home"], rec["ml_away"]
        if rec.get("phase") in ("in", "post"):
            df.at[i, "started"] = True
            df.at[i, "score_text"] = f"{r.away} {rec.get('away_score', 0)} – {r.home} {rec.get('home_score', 0)} · {rec.get('state', '')}"
        if rec.get("completed") and pd.isna(r.result):
            df.at[i, "result"] = rec.get("home_score", 0) - rec.get("away_score", 0)
    return df
w6 = apply_espn(w6, "nfl", "current_spread")
ncaa = apply_espn(ncaa, "ncaa", "market_spread")
espn_asof = esp.get("as_of", "")

nb_w, nb_l = int((nbt.pick_cover == 1).sum()), int((nbt.pick_cover == -1).sum())
tiers = nbt[nbt.tier != "PASS"]
bundle = {
  "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "kalshi_asof": kalshi_asof, "espn_asof": espn_asof,
  "nfl": {
    "week": 6, "params": nfl_p,
    "games": rows(w6),
    "scorecard_md": scorecard,
    "weeks45": rows(ev[["week","away","home","model_spread","open_spread","close_spread","result","pick_open","ats_open","ats_close","tier"]]),
    "power": rows(pr),
    "backtest": {"games": 448, "mae_model": 9.90, "mae_market": 9.60, "ats_all": "228-216 (51.4%)", "note": "2024-25 walk-forward, closing line as market",
                 "buckets": [{"lo": 0, "hi": 1, "w": 66, "l": 68, "pct": 0.493}, {"lo": 1, "hi": 2.5, "w": 78, "l": 73, "pct": 0.517}, {"lo": 2.5, "hi": 4, "w": 52, "l": 44, "pct": 0.542}, {"lo": 4, "hi": 99, "w": 32, "l": 31, "pct": 0.508}]},
    "kalshi": kalshi["nfl"],
    "espn_abbr": {"LAR": "LA", "WSH": "WAS"},
    "nicknames": {"Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR","Bears":"CHI","Bengals":"CIN","Browns":"CLE","Cowboys":"DAL","Broncos":"DEN","Lions":"DET","Packers":"GB","Texans":"HOU","Colts":"IND","Jaguars":"JAX","Chiefs":"KC","Rams":"LA","Chargers":"LAC","Raiders":"LV","Dolphins":"MIA","Vikings":"MIN","Patriots":"NE","Saints":"NO","Giants":"NYG","Jets":"NYJ","Eagles":"PHI","Steelers":"PIT","Seahawks":"SEA","49ers":"SF","Buccaneers":"TB","Titans":"TEN","Commanders":"WAS"},
    "cities": {"Arizona":"ARI","Atlanta":"ATL","Baltimore":"BAL","Buffalo":"BUF","Carolina":"CAR","Chicago":"CHI","Cincinnati":"CIN","Cleveland":"CLE","Dallas":"DAL","Denver":"DEN","Detroit":"DET","Green Bay":"GB","Houston":"HOU","Indianapolis":"IND","Jacksonville":"JAX","Kansas City":"KC","Los Angeles R":"LA","Los Angeles C":"LAC","Las Vegas":"LV","Miami":"MIA","Minnesota":"MIN","New England":"NE","New Orleans":"NO","New York G":"NYG","New York J":"NYJ","Philadelphia":"PHI","Pittsburgh":"PIT","Seattle":"SEA","San Francisco":"SF","Tampa Bay":"TB","Tennessee":"TEN","Washington":"WAS"},
  },
  "ncaa": {
    "date": "2026-10-17", "params": ncaa_p, "games": rows(ncaa), "power": rows(npr.head(60)),
    "backtest": {"games": int(len(nbt)), "mae_model": round(nbt.err_model.mean(), 2), "mae_blend": round(nbt.err_blend.mean(), 2), "mae_market": round(nbt.err_market.mean(), 2),
                 "ats_all": f"{nb_w}-{nb_l}", "ats_tiers": f"{int((tiers.pick_cover==1).sum())}-{int((tiers.pick_cover==-1).sum())}", "note": "2026 Weeks 4-7 walk-forward vs ESPN/DraftKings line",
                 "buckets": [{"lo": 0, "hi": 3, "w": 46, "l": 45, "pct": 0.505}, {"lo": 3, "hi": 5, "w": 14, "l": 23, "pct": 0.378}, {"lo": 5, "hi": 8, "w": 22, "l": 22, "pct": 0.5}, {"lo": 8, "hi": 99, "w": 7, "l": 13, "pct": 0.35}]},
    "kalshi": kalshi["ncaa"],
  },
}
json.dump(bundle, open(os.path.join(OUT, "app_data.json"), "w"))
print(os.path.getsize(os.path.join(OUT, "app_data.json")), "bytes;", len(bundle["nfl"]["games"]), "nfl games;", len(bundle["ncaa"]["games"]), "ncaa games")

