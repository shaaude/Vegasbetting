"""Refresh kalshi_snapshot.json from Kalshi's public market API (server side, no browser CORS limits).
Matches each open game market to a game on the current NFL and NCAA boards; keeps old prices for games
it cannot match. Never fails the build: errors are printed and the old snapshot stays."""
import json, os, re, sys, urllib.request
from datetime import datetime, timezone, timedelta
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "output")
API = "https://api.elections.kalshi.com/trade-api/v2/markets"
CITIES = {"Arizona":"ARI","Atlanta":"ATL","Baltimore":"BAL","Buffalo":"BUF","Carolina":"CAR","Chicago":"CHI","Cincinnati":"CIN","Cleveland":"CLE",
  "Dallas":"DAL","Denver":"DEN","Detroit":"DET","Green Bay":"GB","Houston":"HOU","Indianapolis":"IND","Jacksonville":"JAX","Kansas City":"KC",
  "Los Angeles R":"LA","Los Angeles C":"LAC","Las Vegas":"LV","Miami":"MIA","Minnesota":"MIN","New England":"NE","New Orleans":"NO",
  "New York G":"NYG","New York J":"NYJ","Philadelphia":"PHI","Pittsburgh":"PIT","Seattle":"SEA","San Francisco":"SF","Tampa Bay":"TB",
  "Tennessee":"TEN","Washington":"WAS"}

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "edge-board/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def markets(series):
    out, cursor = [], ""
    for _ in range(10):
        j = get(f"{API}?series_ticker={series}&status=open&limit=1000" + (f"&cursor={cursor}" if cursor else ""))
        out += j.get("markets", []); cursor = j.get("cursor") or ""
        if not cursor: break
    return out

def num(m, key):
    """Price in dollars from either the *_dollars string fields or the legacy cent integers."""
    for k in (key + "_dollars", key):
        v = m.get(k)
        if v in (None, ""): continue
        try: v = float(v)
        except (TypeError, ValueError): continue
        return v / 100 if v > 1 else v
    return None

def price(m):
    bid, ask, last = num(m, "yes_bid"), num(m, "yes_ask"), num(m, "last_price")
    if bid and ask and 0 < bid <= ask and ask - bid <= 0.15: return round((bid + ask) / 2, 3)
    return round(last, 3) if last else None

def norm(s):
    s = s.lower().replace("st.", "state").replace("&", "and")
    s = re.sub(r"\(.*?\)", "", s); s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def nfl_team(sub):
    if sub in CITIES: return CITIES[sub]
    for c, a in CITIES.items():
        if sub.startswith(c): return a
    return None

def date_tags(dates):
    return {d.strftime("%y%b%d").upper() for d in dates}

def group(ms, team_fn):
    ev = {}
    for m in ms:
        t = team_fn(m.get("yes_sub_title") or "")
        p = price(m)
        if t and p is not None:
            ev.setdefault(m["event_ticker"], {})[t] = p
    return ev

def main():
    path = os.path.join(HERE, "kalshi_snapshot.json")
    snap = json.load(open(path)) if os.path.exists(path) else {"nfl": {}, "ncaa": {}}
    n_nfl = n_ncaa = 0
    # NFL
    try:
        w = pd.read_csv(os.path.join(OUT, "week6_board.csv"))
        ev = group(markets("KXNFLGAME"), nfl_team)
        for r in w.itertuples():
            tags = date_tags([pd.Timestamp(r.gameday)])
            for tk, teams in ev.items():
                if r.home in teams and r.away in teams and any(t in tk for t in tags):
                    snap["nfl"][f"2026_06_{r.away}_{r.home}"] = {r.home: teams[r.home], r.away: teams[r.away]}; n_nfl += 1; break
    except Exception as e:
        print("NFL Kalshi refresh failed:", e)
    # NCAA
    try:
        b = pd.read_csv(os.path.join(OUT, "ncaa_saturday_board.csv"))
        ev = group(markets("KXNCAAFGAME"), norm)
        for r in b.itertuples():
            k = pd.Timestamp(r.kickoff_utc); tags = date_tags([k, k - timedelta(hours=8)])
            h, a = norm(r.home), norm(r.away)
            for tk, teams in ev.items():
                hm = next((t for t in teams if t == h), None) or next((t for t in teams if h.startswith(t) or t.startswith(h)), None)
                am = next((t for t in teams if t == a), None) or next((t for t in teams if a.startswith(t) or t.startswith(a)), None)
                if hm and am and hm != am and any(t in tk for t in tags):
                    snap["ncaa"][f"{r.away}@{r.home}"] = {r.home: teams[hm], r.away: teams[am]}; n_ncaa += 1; break
    except Exception as e:
        print("NCAA Kalshi refresh failed:", e)
    if n_nfl or n_ncaa:
        snap["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        json.dump(snap, open(path, "w"), indent=1)
    print(f"Kalshi: matched {n_nfl} NFL and {n_ncaa} NCAA games; as_of {snap.get('as_of')}")

if __name__ == "__main__":
    main()
