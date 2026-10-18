"""Server-side ESPN scoreboard pull (lines, totals, moneylines, scores) -> output/espn_snapshot.json.
Runs on GitHub's runners, where browser CORS rules don't apply. Never fails the build: on any error the
previous snapshot is kept. Odds are only overwritten when ESPN still lists them (books pull odds at kickoff)."""
import json, os, re, urllib.request
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__)); PATH = os.path.join(HERE, "output", "espn_snapshot.json")
NFL_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
URLS = {"nfl": ["https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"],
        "ncaa": [f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups={g}&dates={d}&limit=100"
                 for g in (80,) for d in ("20261016", "20261017")]}

def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": "edge-board/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def spread_from(details, h_abbr, a_abbr, h_loc, a_loc):
    if not details: return None
    if details.strip().upper() in ("EVEN", "PK", "PICK"): return 0.0
    m = re.match(r"^(.+?)\s([+-]?\d+(?:\.\d+)?)$", details.strip())
    if not m: return None
    fav, num = m.group(1).strip(), abs(float(m.group(2)))
    if fav in (h_abbr, h_loc) or (h_loc and h_loc.startswith(fav)): return num
    if fav in (a_abbr, a_loc) or (a_loc and a_loc.startswith(fav)): return -num
    return None

def main():
    snap = json.load(open(PATH)) if os.path.exists(PATH) else {"nfl": {}, "ncaa": {}}
    n = {"nfl": 0, "ncaa": 0}
    for lg, urls in URLS.items():
        for u in urls:
            try:
                j = get(u)
            except Exception as e:
                print(lg, "ESPN fetch failed:", e); continue
            for ev in j.get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                cs = comp.get("competitors", [])
                h = next((c for c in cs if c.get("homeAway") == "home"), None); a = next((c for c in cs if c.get("homeAway") == "away"), None)
                if not h or not a: continue
                if lg == "nfl":
                    hk = NFL_FIX.get(h["team"]["abbreviation"], h["team"]["abbreviation"]); ak = NFL_FIX.get(a["team"]["abbreviation"], a["team"]["abbreviation"])
                else:
                    hk, ak = h["team"].get("location"), a["team"].get("location")
                key = f"{ak}@{hk}"; rec = snap[lg].get(key, {})
                st = (comp.get("status") or {}).get("type", {})
                rec["state"] = st.get("shortDetail", ""); rec["phase"] = st.get("state", "")
                rec["completed"] = bool(st.get("completed"))
                if st.get("state") in ("in", "post"):
                    rec["home_score"] = int(h.get("score") or 0); rec["away_score"] = int(a.get("score") or 0)
                o = (comp.get("odds") or [None])[0]
                if o and st.get("state") == "pre":
                    sp = spread_from(o.get("details"), h["team"].get("abbreviation"), a["team"].get("abbreviation"), h["team"].get("location"), a["team"].get("location"))
                    if sp is not None: rec["spread"] = sp
                    if o.get("overUnder"): rec["total"] = o["overUnder"]
                    hm = (o.get("homeTeamOdds") or {}).get("moneyLine"); am = (o.get("awayTeamOdds") or {}).get("moneyLine")
                    if hm and am: rec["ml_home"], rec["ml_away"] = hm, am
                    rec["odds_asof"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                snap[lg][key] = rec; n[lg] += 1
    snap["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if (n["nfl"] or n["ncaa"]) else snap.get("as_of")
    json.dump(snap, open(PATH, "w"), indent=1)
    print(f"ESPN: {n['nfl']} NFL and {n['ncaa']} college games updated; as_of {snap.get('as_of')}")

if __name__ == "__main__":
    main()
