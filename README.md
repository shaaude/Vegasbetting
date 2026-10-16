# Vegasbetting · Edge Board

NFL and college football spread model with a browsable board, best bets and a parlay builder.

**Site:** https://shaaude.github.io/Vegasbetting/ (GitHub Pages, served from `docs/`)

## What's here
| Path | What it is |
|---|---|
| `docs/index.html` | The Edge Board site. Fetches live ESPN lines/scores, Kalshi and Polymarket prices in your browser on open and every 2 minutes; falls back to the last rebuild's snapshot if a feed refuses the request. |
| `model/` | Python pipeline: opponent-adjusted EPA ratings anchored to market power ratings, QB and injury adjustments, home field, probabilities, tiers, Kelly. `METHODOLOGY.md` has the full write-up and backtests. |
| `model/output/` | Latest boards, power ratings, backtests and the data bundle the page embeds. |
| `.github/workflows/rebuild.yml` | Rebuilds the models and the site on a schedule (Fri evening → Mon afternoon around injury reports, inactives and kickoffs) and on demand. |

## Refresh
- **Odds** refresh on their own in the browser.
- **Models** rebuild on the workflow schedule, or now: Actions tab → *Rebuild Edge Board* → *Run workflow*.
- **Late QB / injury news** nflverse hasn't posted: add a row to `model/starter_overrides.csv` or `model/injury_overrides.csv`, commit, then run the workflow.

## Run locally
```bash
pip install pandas numpy scipy pyarrow
cd model && python rebuild.py && python build_page.py --live --out ../docs/index.html
```

For analysis only; no guarantee of profit.
