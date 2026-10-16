# NFL Spread Model — 2026 season

An opponent-adjusted EPA power-rating model that is anchored to the betting market, adjusted for
quarterback changes, injuries, venue, rest and travel, and graded against opening and closing spreads.
Built from the five-layer framework in the brief (EPA engine → personnel → environment → market → Kelly).

Everything runs from free nflverse data; `fetch_data.sh` refreshes it, `run_2026.py` rebuilds the
Weeks 4–5 scorecard and the current-week board.

## 1. What the model does

For every game the predicted home margin is

```
margin = HFA(venue) + [R_home − R_away] + QB_adj + injury_adj + rest_adj + travel_adj
```

| Layer in the brief | Implementation | File |
|---|---|---|
| EPA & success-rate engine | nflfastR play-by-play, pass/run snaps only, 4th-quarter garbage time removed (win prob <4% or >96%). Per team-game dropback EPA and rush EPA. | `data.py` |
| Opponent-adjusted, regressed ratings | Weighted Bayesian ridge: `EPA/play = off[team] − def[opp]`, solved jointly for all 32 offences and defences, shrunk toward a prior with strength `k` (in plays). Separate pass and rush systems, combined 70/30. | `ratings.py` |
| Prior / market intelligence | The prior is the **market-implied power rating** — team strengths that best reproduce the spreads already posted (ridge fit, `spread = HFA + R_home − R_away`). Two modes: `early` = Weeks 1–3 lines only (the preseason view), `rolling` = all lines before the week, recency-weighted (decay 0.6/week). | `ratings.py`, `model.py` |
| QB adjustment | Every QB gets a shrunken EPA/dropback value (all seasons 2023→, 0.65 decay per season, shrunk toward replacement level −0.10 with weight 200 dropbacks). A team's sample is credited to the dropback-weighted mix of QBs who actually played; the week's named starter is valued against that mix: `Δpts = qb_scale × β × 34 × (v_starter − v_mix)`. Last season's passing offence is shifted the same way for offseason QB changes. | `qb.py` |
| Injuries (non-QB) | Rank-1 starters from the ESPN depth chart in force six days before kickoff × probability out (Out 1.0, Doubtful 0.8, Questionable 0.3; before the Friday report: DNP 0.5, Limited 0.15) × position weight (LT 1.0, EDGE 0.7, CB/RT 0.6, WR 0.55, C/S/DL/TE/RB 0.4, G/LB 0.3–0.35, K 0.4). Diminishing returns within a position group. Only the change versus the team's sample-average burden is applied, so a long-term absence already in the EPA sample is not double-counted. | `injuries.py` |
| Home field / venue | League HFA 1.5 pts (tuned) + venue term from 2021–25 home-minus-away margins, shrunk 60% and capped at ±1 (DEN +1.0, CHI +1.0, MIA +1.0 … SEA −0.9, ARI −0.8). Neutral sites (London) get 0. | `context.py` |
| Rest & travel | Rest-day differential tested and set to 0 (no measurable effect 2024–25). Travel: −0.25/zone beyond the first for the visitor. | `context.py` |
| Weather | Wind ≥10 mph scales the passing-rating gap down (outdoor/open only). Implemented but off by default: no forecast feed, and the games.csv wind field is only known after kickoff. | `context.py` |
| Officiating | Referee penalty-yard profile is computed (`referee_profile`) as a diagnostic. It is not in the spread — there is no evidence crews move margins. | `context.py` |
| Market blend & edge | `blend = 0.8 × market + 0.2 × model` (weight tuned on 2024–25) feeds the win and cover probabilities. Tiers on the raw gap: **PLAY** ≥4 pts, **LEAN** ≥2.5 pts. | `model.py` |
| Bankroll | Quarter-Kelly on the blended cover probability at −110, capped at 3% of bankroll, zero unless the blended edge clears 3%. | `model.py` |

Not implemented because the data is not freely available: public ticket % vs money handle (reverse line
movement), live Kalshi/Polymarket feeds, PFF/ESPN trench win rates. The FTN charting file (pressure, blitz,
play-action) is downloaded and can be layered in later.

## 2. Training protocol

* **Ratings**: for Week *w* only games before *w* are used. Week 4 predictions use Weeks 1–3 of 2026 exactly as
  asked; Week 5 rolls forward with Week 4 added.
* **Hyper-parameters** (shrinkage strengths, pass/rush weight, EPA→points scale, QB/injury scales, HFA, blend
  weight) were **not** fit to the 31 games of Weeks 4–5 — six knobs on 31 games would only fit noise. They were
  tuned by walk-forward on 2024 and 2025, Weeks 4–18 (448 games), minimising mean absolute error against the
  final margin. 2026 Weeks 4–5 are therefore a clean out-of-sample test.

### 2024–25 walk-forward results (448 games, closing line as the market)

| Config | MAE model | MAE blend | MAE closing line | ATS all picks vs close | Win-prob Brier |
|---|---|---|---|---|---|
| `early` prior (Weeks 1–3 lines, EPA weight ~33%) | 10.06 | 9.65 | 9.60 | 202-242 (45.5%) | 0.215 |
| `rolling` prior (recency-weighted lines, EPA weight ~12%) | 9.90 | 9.61 | 9.60 | 228-216 (51.4%) | 0.213 |

What the tuning found:

* Previous-season EPA adds nothing once the prior is the market and QB changes are handled (`rho_prev` → 0).
* Full-EPA (no turnover damping) beat damped variants.
* QB adjustments are the single most valuable layer: MAE 10.23 → 10.10 going from off to full scale.
* Injury index helps modestly (scale 1.0); rest differential does nothing; HFA 1.5 beats 2.0.
* The closing line is still the better forecaster (as expected — nothing built on public data reliably beats
  it). The blended margin matches it; the model's own picks are a coin flip against the close over 448 games,
  with a mild, not-significant edge in the 1–4.5 point disagreement band for the rolling config (139-126, 52.5%)
  and PLAY-tier games (gap ≥4) at 32-31. The market's own win-probability Brier is 0.202 vs the model's 0.213.

Win probabilities are reasonably calibrated (rolling config, 2024–25): predicted 25% → actual 29%, 36% → 30%,
45% → 38%, 55% → 52%, 65% → 70%, 79% → 80% — the model is a little too kind to home teams in toss-ups.

## 3. 2026 Weeks 4–5 scorecard (out of sample)

Opening lines are DraftKings openers from SI Betting (Sep 28 for Week 4, Oct 5 for Week 5); the GB–TB opener was
never posted (Mayfield injury) so the closing line is used there. Closing lines are from nflverse.

| | Model | Opening line | Closing line |
|---|---|---|---|
| MAE vs final margin | **8.26** | 8.47 | 8.81 |
| Straight-up winners | 71% | — | 58% |
| Win-prob Brier | 0.224 | — | 0.232 |

* ATS, model side in every game: **16-14-1 vs the opener, 18-10-3 vs the close.**
* PLAY + LEAN tiers (gap ≥2.5 pts vs the opener): **8-4**; gap ≥3 pts: 6-2.
* In 13 of the 19 games whose line moved ≥0.5, it moved toward the model's side — the model was reading the
  market's own direction (closing-line value), which is the more reliable signal than 31 ATS results.
* Week 4 was strong (MAE 6.35 vs 7.31 for the close, 9-6-1 ATS); Week 5 was ordinary (10.30 vs 10.40, 7-8).
* The big QB calls were right directionally: BAL with Huntley (−6.0 pts applied, market moved 7), TB with
  Jalon Daniels, CHI with Keenum/Bagent, ATL with Penix back (+3.4).

Full per-game detail (ratings, every adjustment, both lines, result, ATS grade) is in `output/predictions_weeks4_5.csv`.

**Read this with the right scale**: 31 games is a small sample. 18-10 against the close is encouraging and
consistent with the backtest's calibration, but the 448-game backtest says to expect roughly break-even against
the close long-run; the realistic edge is against **openers** and in **QB/injury news** the market has not yet
priced.

## 4. Refinements made after grading

1. **Prior switched to the rolling market rating** (production default): better MAE, Brier and ATS on 448
   games; 2026 Weeks 4–5 MAE 8.26 vs 8.36 for the Weeks-1–3-only prior. The pure variant is kept
   (`params_tuned.json`, `prior_mode=early`) and shown on the board as `model_spread_pure`.
2. **Injury model now reads practice participation** when the final report is not yet out, and the week's
   named starter is checked against the report: the board flags `qb_risk` and shows `model_spread_if_backup`
   using the chart's #2 QB.
3. **Rest adjustment set to zero**, HFA to 1.5, injury scale to 1.0, QB scale to 1.0 — all from the backtest.
4. **Recommendation tiers moved to point gaps** (PLAY ≥4, LEAN ≥2.5). A 3% probability edge against the close
   is almost never defensible with a 20%-weight model, so the stake column stays at 0 for most games by design;
   the tiers tell you where the model disagrees enough to act against an opener.
5. Diagnostics on the 31 games: no home/away bias worth acting on (residual mean −2.0, dominated by two
   blowouts — IND 30-13 in London, ATL 45-24); nothing structural to change.

## 5. Week 6 board

`output/week6_board.csv` — ratings through Week 5, DraftKings openers (Oct 12) and the current nflverse line,
model spread (rolling), `model_spread_pure`, edge vs open and vs current, tier, Kelly stake, QB-risk flags and
the mid-week injury detail. Thursday's SEA@DEN (DEN +2.5, LEAN) is already graded: DEN 3-point win, cover.

Two games hinge on Friday's QB news: BAL@CLE (Lamar Jackson DNP Wed/Thu — model BAL −7.4 with him, CLE −0.2
with Huntley) and CAR@PHI (Jalen Hurts DNP — CAR −1.7 with him, CAR −5.2 with McKee; the market has already
moved to CAR −3.5). Re-run after the final injury report.

`output/power_ratings_through_week5.csv` has the full 32-team table (rating, pass and rush components,
raw adjusted EPA units, market prior).

## 6. Running it

```bash
bash fetch_data.sh                     # refresh nflverse inputs (default dir /tmp/nfl, or set NFL_DATA_DIR)
pip install pandas numpy scipy pyarrow
python3 run_2026.py                    # rebuilds output/ (Weeks 4-5 scorecard + Week 6 board)
NFL_PARAMS=params_tuned.json python3 run_2026.py   # same with the Weeks-1-3-only prior
python3 backtest.py                    # re-tune on 2024-25 (~4 min)
python3 tune_rolling.py                # re-tune the rolling-prior config (~2 min)
```

For Week 7 onward: add the openers to `opening_lines.csv` (home-favoured = positive, nflverse convention),
change the week numbers at the bottom of `run_2026.py` (or call `predict_week(cache, 7, p)` directly), and
re-run after the Friday injury report.

## 7. Files

```
data.py         loaders, play filter, per-game unit stats
ratings.py      ridge-adjusted EPA ratings, market-implied ratings
qb.py           QB values, sample mix, starter-change delta
injuries.py     starters, position weights, injury index, backup QB
context.py      HFA, venue, rest, travel, wind, referee profile
model.py        Params, SeasonCache, assembly, probabilities, Kelly, grading
backtest.py     walk-forward evaluation + staged tuning (early prior)
tune_rolling.py tuning for the rolling market prior
run_2026.py     Weeks 4-5 scorecard, refinement diagnostics, Week 6 board
opening_lines.csv   DraftKings openers, Weeks 4-6
params_tuned.json / params_tuned_rolling.json   tuned settings (rolling = default)
backtest_2024_2025*.csv, tuning_*.csv           backtest game logs and tuning grids
output/         predictions_weeks4_5.csv, scorecard_weeks4_5.md, power_ratings_through_week5.csv, week6_board.csv
```
