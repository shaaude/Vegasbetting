## Weeks 4-5 scorecard (31 games, walk-forward)

- Mean absolute error vs final margin: model 8.26 | blend 8.40 | opening line 8.47 | closing line 8.81
- Straight-up winners: model 71.0% | closing line 58.1%
- ATS, every game, pick = model side vs opener: 16-14-1 (53.3%); vs close: 18-10-3 (64.3%)
- ATS, PLAY tier only (|edge vs open| >= 4 pts, n=3): 2-1 (66.7%); same tickets graded at the close: 2-1 (66.7%)
- ATS, PLAY + LEAN tiers (|edge vs open| >= 2.5 pts, n=12): 8-4 (66.7%); at the close: 8-4 (66.7%)
  - |edge vs open| 0-1.5 pts: n=12, ATS 4-7-1 (36.4%)
  - |edge vs open| 1.5-3 pts: n=11, ATS 6-5 (54.5%)
  - |edge vs open| 3-99 pts: n=8, ATS 6-2 (75.0%)
- Of 19 games whose line moved >= 0.5, the market moved toward the model side 68% of the time
- Win-probability Brier: model 0.224 | closing line 0.232
- Week 4: MAE model 6.35 vs open 7.22 vs close 7.31; all-game ATS vs open 9-6-1 (60.0%); PLAY+LEAN 5-1 (83.3%)
- Week 5: MAE model 10.30 vs open 9.80 vs close 10.40; all-game ATS vs open 7-8 (46.7%); PLAY+LEAN 3-3 (50.0%)

## Refinement diagnostics

- Regression result ~ a*model + b*close: a=1.80, b=-0.34 (intercept -2.64); tuned blend weight on market = 0.80
- Model residual (result - model): mean -1.98, home favorites +1.07, home dogs -9.43
- Games with a material QB adjustment (n=10): model MAE 9.71 vs close 10.75, ATS vs open 5-4-1 (55.6%)
- Large-edge games (>=3 pts vs open, n=8): ATS 6-2 (75.0%); these are where a structural miss would show