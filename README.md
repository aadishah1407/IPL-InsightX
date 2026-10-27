# IPL Analytics System

A Streamlit data-science demonstration for IPL 2027 scenarios, built around the supplied `ipl_ball_by_ball_2021_2026.csv` and `ipl_matches_2021_2026.csv` files.

> **Important:** the source data has historical performance only. It does not verify 2027 squads, contracts, roles, or availability. `ipl_squads.csv` is intentionally a header-only template. Predictions and XIs are provisional decision support, not guaranteed outcomes.

## Start the application

1. Install dependencies from `requirements.txt` in your chosen Python environment.
2. From this folder, run `streamlit run app.py`.
3. Open the local URL printed by Streamlit.
4. The XI selector automatically builds franchise player pools from observed batting and bowling participation in the 2021–2026 ball-by-ball data.

The two historical CSVs can stay at the project root. The loader also supports putting them under `data/`.

## Squad CSV schema

Required columns: `player_name`, `team`, `role`, `is_overseas`, `is_available`, `batting_style`, `bowling_style`.

Use `true`/`false` values for overseas and availability. Teams must use the franchise names shown in the app. Suitable roles include Batter, Wicketkeeper, All-rounder, and Bowler. Do not populate this file from guesses; record a source/date alongside your research if required for your submission.

## Project structure

- `app.py` – interactive Streamlit UI with five sidebar sections
- `data_loader.py`, `preprocessing.py` – input discovery, schema checks and cleaning
- `feature_engineering.py` – player-match, player-season, phase and team aggregations
- `model_training.py` – leakage-aware Random Forest, chronological holdout, baselines and 2027 features
- `playing_xi.py` – transparent score-based selection from historical team participation
- `team_analysis.py`, `matchup_analysis.py`, `visualizations.py` – reusable analytics and charts
- `tests/` – unit tests for cricket accounting and chronology
- `outputs/` – generated, inspectable CSVs (created when requested in Data and Squad Management)

## Data and modelling choices

- The supplied files contain 102,115 ball records and 427 match rows spanning six seasons (2021–2026); the observed franchise alias Royal Challengers Bangalore is normalized to Royal Challengers Bengaluru.
- Exact duplicate rows are removed; negative scores and rows without essential player/match identifiers are excluded. Optional fields are handled defensively.
- Cricsheet `over` is zero-indexed. Powerplay is overs 0–5; middle overs 6–14; death overs 15 onward.
- Batter balls faced exclude wides (no-balls count as batter balls faced). Bowler legal balls exclude wides and no-balls. Bowling runs exclude byes, leg-byes and penalties. Overs are legal balls divided by six.
- Bowler wickets are limited to credited dismissal kinds (bowled, caught, caught and bowled, lbw, stumped, hit wicket); run-outs are not credited.
- Training examples predict target-season runs from information through the prior season. Only players observed in consecutive seasons are supervised examples; missed seasons/retirements and participation probability are not modelled. The latest target season is held out as a whole chronological block, so the actual split can differ from an exact 80:20. MAE, RMSE, R² where meaningful, training-mean baseline and previous-runs baseline are reported separately from the final model trained on all eligible examples.
- The recommendation score combines model forecast, recent batting, strike rate, wickets and economy with explicit role bonuses; it then checks exactly 11 distinct players, role coverage and a maximum of four overseas players. Incomplete data produces a warning rather than fabricated names.
- The XI selector's historical player-team associations are inferred from match participation. The historical CSV does not identify wicketkeepers, overseas eligibility, contracts, or current availability; role labels are heuristics and those constraints cannot be fully verified.
- Season win rates use decided matches as the denominator. Tied/no-result/abandoned games are not counted as losses.

## Inspectable exports

Use **Data and Squad Management → Build and save inspectable CSV outputs** to write cleaned ball/match data, player-match and player-season summaries, team and phase summaries, and the supervised modelling table into `outputs/`.

## Tests

Run `pytest -q` from the project folder.
