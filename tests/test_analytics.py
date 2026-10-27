import pandas as pd
import pytest

from feature_engineering import historical_squad_stats, player_season_stats
from model_training import build_next_season_dataset
from playing_xi import recommend_xi
from preprocessing import clean_data
from matchup_analysis import matchup_summary


def fixture_frames():
    ball = pd.DataFrame([
        {"match_id": 1, "season": 2021, "match_date": "01-04-2021", "team1": "Royal Challengers Bangalore", "team2": "Mumbai Indians", "batting_team": "Mumbai Indians", "innings": 1, "over": 0, "delivery_in_over": 1, "batter": "A", "bowler": "B", "batter_runs": 4, "total_runs": 4, "extras_total": 0, "extras_json": "{}", "is_wicket": 0, "wicket_kinds": "", "player_out": ""},
        {"match_id": 1, "season": 2021, "match_date": "01-04-2021", "team1": "Royal Challengers Bangalore", "team2": "Mumbai Indians", "batting_team": "Mumbai Indians", "innings": 1, "over": 0, "delivery_in_over": 2, "batter": "A", "bowler": "B", "batter_runs": 0, "total_runs": 1, "extras_total": 1, "extras_json": "{\"wides\":1}", "is_wicket": 0, "wicket_kinds": "", "player_out": ""},
        {"match_id": 1, "season": 2021, "match_date": "01-04-2021", "team1": "Royal Challengers Bangalore", "team2": "Mumbai Indians", "batting_team": "Mumbai Indians", "innings": 1, "over": 0, "delivery_in_over": 3, "batter": "A", "bowler": "B", "batter_runs": 0, "total_runs": 0, "extras_total": 0, "extras_json": "{}", "is_wicket": 1, "wicket_kinds": "run out", "player_out": "A"},
        {"match_id": 1, "season": 2021, "match_date": "01-04-2021", "team1": "Royal Challengers Bangalore", "team2": "Mumbai Indians", "batting_team": "Mumbai Indians", "innings": 1, "over": 0, "delivery_in_over": 4, "batter": "A", "bowler": "B", "batter_runs": 0, "total_runs": 1, "extras_total": 1, "extras_json": "{\"noballs\":1}", "is_wicket": 0, "wicket_kinds": "", "player_out": ""},
    ])
    matches = pd.DataFrame([{"match_id": 1, "season": 2021, "match_date": "01-04-2021", "team1": "Royal Challengers Bangalore", "team2": "Mumbai Indians", "match_winner": "Mumbai Indians"}])
    return ball, matches


def test_cricket_accounting_and_runout_credit():
    ball, matches = clean_data(*fixture_frames())
    assert ball.batter_ball.tolist() == [True, False, True, True]
    assert ball.legal_ball.tolist() == [True, False, True, False]
    assert ball.bowler_wicket.tolist() == [False, False, False, False]
    assert matches.team1.iloc[0] == "Royal Challengers Bengaluru"
    stats = player_season_stats(ball).set_index("player")
    batter_stats = stats.loc["A"]
    bowler_stats = stats.loc["B"]
    assert batter_stats.runs == 4
    assert batter_stats.balls_faced == 3
    assert bowler_stats.legal_balls == 2
    assert bowler_stats.wickets == 0


def test_training_rows_use_prior_seasons_and_next_season_labels():
    stats = pd.DataFrame([
        {"season": year, "player": "A", "runs": runs, "strike_rate": 100.0, "batting_average": 30.0, "batting_innings": 10, "total_matches": 10, "wickets": 1}
        for year, runs in [(2021, 100), (2022, 120), (2023, 90)]
    ])
    examples = build_next_season_dataset(stats)
    assert list(examples.target_season) == [2022, 2023]
    assert examples.prev_runs.tolist() == [100.0, 120.0]
    assert examples.target_runs.tolist() == [120, 90]


def test_matchup_counts_legal_balls_and_credited_dismissals():
    ball, matches = clean_data(*fixture_frames())
    result = matchup_summary(ball, "A", "B")
    assert result["runs"] == 4
    assert result["legal_balls"] == 2
    assert result["balls_faced"] == 3
    assert result["strike_rate"] == pytest.approx(4 / 3 * 100)
    assert result["bowler_dismissals"] == 0


def test_historical_squad_uses_recorded_team_participation():
    ball, matches = clean_data(*fixture_frames())
    squads = historical_squad_stats(ball)
    assert set(squads.team) == {"Mumbai Indians", "Royal Challengers Bengaluru"}
    assert squads.set_index("player_name").loc["A", "team"] == "Mumbai Indians"
    assert squads.set_index("player_name").loc["B", "team"] == "Royal Challengers Bengaluru"
    assert squads.is_overseas.isna().all()


def test_recommend_xi_uses_integer_score_and_keeps_team_selection_logic():
    squad = pd.DataFrame([
        {"player_name": f"P{i}", "team": "Royal Challengers Bengaluru", "role": "Batter", "is_overseas": False, "is_available": True, "batting_style": "Right-hand bat", "bowling_style": ""}
        for i in range(1, 13)
    ])
    season_stats = pd.DataFrame([
        {"player": f"P{i}", "season": 2025, "runs": 400 + i, "strike_rate": 120.0 + i, "wickets": 0, "economy": 8.0, "total_matches": 10}
        for i in range(1, 13)
    ])
    forecasts = pd.DataFrame({"player": [f"P{i}" for i in range(1, 13)], "predicted_runs": [350 + i for i in range(1, 13)]})

    result = recommend_xi(squad, season_stats, forecasts, "Royal Challengers Bengaluru", require_wicketkeeper=False)

    assert len(result["xi"]) == 11
    assert (result["xi"]["score"] == result["xi"]["score"].astype(int)).all()
    assert (result["xi"]["predicted_runs"] == result["xi"]["predicted_runs"].astype(int)).all()
    assert (result["xi"]["recent_runs"] == result["xi"]["recent_runs"].astype(int)).all()
