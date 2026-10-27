"""Team comparison, head-to-head, and season analysis helpers."""
import numpy as np
import pandas as pd


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> dict:
    pair = matches[((matches.team1 == team_a) & (matches.team2 == team_b)) | ((matches.team1 == team_b) & (matches.team2 == team_a))]
    decisions = pair[pair.match_winner.isin([team_a, team_b])]
    return {"matches": int(len(pair)), "team_a_wins": int((decisions.match_winner == team_a).sum()), "team_b_wins": int((decisions.match_winner == team_b).sum()), "no_result_or_other": int(len(pair) - len(decisions))}


def batting_bowling_summary(ball: pd.DataFrame, team: str) -> dict:
    batting = ball[ball.batting_team == team]
    bowling = ball[(ball.team1 == team) | (ball.team2 == team)]
    conceded = bowling[bowling.batting_team != team]
    bat_legal = int(batting.legal_ball.sum())
    bowl_legal = int(conceded.legal_ball.sum())
    return {"runs_scored": int(batting.total_runs.sum()), "batting_strike_rate": float(batting.batter_runs.sum() * 100 / batting.batter_ball.sum()) if batting.batter_ball.sum() else np.nan,
        "runs_conceded": int(conceded.bowler_runs_conceded.sum()), "bowling_economy": float(conceded.bowler_runs_conceded.sum() * 6 / bowl_legal) if bowl_legal else np.nan,
        "legal_balls_faced": bat_legal, "legal_balls_bowled": bowl_legal}
