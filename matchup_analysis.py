"""Batter-bowler matchup statistics calculated from delivery data."""
import numpy as np
import pandas as pd


def matchup_summary(ball: pd.DataFrame, batter: str, bowler: str, seasons=None, innings=None) -> dict:
    frame = ball[(ball.batter == batter) & (ball.bowler == bowler)].copy()
    if seasons:
        frame = frame[frame.season.isin(seasons)]
    if innings:
        frame = frame[frame.innings.isin(innings)]
    if frame.empty:
        return {"deliveries": 0}
    dismissals = frame[frame.player_out.eq(batter) & frame.bowler_wicket]
    runs = int(frame.batter_runs.sum())
    legal_balls = int(frame.legal_ball.sum())
    balls_faced = int(frame.batter_ball.sum())
    return {"runs": runs, "deliveries": int(len(frame)), "legal_balls": legal_balls,
        "balls_faced": balls_faced, "strike_rate": runs * 100 / balls_faced if balls_faced else np.nan,
        "bowler_dismissals": int(len(dismissals)), "fours": int((frame.batter_runs == 4).sum()),
        "sixes": int((frame.batter_runs == 6).sum()), "matches": int(frame.match_id.nunique()), "data": frame}
