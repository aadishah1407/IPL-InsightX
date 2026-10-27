"""Player, team, and phase aggregation utilities."""
import numpy as np
import pandas as pd


def player_match_stats(ball: pd.DataFrame) -> pd.DataFrame:
    """Return one batting/bowling summary per player and match."""
    keys = ["match_id", "season", "match_date", "venue", "city"]
    keys = [key for key in keys if key in ball.columns]
    bat = ball.groupby(keys + ["batter"], dropna=False).agg(
        runs=("batter_runs", "sum"), balls_faced=("batter_ball", "sum"),
        fours=("batter_runs", lambda s: int((s == 4).sum())), sixes=("batter_runs", lambda s: int((s == 6).sum())),
        batting_innings=("innings", "nunique"),
    ).reset_index().rename(columns={"batter": "player"})
    bowl = ball.groupby(keys + ["bowler"], dropna=False).agg(
        runs_conceded=("bowler_runs_conceded", "sum"), legal_balls=("legal_ball", "sum"),
        wickets=("bowler_wicket", "sum"), bowling_innings=("innings", "nunique"),
    ).reset_index().rename(columns={"bowler": "player"})
    result = bat.merge(bowl, on=keys + ["player"], how="outer")
    for col in ("runs", "balls_faced", "fours", "sixes", "batting_innings", "runs_conceded", "legal_balls", "wickets", "bowling_innings"):
        result[col] = result[col].fillna(0)
    result["strike_rate"] = np.where(result.balls_faced > 0, result.runs * 100 / result.balls_faced, np.nan)
    result["economy"] = np.where(result.legal_balls > 0, result.runs_conceded * 6 / result.legal_balls, np.nan)
    return result


def player_season_stats(ball: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player batting and bowling season totals and rates."""
    bat = ball.groupby(["season", "batter"], as_index=False).agg(
        runs=("batter_runs", "sum"), balls_faced=("batter_ball", "sum"),
        fours=("batter_runs", lambda s: int((s == 4).sum())), sixes=("batter_runs", lambda s: int((s == 6).sum())),
        batting_innings=("match_id", "nunique"), matches_batted=("match_id", "nunique"),
    ).rename(columns={"batter": "player"})
    dismissals = ball[ball["player_out"].notna() & (ball["player_out"].astype(str).str.strip() != "")].groupby(["season", "player_out"]).size().rename("dismissals").reset_index().rename(columns={"player_out": "player"})
    bat = bat.merge(dismissals, on=["season", "player"], how="left")
    bat["dismissals"] = bat.dismissals.fillna(0)
    bowl = ball.groupby(["season", "bowler"], as_index=False).agg(
        runs_conceded=("bowler_runs_conceded", "sum"), legal_balls=("legal_ball", "sum"), wickets=("bowler_wicket", "sum"),
        bowling_matches=("match_id", "nunique"),
    ).rename(columns={"bowler": "player"})
    stats = bat.merge(bowl, on=["season", "player"], how="outer")
    count_cols = ["runs", "balls_faced", "fours", "sixes", "batting_innings", "matches_batted", "dismissals", "runs_conceded", "legal_balls", "wickets", "bowling_matches"]
    stats[count_cols] = stats[count_cols].fillna(0)
    stats["batting_average"] = np.where(stats.dismissals > 0, stats.runs / stats.dismissals, np.nan)
    stats["strike_rate"] = np.where(stats.balls_faced > 0, stats.runs * 100 / stats.balls_faced, np.nan)
    stats["bowling_average"] = np.where(stats.wickets > 0, stats.runs_conceded / stats.wickets, np.nan)
    stats["economy"] = np.where(stats.legal_balls > 0, stats.runs_conceded * 6 / stats.legal_balls, np.nan)
    stats["overs_bowled"] = stats.legal_balls / 6
    stats["total_matches"] = stats[["matches_batted", "bowling_matches"]].max(axis=1)
    return stats.sort_values(["season", "runs"], ascending=[True, False]).reset_index(drop=True)


def historical_squad_stats(ball: pd.DataFrame) -> pd.DataFrame:
    """Build historical team-player pools from observed participation (2021–2026).

    Team membership is inferred from the batting side and the opposing side in each
    match. Playing role is only a batting/bowling participation heuristic; keeper
    status and overseas eligibility are not present in the source data.
    """
    required = {"match_id", "season", "team1", "team2", "batting_team", "batter", "bowler"}
    if not required.issubset(ball.columns):
        return pd.DataFrame(columns=["player_name", "team", "role", "is_overseas", "is_available",
                                     "batting_style", "bowling_style", "matches", "runs", "wickets"])

    frame = ball[ball.season.between(2021, 2026)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["player_name", "team", "role", "is_overseas", "is_available",
                                     "batting_style", "bowling_style", "matches", "runs", "wickets"])

    batting = frame[frame.batter.notna() & frame.batting_team.notna()].groupby(
        ["batting_team", "batter"], as_index=False
    ).agg(runs=("batter_runs", "sum"), batting_matches=("match_id", "nunique"))
    batting = batting.rename(columns={"batting_team": "team", "batter": "player_name"})

    bowling = frame[frame.bowler.notna() & frame.batting_team.notna()].copy()
    bowling["team"] = np.where(bowling.batting_team == bowling.team1, bowling.team2,
                                np.where(bowling.batting_team == bowling.team2, bowling.team1, pd.NA))
    bowling = bowling[bowling.team.notna()].groupby(["team", "bowler"], as_index=False).agg(
        legal_balls=("legal_ball", "sum"), wickets=("bowler_wicket", "sum"),
        bowling_matches=("match_id", "nunique"),
    ).rename(columns={"bowler": "player_name"})

    pool = batting.merge(bowling, on=["team", "player_name"], how="outer")
    for column in ("runs", "batting_matches", "legal_balls", "wickets", "bowling_matches"):
        pool[column] = pool[column].fillna(0)
    has_batted = pool.batting_matches.gt(0)
    has_bowled = pool.legal_balls.gt(0)
    pool["role"] = np.select(
        [has_batted & has_bowled & pool.legal_balls.ge(30), has_batted, has_bowled],
        ["All-rounder", "Batter", "Bowler"], default="Unknown",
    )
    pool["matches"] = pool[["batting_matches", "bowling_matches"]].max(axis=1)
    # Unknowns remain missing rather than asserting these attributes from performance data.
    pool["is_overseas"] = pd.NA
    pool["is_available"] = True
    pool["batting_style"] = pd.NA
    pool["bowling_style"] = pd.NA
    return pool[["player_name", "team", "role", "is_overseas", "is_available", "batting_style",
                 "bowling_style", "matches", "runs", "wickets"]].sort_values(
                     ["team", "player_name"], kind="stable").reset_index(drop=True)


def phase_stats(ball: pd.DataFrame) -> pd.DataFrame:
    """Summarize innings phases using zero-indexed Cricsheet over numbers."""
    frame = ball.copy()
    frame["phase"] = np.select([frame["over"] < 6, frame["over"] >= 15], ["Powerplay", "Death"], default="Middle")
    return frame.groupby(["season", "batting_team", "phase"], as_index=False).agg(
        runs=("total_runs", "sum"), batter_runs=("batter_runs", "sum"), legal_balls=("legal_ball", "sum"),
        wickets=("is_wicket", "sum"), matches=("match_id", "nunique"),
    )


def team_season_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute season-level team matches, wins, and win rates; no-results excluded from decisions."""
    team_rows = pd.concat([
        matches[["season", "team1", "match_winner"]].rename(columns={"team1": "team"}),
        matches[["season", "team2", "match_winner"]].rename(columns={"team2": "team"}),
    ], ignore_index=True).dropna(subset=["team"])
    team_rows["decided"] = team_rows.match_winner.notna() & (team_rows.match_winner.astype(str).str.strip() != "")
    team_rows["win"] = team_rows.decided & (team_rows.team == team_rows.match_winner)
    return team_rows.groupby(["season", "team"], as_index=False).agg(matches=("team", "size"), decisions=("decided", "sum"), wins=("win", "sum")).assign(
        win_percentage=lambda df: np.where(df.decisions > 0, 100 * df.wins / df.decisions, np.nan)
    )
