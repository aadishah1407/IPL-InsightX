"""Validation and cricket-aware cleaning for the supplied Cricsheet-style CSVs."""
import json
import re

import pandas as pd

TEAM_ALIASES = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
}
BOWLER_WICKET_KINDS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
REQUIRED_BALL = ["match_id", "season", "innings", "over", "batter", "bowler", "batter_runs", "total_runs"]
REQUIRED_MATCH = ["match_id", "season", "team1", "team2"]


def canonical_team(value):
    if pd.isna(value):
        return value
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return TEAM_ALIASES.get(cleaned.casefold(), cleaned)


def _extras(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value) or not str(value).strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _wicket_set(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip().casefold() for part in re.split(r"[,|]", str(value)) if part.strip()}


def _require_columns(frame, required, label):
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"{label} dataset is missing required columns: {', '.join(absent)}")


def clean_data(ball: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize names/types and derive legal-ball and bowler-credit flags."""
    ball = ball.copy()
    matches = matches.copy()
    _require_columns(ball, REQUIRED_BALL, "Ball-by-ball")
    _require_columns(matches, REQUIRED_MATCH, "Match")
    for column, default in {
        "match_date": pd.NaT, "venue": "", "city": "", "team1": "", "team2": "",
        "batting_team": "", "toss_winner": "", "match_winner": "", "extras_json": "{}",
        "extras_total": 0, "is_wicket": 0, "wicket_kinds": "", "player_out": "",
        "delivery_in_over": pd.NA,
    }.items():
        if column not in ball.columns:
            ball[column] = default
    for column, default in {"match_date": pd.NaT, "team1": "", "team2": "", "match_winner": ""}.items():
        if column not in matches.columns:
            matches[column] = default
    for frame in (ball, matches):
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        frame["match_date"] = pd.to_datetime(frame["match_date"], dayfirst=True, errors="coerce") if "match_date" in frame else pd.NaT
        for col in ("team1", "team2", "batting_team", "toss_winner", "match_winner"):
            if col in frame:
                frame[col] = frame[col].map(canonical_team)
    for col in ("batter", "bowler", "non_striker", "player_out", "player_of_match"):
        if col in ball.columns:
            ball[col] = ball[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    if "player_of_match" in matches.columns:
        matches["player_of_match"] = matches["player_of_match"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    numeric = ["innings", "over", "delivery_in_over", "batter_runs", "total_runs", "extras_total", "is_wicket"]
    for col in numeric:
        if col in ball:
            ball[col] = pd.to_numeric(ball[col], errors="coerce")
    ball = ball.drop_duplicates().copy()
    ball = ball[ball["season"].notna() & ball["match_id"].notna() & ball["batter"].notna() & ball["bowler"].notna()]
    ball = ball[ball["batter_runs"].ge(0) & ball["total_runs"].ge(0)]
    ball["extras"] = ball.get("extras_json", pd.Series("{}", index=ball.index)).map(_extras)
    ball["is_wide"] = ball["extras"].map(lambda x: float(x.get("wides", 0) or 0) > 0)
    ball["is_no_ball"] = ball["extras"].map(lambda x: float(x.get("noballs", x.get("no_balls", 0)) or 0) > 0)
    ball["bowler_runs_conceded"] = ball.apply(
        lambda row: max(0, float(row["total_runs"]) - sum(float(row["extras"].get(key, 0) or 0) for key in ("byes", "legbyes", "penalty"))),
        axis=1,
    )
    # A batter's balls faced exclude wides; a bowler's legal balls exclude wides and no-balls.
    ball["batter_ball"] = ~ball["is_wide"]
    ball["legal_ball"] = ~(ball["is_wide"] | ball["is_no_ball"])
    ball["wicket_kind_set"] = ball.get("wicket_kinds", pd.Series("", index=ball.index)).map(_wicket_set)
    ball["bowler_wicket"] = ball["wicket_kind_set"].map(lambda kinds: bool(kinds & BOWLER_WICKET_KINDS))
    ball["season"] = ball["season"].astype(int)
    matches = matches.drop_duplicates(subset=["match_id"], keep="first")
    matches = matches[matches["season"].notna() & matches["match_id"].notna()].copy()
    matches["season"] = matches["season"].astype(int)
    return ball.reset_index(drop=True), matches.reset_index(drop=True)
