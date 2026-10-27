"""Dataset discovery and loading utilities for the IPL analytics app."""
from pathlib import Path

import pandas as pd
from preprocessing import canonical_team

ROOT = Path(__file__).resolve().parent
BALL_CANDIDATES = [ROOT / "ipl_ball_by_ball_2021_2026.csv", ROOT / "data" / "ipl_ball_by_ball_2021_2026.csv"]
MATCH_CANDIDATES = [ROOT / "ipl_matches_2021_2026.csv", ROOT / "data" / "ipl_matches_2021_2026.csv"]
SQUAD_PATH = ROOT / "ipl_squads.csv"
OUTPUT_DIR = ROOT / "outputs"


def _find(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find dataset. Looked in: {', '.join(map(str, candidates))}")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the supplied source CSVs without assuming optional columns."""
    ball = pd.read_csv(_find(BALL_CANDIDATES), low_memory=False)
    matches = pd.read_csv(_find(MATCH_CANDIDATES), low_memory=False)
    return ball, matches


def load_squads() -> pd.DataFrame:
    """Load manually verified squad data; return an empty, correctly shaped table if absent."""
    columns = ["player_name", "team", "role", "is_overseas", "is_available", "batting_style", "bowling_style"]
    if not SQUAD_PATH.exists():
        return pd.DataFrame(columns=columns)
    squads = pd.read_csv(SQUAD_PATH).reindex(columns=columns)
    if "team" in squads:
        squads["team"] = squads["team"].map(canonical_team)
    if "player_name" in squads:
        squads["player_name"] = squads["player_name"].astype("string").str.strip()
    return squads
