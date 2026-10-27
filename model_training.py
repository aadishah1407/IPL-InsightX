"""Leakage-aware next-season Random Forest training and evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

FEATURES = ["prev_runs", "prev_strike_rate", "prev_average", "prev_innings", "prev_matches", "prev_wickets", "runs_mean_3", "runs_std_3", "seasons_seen", "prev_runs_per_match"]


def build_next_season_dataset(season_stats: pd.DataFrame) -> pd.DataFrame:
    """Create only observed consecutive-season labels; all predictors end before target season."""
    stats = season_stats.copy().sort_values(["player", "season"])
    rows = []
    available_seasons = sorted(stats.season.unique())
    for target_year in available_seasons[1:]:
        prior = stats[stats.season < target_year]
        target_players = set(stats.loc[stats.season == target_year, "player"])
        # Historical features use the most recent season strictly before target; no target-season values leak in.
        for player, group in prior.groupby("player", sort=False):
            if player not in target_players:
                continue
            group = group.sort_values("season")
            last = group.iloc[-1]
            recent = group.tail(3).runs.astype(float)
            rows.append({
                "player": player, "target_season": int(target_year), "prev_runs": float(last.runs),
                "prev_strike_rate": last.strike_rate, "prev_average": last.batting_average,
                "prev_innings": float(last.batting_innings), "prev_matches": float(last.total_matches),
                "prev_wickets": float(last.wickets), "runs_mean_3": float(recent.mean()),
                "runs_std_3": float(recent.std(ddof=0)) if len(recent) > 1 else 0.0,
                "seasons_seen": float(len(group)), "prev_runs_per_match": float(last.runs / max(last.total_matches, 1)),
            })
    examples = pd.DataFrame(rows)
    if examples.empty:
        return examples
    targets = stats[["season", "player", "runs"]].rename(columns={"season": "target_season", "runs": "target_runs"})
    return examples.merge(targets, on=["target_season", "player"], how="inner")


def _model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RandomForestRegressor(
        n_estimators=350, min_samples_leaf=2, max_features=0.85, random_state=42, n_jobs=-1
    ))


def evaluate_and_fit(examples: pd.DataFrame) -> dict:
    """Chronologically hold out the latest target-season block, then fit a deployment model on all examples."""
    if len(examples) < 12 or examples.target_season.nunique() < 2:
        return {"model": None, "metrics": {}, "predictions": pd.DataFrame(), "examples": examples, "warning": "Too few player-season examples or target seasons for meaningful chronological evaluation."}
    sorted_data = examples.sort_values(["target_season", "player"]).reset_index(drop=True)
    # Hold out whole target seasons, keeping all players from a season in the same partition.
    seasons = sorted(sorted_data.target_season.unique())
    test_seasons = {seasons[-1]}
    test_mask = sorted_data.target_season.isin(test_seasons)
    train, test = sorted_data.loc[~test_mask], sorted_data.loc[test_mask]
    if train.empty or test.empty:
        return {"model": None, "metrics": {}, "predictions": pd.DataFrame(), "examples": examples, "warning": "Not enough chronological target-season blocks to split."}
    evaluation_model = _model().fit(train[FEATURES], train.target_runs)
    test = test.copy()
    test["rf_predicted_runs"] = np.maximum(0, evaluation_model.predict(test[FEATURES]))
    test["baseline_mean_runs"] = float(train.target_runs.mean())
    test["baseline_previous_runs"] = test.prev_runs.clip(lower=0)
    mae = mean_absolute_error(test.target_runs, test.rf_predicted_runs)
    rmse = mean_squared_error(test.target_runs, test.rf_predicted_runs) ** 0.5
    r2 = r2_score(test.target_runs, test.rf_predicted_runs) if len(test) >= 2 else np.nan
    metrics = {
        "train_samples": int(len(train)), "test_samples": int(len(test)),
        "train_fraction": float(len(train) / len(sorted_data)), "test_fraction": float(len(test) / len(sorted_data)),
        "train_target_seasons": sorted(train.target_season.unique().tolist()), "test_target_seasons": sorted(test.target_season.unique().tolist()),
        "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
        "baseline_mean_mae": float(mean_absolute_error(test.target_runs, test.baseline_mean_runs)),
        "baseline_previous_mae": float(mean_absolute_error(test.target_runs, test.baseline_previous_runs)),
    }
    # Final deployment model uses all eligible 2022-2026 transitions; held-out metrics above remain untouched.
    deployment_model = _model().fit(sorted_data[FEATURES], sorted_data.target_runs)
    warning = "Only a small number of target seasons are available; results are exploratory, not a guarantee."
    if len(test) < 25:
        warning += " The latest-season test cohort is small."
    return {"model": deployment_model, "evaluation_model": evaluation_model, "metrics": metrics, "predictions": test, "examples": sorted_data, "warning": warning}


def make_current_features(season_stats: pd.DataFrame) -> pd.DataFrame:
    """Build 2027 inference features through the latest historical season for every observed player."""
    rows = []
    for player, group in season_stats.sort_values("season").groupby("player", sort=False):
        group = group.sort_values("season")
        last = group.iloc[-1]
        recent = group.tail(3).runs.astype(float)
        rows.append({"player": player, "last_season": int(last.season), "prev_runs": float(last.runs),
            "prev_strike_rate": last.strike_rate, "prev_average": last.batting_average,
            "prev_innings": float(last.batting_innings), "prev_matches": float(last.total_matches),
            "prev_wickets": float(last.wickets), "runs_mean_3": float(recent.mean()),
            "runs_std_3": float(recent.std(ddof=0)) if len(recent) > 1 else 0.0,
            "seasons_seen": float(len(group)), "prev_runs_per_match": float(last.runs / max(last.total_matches, 1))})
    return pd.DataFrame(rows)
