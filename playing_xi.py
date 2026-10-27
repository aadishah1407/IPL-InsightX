"""Transparent score and role/overseas-aware XI recommendation."""
import numpy as np
import pandas as pd


def role_group(role):
    text = str(role or "").casefold()
    if any(key in text for key in ["wicket", "keeper", "wk"]):
        return "Wicketkeeper"
    if any(key in text for key in ["all-round", "allround", "all round"]):
        return "All-rounder"
    if any(key in text for key in ["bowler", "pace", "fast", "spin"]):
        return "Bowler"
    if any(key in text for key in ["batter", "batsman", "opener", "top order"]):
        return "Batter"
    return "Unknown"


def recommend_xi(squad: pd.DataFrame, season_stats: pd.DataFrame, forecasts: pd.DataFrame, team: str,
                 require_wicketkeeper: bool = True) -> dict:
    roster = squad[squad.team == team].copy()
    if roster.empty:
        return {"warning": "No historical player participation is available for this franchise."}
    if "overseas_known" not in roster:
        roster["overseas_known"] = roster.is_overseas.notna()
    # Keep the score inputs from aggregated player-season history, not duplicate
    # participation summary columns carried by a historical roster source.
    roster = roster.drop(columns=["runs", "wickets", "matches"], errors="ignore")
    roster["is_available"] = roster.is_available.astype(str).str.casefold().map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}).fillna(True)
    roster["is_overseas"] = roster.is_overseas.astype(str).str.casefold().map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}).fillna(False)
    roster = roster[roster.is_available].drop_duplicates("player_name")
    recent = season_stats.sort_values("season").groupby("player").tail(3).groupby("player").agg(
        recent_runs=("runs", "mean"), recent_sr=("strike_rate", "mean"), wickets=("wickets", "sum"), economy=("economy", "mean"), matches=("total_matches", "sum")).reset_index()
    pool = roster.merge(recent, left_on="player_name", right_on="player", how="left").drop(columns="player", errors="ignore")
    pool = pool.merge(forecasts[["player", "predicted_runs"]], left_on="player_name", right_on="player", how="left").drop(columns="player", errors="ignore")
    pool["role_group"] = pool.role.map(role_group)
    pool["predicted_runs"] = pool.predicted_runs.fillna(pool.recent_runs.fillna(0))
    pool["recent_runs"] = pool.recent_runs.fillna(0)
    pool["wickets"] = pool.wickets.fillna(0)
    pool["recent_sr"] = pool.recent_sr.fillna(0)
    pool["economy"] = pool.economy.fillna(10)
    # Independent, interpretable multi-skill score; capped prediction avoids runs dominating role value.
    pool["score"] = (0.42 * pool.predicted_runs.clip(0, 700) / 7 + 0.26 * pool.recent_runs.clip(0, 700) / 7 +
        0.10 * pool.recent_sr.clip(0, 220) / 2.2 + 0.15 * pool.wickets.clip(0, 30) * 2 +
        0.07 * (12 - pool.economy.clip(4, 12)))
    pool.loc[pool.role_group == "Wicketkeeper", "score"] += 3
    pool.loc[pool.role_group == "All-rounder", "score"] += 4
    pool.loc[pool.role_group == "Bowler", "score"] += 2
    pool.loc[pool.role_group == "Unknown", "score"] -= 3
    pool = pool.sort_values(["score", "player_name"], ascending=[False, True])
    selected = []
    # Meet role coverage first, then fill by score. A player can satisfy only one category in this transparent heuristic.
    role_requirements = [("Batter", 3), ("All-rounder", 1), ("Bowler", 4)]
    if require_wicketkeeper:
        role_requirements.insert(0, ("Wicketkeeper", 1))
    for group, minimum in role_requirements:
        for _, row in pool[pool.role_group == group].iterrows():
            if sum(item.player_name == row.player_name for item in selected) >= 1:
                continue
            if sum(item.is_overseas for item in selected) >= 4 and bool(row.is_overseas):
                continue
            if len(selected) >= 11:
                break
            selected.append(row)
            if sum(item.role_group == group for item in selected) >= minimum:
                break
    for _, row in pool.iterrows():
        if len(selected) >= 11:
            break
        if any(item.player_name == row.player_name for item in selected):
            continue
        if bool(row.is_overseas) and sum(item.is_overseas for item in selected) >= 4:
            continue
        selected.append(row)
    xi = pd.DataFrame(selected)
    checks = {"exactly_11": len(xi) == 11, "distinct": xi.player_name.nunique() == len(xi),
        "overseas_limit": int(xi.is_overseas.sum()) <= 4 if not xi.empty else True,
        "has_wicketkeeper": bool((xi.role_group == "Wicketkeeper").any()) or not require_wicketkeeper,
        "four_bowling_options": int(xi.role_group.isin(["Bowler", "All-rounder"]).sum()) >= 4}
    missed = pool[~pool.player_name.isin(xi.player_name)].head(8).copy()
    if not all(checks.values()):
        warning = "Squad information is insufficient to satisfy all XI constraints. This is a provisional partial recommendation; add verified roles/availability."
    elif not bool(roster.overseas_known.all()):
        warning = ("Historical participation-based XI for 2021–2026, not a 2027 roster. The source data does not identify "
                   "wicketkeepers, overseas eligibility, current contracts, or availability; role labels are inferred from "
                   "batting/bowling activity, so overseas and wicketkeeper constraints cannot be verified.")
    else:
        warning = "Provisional, data-driven recommendation only; historical participation does not verify a current squad."
    xi["score"] = np.rint(xi["score"]).astype(int)
    xi["predicted_runs"] = np.rint(xi["predicted_runs"]).astype(int)
    xi["recent_runs"] = np.rint(xi["recent_runs"]).astype(int)
    xi["selection_reason"] = xi.apply(lambda r: f"Score {r.score:.1f}; {r.role_group}; recent runs {r.recent_runs:.0f}, wickets {r.wickets:.0f}, 2027 runs forecast {r.predicted_runs:.0f}.", axis=1)
    missed["miss_reason"] = missed.apply(lambda r: "Ranked below selected players on the multi-skill score; role balance and four-overseas cap also apply.", axis=1)
    return {"xi": xi, "missed": missed, "checks": checks, "warning": warning, "eligible_count": len(pool)}
