"""Streamlit application: IPL Squad Selector & Player Matchup Analytics."""
import base64
from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import OUTPUT_DIR, SQUAD_PATH, load_raw_data, load_squads
from feature_engineering import historical_squad_stats, phase_stats, player_match_stats, player_season_stats, team_season_stats
from matchup_analysis import matchup_summary
from model_training import FEATURES, build_next_season_dataset, evaluate_and_fit, make_current_features
from playing_xi import recommend_xi
from preprocessing import clean_data
from team_analysis import batting_bowling_summary, head_to_head
from visualizations import actual_predicted_chart, season_wins_chart

st.set_page_config(page_title="IPL InsightX", page_icon="🏏", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#17232d; --muted:#6b7a86; --lime:#c8f169; --blue:#1264a3; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em; }
.stApp { background:linear-gradient(145deg,#f7fafc 0%,#eef4f3 58%,#f9fbf6 100%); }
.block-container { max-width:1480px; padding-top:1.5rem; }
[data-testid="stMetric"] { background:white; padding:1rem 1.2rem; border:1px solid #e4eaed; border-radius:16px; box-shadow:0 5px 16px #17232d0a; }
[data-testid="stSidebar"] { background:#14212a; }
[data-testid="stSidebar"] * { color:#f4f7f8 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#17232d !important; }
div[data-testid="stAlert"] { border-radius:14px; }
.hero { padding:2rem 2.2rem; border-radius:22px; color:#fff; background:linear-gradient(115deg,#12222e,#154c68 72%,#257a74); margin-bottom:1.2rem; }
.hero p { color:#d1e5e8; max-width:800px; }
.pill { display:inline-block; background:#c8f169; color:#1c3022; border-radius:99px; padding:.25rem .7rem; font-size:.78rem; font-weight:700; margin-bottom:.6rem; }
.caption-card { border:1px solid #e4eaed; background:#fff; border-radius:14px; padding:1rem 1.15rem; }
.logo-wrap { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.3rem; }
.logo-wrap img { width:36px; height:36px; border-radius:10px; }
.brand-image { width:100%; max-width:260px; height:auto; object-fit:contain; display:block; }
.brand-title { margin:0; font-size:3rem; line-height:1.05; font-weight:800; color:#17232d; }
.brand-row { display:flex; align-items:center; gap:14px; margin:0 0 10px 0; }
.brand-row img { width:180px; height:auto; max-height:180px; object-fit:contain; display:block; }
.brand-wordmark { font-size:3rem; line-height:0.9; font-weight:800; letter-spacing:-0.04em; color:#17232d; }
.brand-wordmark span { display:block; }
.sidebar-brand { display:flex; align-items:center; gap:12px; margin:0 0 12px 0; }
.sidebar-brand img { width:110px; height:auto; max-height:110px; object-fit:contain; display:block; }
.sidebar-brand .sidebar-wordmark { font-size:2.1rem; line-height:0.9; font-weight:800; letter-spacing:-0.04em; color:#f4f7f8; }
.sidebar-brand .sidebar-wordmark span { display:block; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading and validating the uploaded IPL files…")
def get_data():
    raw_ball, raw_matches = load_raw_data()
    return clean_data(raw_ball, raw_matches)

@st.cache_data(show_spinner="Building player, team and model summaries…")
def get_analytics():
    ball, matches = get_data()
    seasons = player_season_stats(ball)
    examples = build_next_season_dataset(seasons)
    model_result = evaluate_and_fit(examples)
    return seasons, examples, model_result, team_season_stats(matches), phase_stats(ball)

try:
    ball, matches = get_data()
    season_stats, examples, model_result, team_stats, phases = get_analytics()
    squad = load_squads()
    historical_squads = historical_squad_stats(ball)
except Exception as exc:
    st.error(f"Data could not be loaded: {exc}")
    st.info("Keep the two uploaded IPL CSV files beside app.py, then reload the app.")
    st.stop()

if not model_result.get("warning") and model_result.get("model") is None:
    model_result["warning"] = "Model is unavailable because there are not enough historical examples."

# Load recent individual summaries for an interpretable recommendation score.
current_features = make_current_features(season_stats)
if model_result.get("model") is not None and not current_features.empty:
    current_features["predicted_runs"] = np.maximum(0, model_result["model"].predict(current_features[FEATURES]))
else:
    current_features["predicted_runs"] = np.nan
forecasts = current_features.rename(columns={"predicted_runs": "predicted_runs"})
forecasts = forecasts[["player", "predicted_runs"]] if not forecasts.empty else pd.DataFrame(columns=["player", "predicted_runs"])

def get_logo_base64():
    logo_path = Path(__file__).resolve().parent / "assets" / "ipl_insightx_logo.png"
    return base64.b64encode(logo_path.read_bytes()).decode("utf-8")

logo_b64 = get_logo_base64()

pages = ["Home Dashboard", "IPL 2027 Player Predictions", "Best Playing XI Selector", "Team Performance Comparison", "Batter vs Bowler Analysis"]
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <img src="data:image/png;base64,{logo_b64}" alt="IPL InsightX logo" />
            <div class="sidebar-wordmark">IPL<br><span>InsightX</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Historical evidence • Transparent ML • 2027 scenarios")
    page = st.radio("Explore", pages, label_visibility="collapsed")
    st.divider()
    st.caption(f"{len(matches):,} matches · {len(ball):,} deliveries · seasons {matches.season.min()}–{matches.season.max()}")
    st.caption("Historical player pools are derived from 2021–2026 match participation.")

st.markdown(
    f"""
    <div class="brand-row">
        <img src="data:image/png;base64,{logo_b64}" alt="IPL InsightX logo" />
        <div class="brand-wordmark">IPL <span>InsightX</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def hero(eyebrow, title, intro):
    st.markdown(f'<section class="hero"><span class="pill">{eyebrow}</span><h1>{title}</h1><p>{intro}</p></section>', unsafe_allow_html=True)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def selected_seasons(key="season-filter"):
    years = sorted(matches.season.unique().tolist())
    return st.multiselect("Seasons", years, default=years, key=key)

if page == "Home Dashboard":
    hero("IPL 2027 • DECISION SUPPORT", "Build a smarter XI. Read the game behind the numbers.",
         "Explore player forecasts, team trends and batter–bowler history using the supplied 2021–2026 IPL datasets. Every recommendation is evidence-led—not a match guarantee.")
    wins = matches.match_winner.dropna()
    metric_row([("Matches", f"{len(matches):,}", "2021–2026"), ("Ball records", f"{len(ball):,}", "cleaned + validated"),
                ("Franchises", str(pd.unique(pd.concat([matches.team1, matches.team2])).size), "canonical team names"),
                ("2027 squad status", "Not verified", "upload a confirmed roster")])
    st.markdown("### The season at a glance")
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(season_wins_chart(team_stats), use_container_width=True)
    with right:
        latest = matches.season.max()
        leaders = team_stats[team_stats.season == latest].sort_values(["wins", "win_percentage"], ascending=False).head(6)
        st.markdown(f"#### {latest} wins table")
        st.dataframe(leaders[["team", "matches", "wins", "win_percentage"]].rename(columns={"team":"Team", "matches":"Matches", "wins":"Wins", "win_percentage":"Win %"}), hide_index=True, use_container_width=True)
        st.info("Win percentage uses decided matches as the denominator; abandoned/no-result matches do not count as losses.")
    st.markdown("### Explore the system")
    tiles = st.columns(3)
    for col, title, text in zip(tiles, ["Forecast", "Select", "Compare"], ["A chronological Random Forest run forecast with a held-out season.", "A role- and overseas-aware XI builder driven by your verified squad file.", "Season records, head-to-head results, match phases and actual matchup data."]):
        col.markdown(f'<div class="caption-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

elif page == "IPL 2027 Player Predictions":
    hero("PLAYER OUTLOOK", "IPL 2027 batting forecasts", "Expected runs for a player who participates. Forecasts are based on historical participation and performance; new players receive a model-median cold-start estimate where a verified squad roster is supplied.")
    if model_result.get("warning"):
        st.warning(model_result["warning"])
    if model_result.get("model") is None:
        st.error("Not enough player-season history to fit the forecasting model.")
    else:
        feature_frame = current_features.copy()
        if not squad.empty:
            squad_names = squad.player_name.dropna().astype(str).unique()
            missing_names = sorted(set(squad_names) - set(feature_frame.player))
            if missing_names:
                cold = pd.DataFrame({"player": missing_names})
                for col in FEATURES:
                    cold[col] = np.nan
                cold["last_season"] = np.nan
                cold["predicted_runs"] = np.maximum(0, model_result["model"].predict(cold[FEATURES]))
                feature_frame = pd.concat([feature_frame, cold], ignore_index=True)
                feature_frame["forecast_basis"] = np.where(feature_frame.last_season.isna(), "Cold-start model median (limited evidence)", "Historical player record")
            else:
                feature_frame["forecast_basis"] = "Historical player record"
        else:
            feature_frame["forecast_basis"] = "Historical player record"
        teams = ["All players"] + sorted(squad.team.dropna().unique().tolist()) if not squad.empty else ["All historical players (no verified roster)" ]
        choice = st.selectbox("Squad filter", teams)
        if not squad.empty and choice != "All players":
            names = squad.loc[squad.team == choice, "player_name"]
            display = feature_frame[feature_frame.player.isin(names)]
        else:
            display = feature_frame
        search = st.text_input("Search player")
        if search:
            display = display[display.player.str.contains(search, case=False, na=False)]
        display = display.sort_values("predicted_runs", ascending=False)
        prediction_table = display[["player", "predicted_runs", "last_season", "prev_runs", "prev_strike_rate", "prev_wickets", "forecast_basis"]].rename(columns={"player":"Player", "predicted_runs":"Expected runs", "last_season":"Last observed season", "prev_runs":"Last-season runs", "prev_strike_rate":"Last-season SR", "prev_wickets":"Last-season wickets", "forecast_basis":"Forecast basis"})
        st.dataframe(prediction_table.style.format({"Expected runs": "{:.0f}", "Last-season runs": "{:.0f}", "Last-season wickets": "{:.0f}"}), hide_index=True, use_container_width=True)
        st.caption("These are conditional run forecasts—not participation probabilities. Cold-start estimates are heavily uncertain and are clearly marked.")
        st.download_button("Download forecasts as CSV", display.to_csv(index=False).encode("utf-8"), "ipl_2027_player_forecasts.csv", "text/csv")

elif page == "Best Playing XI Selector":
    hero("HISTORICAL PARTICIPATION", "Best Playing XI selector", "Choose a franchise to see players recorded for that team during IPL 2021–2026. Selection scores use historical performance and inferred batting/bowling activity; this is not a verified 2027 squad.")
    if historical_squads.empty:
        st.warning("Could not derive historical team squads. Check that the ball-by-ball CSV contains team, batting-team, batter, and bowler columns.")
    else:
        team = st.selectbox("Franchise", sorted(historical_squads.team.dropna().unique()))
        historical_team = historical_squads[historical_squads.team == team].copy()
        st.caption(f"{len(historical_team)} distinct players recorded for {team} across 2021–2026. Historical team affiliation is inferred from match participation.")
        result = recommend_xi(historical_squads, season_stats, forecasts, team, require_wicketkeeper=False)
        st.warning(result["warning"])
        if "xi" in result:
            xi = result["xi"]
            metric_row([("Eligible squad", result["eligible_count"], "verified roster rows"), ("Selected", len(xi), "target: 11"),
                        ("Overseas status", "Unknown", "not in historical CSV"),
                        ("Constraints passed", f"{sum(result['checks'].values())}/{len(result['checks'])}", "see validation")])
            st.markdown("#### Recommended XI")
            st.dataframe(xi[["player_name", "role", "is_overseas", "score", "predicted_runs", "recent_runs", "wickets", "selection_reason"]], hide_index=True, use_container_width=True)
            st.markdown("#### Constraint checks")
            st.json(result["checks"])
            if not result["missed"].empty:
                st.markdown("#### Narrowly missed selection")
                st.dataframe(result["missed"][["player_name", "role", "score", "miss_reason"]], hide_index=True, use_container_width=True)
            with st.expander("Historical squad pool"):
                st.dataframe(historical_team[["player_name", "role", "matches", "runs", "wickets"]].sort_values("runs", ascending=False), hide_index=True, use_container_width=True)
            st.caption("Selection score: 42% predicted runs + 26% recent runs + 10% strike rate + 15% wickets + 7% economy adjustment, with role bonuses. Role categories are inferred from observed activity; wicketkeeper identity and the four-overseas rule cannot be validated from these files.")

elif page == "Team Performance Comparison":
    hero("FRANCHISE COMPARISON", "Compare the season stories", "Review wins, head-to-head results, batting/bowling summaries and phase performance. Match-level outcomes are counted once; ties/no-results are not treated as losses.")
    teams = sorted(pd.unique(pd.concat([matches.team1, matches.team2])).tolist())
    a, b, c = st.columns(3)
    team_a = a.selectbox("Team A", teams, index=0)
    team_b = b.selectbox("Team B", teams, index=min(1, len(teams)-1))
    years = c.multiselect("Seasons", sorted(matches.season.unique()), default=sorted(matches.season.unique()), key="team-years")
    filtered_matches = matches[matches.season.isin(years)]
    filtered_team = team_season_stats(filtered_matches)
    filtered_ball = ball[ball.season.isin(years)]
    season_rows = filtered_team[filtered_team.team.isin([team_a, team_b])]
    st.plotly_chart(season_wins_chart(season_rows), use_container_width=True)
    left, right = st.columns(2)
    for col, team in ((left, team_a), (right, team_b)):
        row = filtered_team[filtered_team.team == team]
        wins_count = int(row.wins.sum())
        played = int(row.decisions.sum())
        col.markdown(f"### {team}")
        col.metric("Wins / decisions", f"{wins_count} / {played}", f"{100*wins_count/played:.1f}%" if played else "no decided matches")
        summary = batting_bowling_summary(filtered_ball, team)
        col.write(f"**Batting:** {summary['runs_scored']:,} team runs · {summary['batting_strike_rate']:.1f} batter SR" if pd.notna(summary['batting_strike_rate']) else f"**Batting:** {summary['runs_scored']:,} team runs")
        col.write(f"**Bowling:** {summary['runs_conceded']:,} runs conceded · {summary['bowling_economy']:.2f} economy" if pd.notna(summary['bowling_economy']) else f"**Bowling:** {summary['runs_conceded']:,} conceded")
    h2h = head_to_head(filtered_matches, team_a, team_b)
    metric_row([("Head-to-head matches", h2h["matches"], f"{min(years) if years else '—'}–{max(years) if years else '—'}"), (f"{team_a} wins", h2h["team_a_wins"], "decided matches"), (f"{team_b} wins", h2h["team_b_wins"], "decided matches"), ("No result / other", h2h["no_result_or_other"], "ties included here")])
    st.markdown("#### Season table")
    st.dataframe(season_rows.sort_values(["season", "team"]), hide_index=True, use_container_width=True)
    st.markdown("#### Phase scoring")
    phase_filtered = phases[(phases.season.isin(years)) & (phases.batting_team.isin([team_a, team_b]))]
    st.plotly_chart(px.bar(phase_filtered, x="phase", y="runs", color="batting_team", barmode="group", title="Runs scored by phase"), use_container_width=True)
    st.caption("Phases use Cricsheet's zero-indexed overs: 0–5 powerplay, 6–14 middle, 15+ death. Historical team franchise labels are canonicalized (Bangalore → Bengaluru).")

elif page == "Batter vs Bowler Analysis":
    hero("HEAD-TO-HEAD", "Batter vs bowler", "Every number below is computed from the uploaded delivery log. Legal balls are used for the strike-rate denominator; wides are deliveries faced but not legal balls, and no-balls count as batter balls faced but not bowler legal balls.")
    batters = sorted(ball.batter.dropna().unique())
    bowlers = sorted(ball.bowler.dropna().unique())
    left, right = st.columns(2)
    batter = left.selectbox("Batter", batters, index=0)
    bowler = right.selectbox("Bowler", bowlers, index=0)
    years = st.multiselect("Season filter", sorted(ball.season.unique()), default=sorted(ball.season.unique()), key="matchup-years")
    inns = st.multiselect("Innings filter", sorted(ball.innings.dropna().unique()), default=sorted(ball.innings.dropna().unique()), key="matchup-innings")
    result = matchup_summary(ball, batter, bowler, years, inns)
    if not result.get("deliveries"):
        st.info("No recorded matchup exists for this selection in the chosen seasons/innings.")
    else:
        metric_row([("Batter runs", result["runs"], f"{result['matches']} matches"), ("Balls faced", result["balls_faced"], f"{result['deliveries']} deliveries · {result['legal_balls']} bowler-legal"),
                    ("Strike rate", f"{result['strike_rate']:.1f}" if pd.notna(result["strike_rate"]) else "—", "runs per legal ball × 100"),
                    ("Bowler dismissals", result["bowler_dismissals"], f"{result['fours']} fours · {result['sixes']} sixes")])
        if result["deliveries"] < 30 or result["matches"] < 3:
            st.warning("Small sample: fewer than 30 deliveries or three matches. Do not draw strong conclusions from this matchup alone.")
        data = result["data"].copy()
        by_season = data.groupby("season", as_index=False).agg(runs=("batter_runs","sum"), deliveries=("match_id","size"), legal_balls=("legal_ball","sum"), dismissals=("bowler_wicket","sum"))
        st.plotly_chart(px.bar(by_season, x="season", y="runs", hover_data=["deliveries","legal_balls","dismissals"], title="Runs by season"), use_container_width=True)
        st.dataframe(by_season, hide_index=True, use_container_width=True)

elif page == "Model Evaluation":
    hero("HONEST VALIDATION", "How the run model performed", "A chronological target-season holdout keeps every player observation from the same season together. The final deployment model is trained on all eligible historical transitions, separately from these evaluation numbers.")
    metrics = model_result.get("metrics", {})
    if model_result.get("warning"):
        st.warning(model_result["warning"])
    if metrics:
        metric_row([("Training samples", metrics["train_samples"], f"target seasons {metrics['train_target_seasons']}"), ("Held-out samples", metrics["test_samples"], f"target season {metrics['test_target_seasons']}"),
                    ("MAE", f"{metrics['mae']:.1f} runs", "lower is better"), ("RMSE", f"{metrics['rmse']:.1f} runs", f"R² {metrics['r2']:.3f}" if pd.notna(metrics["r2"]) else "R² unavailable")])
        st.info(f"Chronological split: {metrics['train_fraction']:.1%} train / {metrics['test_fraction']:.1%} test (whole target seasons retained). The latest-season block may differ from an exact 80:20 split.")
        st.plotly_chart(actual_predicted_chart(model_result["predictions"]), use_container_width=True)
        baseline = pd.DataFrame({"Baseline": ["Training mean", "Previous-season runs", "Random Forest"], "MAE (runs)": [metrics["baseline_mean_mae"], metrics["baseline_previous_mae"], metrics["mae"]]})
        st.plotly_chart(px.bar(baseline, x="Baseline", y="MAE (runs)", title="Held-out MAE comparison (lower is better)", color="Baseline"), use_container_width=True)
        st.dataframe(model_result["predictions"][["player", "target_season", "prev_runs", "target_runs", "rf_predicted_runs", "baseline_previous_runs"]].rename(columns={"player":"Player", "target_season":"Target season", "prev_runs":"Previous-season runs", "target_runs":"Actual runs", "rf_predicted_runs":"RF forecast", "baseline_previous_runs":"Previous-runs baseline"}), hide_index=True, use_container_width=True)
        rf_better = metrics["mae"] < metrics["baseline_previous_mae"]
        st.caption("Random Forest improves on the previous-season-runs baseline in this holdout." if rf_better else "Random Forest does not improve on the previous-season-runs baseline in this holdout; baseline may be preferable for this small historical sample.")
    else:
        st.warning("Evaluation metrics are unavailable because the player-season sample is too small.")
    st.markdown("#### Method notes")
    st.markdown("- Target: runs in season $t$; inputs are aggregated only through season $t-1$. Training labels include observed adjacent-season players only, so this does not model participation/retirement or transfers.\n- The held-out block is the latest eligible target season, with all players in that season kept together. This approximates the requested chronological 80:20 split; exact 80:20 is not claimed when season grouping prevents it.\n- Baselines: training-set mean and previous-season runs. R² is reported only where mathematically meaningful.\n- 2027 predictions are conditional forecasts, not guarantees. The six-season history is too short for strong claims of generalization.")

elif page == "Data and Squad Management":
    hero("DATA CONTROL", "Inspect, export and manage your inputs", "Review the real CSV schema and quality checks, download a blank 2027 roster template, and add only verified squad information. The starter squad file is intentionally empty.")
    st.markdown("#### Source files")
    metric_row([("Ball-by-ball rows", f"{len(ball):,}", f"{ball.match_id.nunique():,} distinct matches"), ("Match rows", f"{len(matches):,}", f"{matches.season.nunique()} seasons"),
                ("Player-season rows", f"{len(season_stats):,}", "aggregated"), ("Training examples", f"{len(examples):,}", "observed consecutive seasons")])
    left, right = st.columns(2)
    with left:
        st.markdown("**Ball-by-ball columns detected**")
        st.code(", ".join(ball.columns.astype(str)), language="text")
        st.dataframe(ball.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)
    with right:
        st.markdown("**Match columns detected**")
        st.code(", ".join(matches.columns.astype(str)), language="text")
        st.write("Seasons:", sorted(matches.season.unique().tolist()))
        st.write("Franchises:", sorted(pd.unique(pd.concat([matches.team1, matches.team2])).tolist()))
    st.markdown("#### Squad data entry")
    st.info("2027 squads are not verified in the supplied files. No official squad, transfer, role, or availability has been fabricated. Use the schema below and save the verified CSV as ipl_squads.csv beside app.py.")
    template_columns = ["player_name", "team", "role", "is_overseas", "is_available", "batting_style", "bowling_style"]
    st.dataframe(pd.DataFrame(columns=template_columns), use_container_width=True)
    st.download_button("Download empty 2027 squad template", pd.DataFrame(columns=template_columns).to_csv(index=False).encode(), "ipl_squads_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload completed squad CSV to validate", type=["csv"])
    if uploaded:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [column for column in template_columns if column not in incoming.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                incoming = incoming[template_columns]
                incoming["team"] = incoming.team.map(lambda value: value)
                dupes = incoming.player_name.duplicated().sum()
                st.success(f"Schema valid: {len(incoming)} roster rows; {dupes} duplicate player names.")
                st.dataframe(incoming, hide_index=True, use_container_width=True)
                if st.button("Save as ipl_squads.csv", type="primary"):
                    incoming.to_csv(SQUAD_PATH, index=False)
                    st.cache_data.clear()
                    st.success("Saved. Reload the app to use the verified roster.")
        except Exception as exc:
            st.error(f"Could not read uploaded roster: {exc}")
    st.markdown("#### Export cleaned datasets")
    if st.button("Build and save inspectable CSV outputs", type="primary"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        clean_ball_export = ball.drop(columns=["extras", "wicket_kind_set"], errors="ignore")
        clean_ball_export.to_csv(OUTPUT_DIR / "cleaned_ball_by_ball.csv", index=False)
        matches.to_csv(OUTPUT_DIR / "cleaned_matches.csv", index=False)
        player_match_stats(ball).to_csv(OUTPUT_DIR / "player_match_stats.csv", index=False)
        season_stats.to_csv(OUTPUT_DIR / "player_season_stats.csv", index=False)
        team_stats.to_csv(OUTPUT_DIR / "team_season_stats.csv", index=False)
        phases.to_csv(OUTPUT_DIR / "phase_stats.csv", index=False)
        examples.to_csv(OUTPUT_DIR / "next_season_training_examples.csv", index=False)
        st.success(f"Saved cleaned and aggregated CSVs under {OUTPUT_DIR.name}/.")
    st.caption("Cleaning normalizes franchise aliases, parses date/season fields, drops exact duplicate records and invalid negative scores, and derives legal-ball flags. Cricsheet over numbers remain zero-indexed; powerplay = 0–5 and death = 15+.")
