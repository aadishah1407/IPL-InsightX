"""Shared Plotly visualizations."""
import plotly.express as px


def season_wins_chart(frame):
    return px.line(frame, x="season", y="win_percentage", color="team", markers=True, title="Season-wise win percentage")


def actual_predicted_chart(frame):
    chart = px.scatter(frame, x="target_runs", y="rf_predicted_runs", color="target_season", hover_name="player", title="Held-out actual vs Random Forest forecast")
    maximum = max(float(frame.target_runs.max()), float(frame.rf_predicted_runs.max()), 1)
    chart.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line={"dash": "dash", "color": "gray"})
    return chart
