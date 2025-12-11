import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

df = pd.read_csv("../Motor_Vehicle_Collisions_-_Crashes_20251208.csv", low_memory=False)

df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"], errors="coerce")
df["CRASH TIME"] = pd.to_datetime(df["CRASH TIME"], format="%H:%M", errors="coerce").dt.time

df["SeverityIndex"] = 5 * df["NUMBER OF PERSONS KILLED"].fillna(0) + df["NUMBER OF PERSONS INJURED"].fillna(0)

df = df.dropna(subset=["CRASH DATE", "CRASH TIME"])

df["year_month"] = df["CRASH DATE"].dt.to_period("M")
df["hour"] = pd.to_datetime(df["CRASH TIME"], format="%H:%M:%S", errors="coerce").dt.hour
df["day_of_week"] = df["CRASH DATE"].dt.day_name()

monthly_data = (
    df.groupby("year_month").agg(count=("SeverityIndex", "size"), severity=("SeverityIndex", "sum")).reset_index()
)
monthly_data["year_month_str"] = monthly_data["year_month"].astype(str)

hourly_data = df.groupby("hour").agg(count=("SeverityIndex", "size"), severity=("SeverityIndex", "sum")).reset_index()

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data = df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
heatmap_pivot = heatmap_data.pivot(index="day_of_week", columns="hour", values="count")
heatmap_pivot = heatmap_pivot.reindex(day_order)

fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        "Monthly Collision Trend by Severity",
        "Hourly Collision Count by Severity",
        "Collision Density by Hour and Day of Week",
    ),
    vertical_spacing=0.12,
    specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "heatmap"}]],
    row_heights=[0.33, 0.33, 0.34],
)

for i in range(len(monthly_data) - 1):
    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month_str"].iloc[i : i + 2],
            y=monthly_data["count"].iloc[i : i + 2],
            mode="lines",
            line=dict(
                color=px.colors.sample_colorscale(
                    "YlOrRd", monthly_data["severity"].iloc[i] / monthly_data["severity"].max()
                )[0],
                width=2,
            ),
            showlegend=False,
            hovertemplate=f"<b>Month:</b> {monthly_data['year_month_str'].iloc[i]}<br><b>Collisions:</b> {monthly_data['count'].iloc[i]}<br><b>Severity:</b> {monthly_data['severity'].iloc[i]:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="YlOrRd",
            showscale=True,
            cmin=monthly_data["severity"].min(),
            cmax=monthly_data["severity"].max(),
            colorbar=dict(title="Severity", x=1.02, y=0.85, len=0.25),
        ),
        hoverinfo="none",
        showlegend=False,
    ),
    row=1,
    col=1,
)

for i in range(len(hourly_data) - 1):
    fig.add_trace(
        go.Scatter(
            x=hourly_data["hour"].iloc[i : i + 2],
            y=hourly_data["count"].iloc[i : i + 2],
            mode="lines",
            line=dict(
                color=px.colors.sample_colorscale(
                    "YlOrRd", hourly_data["severity"].iloc[i] / hourly_data["severity"].max()
                )[0],
                width=2,
            ),
            showlegend=False,
            hovertemplate=f"<b>Hour:</b> {hourly_data['hour'].iloc[i]}<br><b>Collisions:</b> {hourly_data['count'].iloc[i]}<br><b>Severity:</b> {hourly_data['severity'].iloc[i]:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="YlOrRd",
            showscale=True,
            cmin=hourly_data["severity"].min(),
            cmax=hourly_data["severity"].max(),
            colorbar=dict(title="Severity", x=1.02, y=0.5, len=0.25),
        ),
        hoverinfo="none",
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale="YlOrRd",
        colorbar=dict(title="Collisions", x=1.02, y=0.15, len=0.25),
    ),
    row=3,
    col=1,
)

fig.update_xaxes(title_text="Month", tickangle=45, row=1, col=1)
fig.update_yaxes(title_text="Number of Collisions", row=1, col=1)

fig.update_xaxes(title_text="Hour of Day (0-23)", tickmode="linear", tick0=0, dtick=1, row=2, col=1)
fig.update_yaxes(title_text="Number of Collisions", row=2, col=1)

fig.update_xaxes(title_text="Hour of Day (0-23)", tickmode="linear", tick0=0, dtick=1, row=3, col=1)
fig.update_yaxes(title_text="Day of Week", row=3, col=1)

fig.update_layout(
    title_text="NYC Motor Vehicle Collisions - Temporal Analysis",
    title_font_size=24,
    title_x=0.5,
    title_xanchor="center",
    height=1800,
    showlegend=False,
)

fig.write_html("temporal_analysis_dashboard.html")
print("Generated: temporal_analysis_dashboard.html")

print("\nSummary Statistics:")
print(f"Total collisions analyzed: {len(df):,}")
print(f"Date range: {df['CRASH DATE'].min()} to {df['CRASH DATE'].max()}")
print(f"Total SeverityIndex: {df['SeverityIndex'].sum():,.0f}")
print(f"Average SeverityIndex per collision: {df['SeverityIndex'].mean():.2f}")
print(f"\nPeak collision hour: {df['hour'].mode()[0]}:00")
print(f"Peak collision day: {df['day_of_week'].mode()[0]}")
