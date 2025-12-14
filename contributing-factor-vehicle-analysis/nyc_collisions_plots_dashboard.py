#!/usr/bin/env python3
"""
NYC Collisions (cleaned CSV) -> interactive plots + one HTML dashboard.

Default data path (your Mac external drive path):
/Volumes/T7/MAC/COLLEGE/vizualization/NYC_Collisions_Project/Motor_Vehicle_Collisions_clean_for_tableau.csv

Run:
python nyc_collisions_plots_dashboard.py
or:
python nyc_collisions_plots_dashboard.py --data "/path/to/cleaned.csv" --out "dashboard.html"
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_DATA = r"/Volumes/T7/MAC/COLLEGE/vizualization/NYC_Collisions_Project/Motor_Vehicle_Collisions_clean_for_tableau.csv"


def heatmap_from_pivot(pivot: pd.DataFrame, title: str, colorscale=None, showscale=True):
    """Create a Plotly Heatmap from a pivot table (no extra features)."""
    z = pivot.values
    x = pivot.columns.tolist()
    y = pivot.index.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale or "Blues",
            showscale=showscale,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to cleaned CSV")
    p.add_argument("--out", default="NYC_Collisions_Dashboard.html", help="Output HTML path")
    p.add_argument("--top_vehicle_n", type=int, default=12, help="Top N vehicles to show")
    p.add_argument("--top_factor_n", type=int, default=15, help="Top N factors to show")
    p.add_argument("--top_heatmap_vehicle_n", type=int, default=10, help="Top N vehicles in SeverityIndex heatmap")
    p.add_argument("--top_heatmap_factor_n", type=int, default=12, help="Top N factors in SeverityIndex heatmap")
    args = p.parse_args()

    df = pd.read_csv(args.data, low_memory=False)

    # Your cleaned columns (expected):
    # vehicle_type_final, factor_final, severity_category, NUMBER OF PERSONS INJURED, NUMBER OF PERSONS KILLED
    df["NUMBER OF PERSONS INJURED"] = pd.to_numeric(df["NUMBER OF PERSONS INJURED"], errors="coerce").fillna(0)
    df["NUMBER OF PERSONS KILLED"] = pd.to_numeric(df["NUMBER OF PERSONS KILLED"], errors="coerce").fillna(0)
    df["SeverityIndex"] = df["NUMBER OF PERSONS INJURED"] + 5 * df["NUMBER OF PERSONS KILLED"]

    df = df.dropna(subset=["vehicle_type_final", "factor_final", "severity_category"])

    # --------- Top lists (based on SeverityIndex) ---------
    top_vehicles = (
        df.groupby("vehicle_type_final")["SeverityIndex"].sum()
        .sort_values(ascending=False)
        .head(args.top_vehicle_n)
        .index.tolist()
    )
    top_factors = (
        df.groupby("factor_final")["SeverityIndex"].sum()
        .sort_values(ascending=False)
        .head(args.top_factor_n)
        .index.tolist()
    )

    # --------- Viz 1: Vehicle × Severity (stacked bar) ---------
    v1 = (
        df[df["vehicle_type_final"].isin(top_vehicles)]
        .groupby(["vehicle_type_final", "severity_category"])
        .size()
        .reset_index(name="crash_count")
    )
    fig1 = px.bar(
        v1, x="vehicle_type_final", y="crash_count", color="severity_category",
        barmode="stack", title="Crash Severity by Vehicle Type (Crash Count)",
        labels={"vehicle_type_final": "Vehicle Type", "crash_count": "Crash Count", "severity_category": "Severity"}
    )
    fig1.update_layout(xaxis_tickangle=-25, legend_title_text="Severity")

    # --------- Viz 2: Factor × Severity (stacked bar) ---------
    v2 = (
        df[df["factor_final"].isin(top_factors)]
        .groupby(["factor_final", "severity_category"])
        .size()
        .reset_index(name="crash_count")
    )
    fig2 = px.bar(
        v2, y="factor_final", x="crash_count", color="severity_category",
        barmode="stack", orientation="h", title="Crash Severity by Contributing Factor (Crash Count)",
        labels={"factor_final": "Contributing Factor", "crash_count": "Crash Count", "severity_category": "Severity"}
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title_text="Severity")

    # --------- Viz 3: Factor × Severity matrix (counts) ---------
    pivot_factor = (
        df[df["factor_final"].isin(top_factors)]
        .pivot_table(index="factor_final", columns="severity_category", values="SeverityIndex", aggfunc="size", fill_value=0)
    )
    col_order = [c for c in ["Fatal", "Injury", "Property Damage Only"] if c in pivot_factor.columns]
    pivot_factor = pivot_factor[col_order]
    fig3 = heatmap_from_pivot(pivot_factor, "Factor × Severity Matrix (Crash Count)", colorscale="Greys", showscale=True)

    # --------- Viz 4: Vehicle × Severity matrix (counts) ---------
    pivot_vehicle = (
        df[df["vehicle_type_final"].isin(top_vehicles)]
        .pivot_table(index="vehicle_type_final", columns="severity_category", values="SeverityIndex", aggfunc="size", fill_value=0)
    )
    col_order = [c for c in ["Fatal", "Injury", "Property Damage Only"] if c in pivot_vehicle.columns]
    pivot_vehicle = pivot_vehicle[col_order]
    fig4 = heatmap_from_pivot(pivot_vehicle, "Vehicle Type × Severity Matrix (Crash Count)", colorscale="Greys", showscale=True)

    # --------- Viz 5: Severity Index heatmap (vehicle × factor; SUM) ---------
    top_hm_vehicles = (
        df.groupby("vehicle_type_final")["SeverityIndex"].sum()
        .sort_values(ascending=False)
        .head(args.top_heatmap_vehicle_n)
        .index.tolist()
    )
    top_hm_factors = (
        df.groupby("factor_final")["SeverityIndex"].sum()
        .sort_values(ascending=False)
        .head(args.top_heatmap_factor_n)
        .index.tolist()
    )
    hm = (
        df[df["vehicle_type_final"].isin(top_hm_vehicles) & df["factor_final"].isin(top_hm_factors)]
        .pivot_table(index="vehicle_type_final", columns="factor_final", values="SeverityIndex", aggfunc="sum", fill_value=0)
    )
    fig5 = heatmap_from_pivot(hm, "Severity Index Heatmap (Sum of Injured + 5×Killed)", colorscale="YlOrRd", showscale=True)

    # --------- One-page dashboard ---------
    dash = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap", "colspan": 2}, None]],
        subplot_titles=(
            "Crash Severity by Vehicle Type (Count)",
            "Crash Severity by Contributing Factor (Count)",
            "Factor × Severity Matrix (Count)",
            "Vehicle × Severity Matrix (Count)",
            "Severity Index Heatmap (Sum)"
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.07,
    )

    for tr in fig1.data: dash.add_trace(tr, row=1, col=1)
    for tr in fig2.data: dash.add_trace(tr, row=1, col=2)
    for tr in fig3.data: dash.add_trace(tr, row=2, col=1)
    for tr in fig4.data: dash.add_trace(tr, row=2, col=2)
    for tr in fig5.data: dash.add_trace(tr, row=3, col=1)

    dash.update_layout(
        title_text="NYC Motor Vehicle Collisions — Severity Dashboard",
        height=1200,
        margin=dict(l=40, r=30, t=80, b=30),
        showlegend=True,
        legend_title_text="Severity (bars)",
    )
    dash.update_xaxes(tickangle=-25, row=1, col=1)

    out_path = Path(args.out)
    dash.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
