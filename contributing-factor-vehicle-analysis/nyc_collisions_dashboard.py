#!/usr/bin/env python3
"""
NYC Motor Vehicle Collisions — Plots + HTML Dashboard

Creates the following visuals (as Plotly interactive charts) and writes a single HTML dashboard:
1) Severity by Vehicle Type (stacked bar)
2) Severity by Contributing Factor (stacked bar)
3) Factor × Severity matrix (counts; heatmap-style table)
4) Vehicle Type × Severity matrix (counts; heatmap-style table)
5) Severity Index heatmap (vehicle type × factor), where:
   SeverityIndex = Injured + 5 * Killed

USAGE
-----
python nyc_collisions_dashboard.py --data /path/to/your_cleaned.csv --out dashboard.html

Notes:
- The script is robust to common NYC Open Data column name variants (with/without spaces, different casing).
- If your cleaned dataset already includes `vehicle_type_final` and `factor_final`, the script uses them.
- If not, it attempts a best-effort fallback to raw fields (Vehicle Type Code 1, Contributing Factor Vehicle 1).
"""

from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _norm(s: str) -> str:
    """Normalize column names for comparison."""
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """
    Return the first matching column name from `candidates` (case/space-insensitive).
    """
    norm_to_actual = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_to_actual:
            return norm_to_actual[key]
    if required:
        raise KeyError(f"Could not find any of these columns: {candidates}\nAvailable columns:\n{list(df.columns)[:50]} ...")
    return None


def ensure_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have numeric injured/killed and a severity_category.
    Severity logic:
      - Fatal if killed > 0
      - Injury if injured > 0
      - else PDO
    """
    killed_col = pick_col(df, [
        "NUMBER OF PERSONS KILLED", "Number Of Persons Killed", "number_of_persons_killed",
        "PERSONS_KILLED", "killed"
    ])
    injured_col = pick_col(df, [
        "NUMBER OF PERSONS INJURED", "Number Of Persons Injured", "number_of_persons_injured",
        "PERSONS_INJURED", "injured"
    ])

    df = df.copy()
    df[killed_col] = pd.to_numeric(df[killed_col], errors="coerce").fillna(0).astype(float)
    df[injured_col] = pd.to_numeric(df[injured_col], errors="coerce").fillna(0).astype(float)

    if pick_col(df, ["severity_category", "Severity Category"], required=False) is None:
        df["severity_category"] = np.select(
            [df[killed_col] > 0, df[injured_col] > 0],
            ["Fatal", "Injury"],
            default="Property Damage Only"
        )
    else:
        # Normalize existing field values a bit
        sev_col = pick_col(df, ["severity_category", "Severity Category"], required=False)
        df["severity_category"] = df[sev_col].astype(str).str.strip()

    df["SeverityIndex"] = df[injured_col] + 5.0 * df[killed_col]
    return df


def ensure_dims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have `vehicle_type_final` and `factor_final`.
    If not present, best-effort fallback to common raw fields.
    """
    df = df.copy()

    vt = pick_col(df, ["vehicle_type_final", "Vehicle Type Final", "vehicle_type"], required=False)
    if vt is None:
        vt_raw = pick_col(df, ["VEHICLE TYPE CODE 1", "Vehicle Type Code 1", "vehicle_type_code_1"], required=False)
        if vt_raw is None:
            raise KeyError("Could not find `vehicle_type_final` (preferred) or a common raw vehicle type field.")
        df["vehicle_type_final"] = df[vt_raw].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    else:
        df["vehicle_type_final"] = df[vt].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    fc = pick_col(df, ["factor_final", "Factor Final", "contributing_factor_final"], required=False)
    if fc is None:
        fc_raw = pick_col(df, ["CONTRIBUTING FACTOR VEHICLE 1", "Contributing Factor Vehicle 1", "contributing_factor_vehicle_1"], required=False)
        if fc_raw is None:
            raise KeyError("Could not find `factor_final` (preferred) or a common raw contributing factor field.")
        df["factor_final"] = df[fc_raw].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    else:
        df["factor_final"] = df[fc].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    # Drop obvious blanks
    df.loc[df["vehicle_type_final"].isin(["", "unknown", "unspecified"]), "vehicle_type_final"] = np.nan
    df.loc[df["factor_final"].isin(["", "unspecified", "unknown"]), "factor_final"] = np.nan
    return df


def top_n_by_measure(df: pd.DataFrame, dim: str, measure: str, n: int) -> list[str]:
    s = df.groupby(dim, dropna=True)[measure].sum().sort_values(ascending=False)
    return s.head(n).index.tolist()


def build_figures(df: pd.DataFrame,
                  top_vehicle_n: int = 12,
                  top_factor_n: int = 15,
                  top_heatmap_vehicle_n: int = 10,
                  top_heatmap_factor_n: int = 12) -> dict[str, go.Figure]:
    """
    Build all requested figures as Plotly Figures.
    """
    figs: dict[str, go.Figure] = {}

    # ---- Viz 1: Severity by Vehicle Type (stacked bar) ----
    top_vehicles = top_n_by_measure(df, "vehicle_type_final", "SeverityIndex", top_vehicle_n)
    v1 = (df[df["vehicle_type_final"].isin(top_vehicles)]
          .groupby(["vehicle_type_final", "severity_category"], dropna=False)
          .size()
          .reset_index(name="crash_count"))

    fig1 = px.bar(
        v1, x="vehicle_type_final", y="crash_count", color="severity_category",
        barmode="stack", title="Crash Severity by Vehicle Type (Crash Count)",
        labels={"vehicle_type_final": "Vehicle Type", "crash_count": "Crash Count", "severity_category": "Severity"}
    )
    fig1.update_layout(legend_title_text="Severity", xaxis_tickangle=-25)
    figs["vehicle_severity_bar"] = fig1

    # ---- Viz 2: Severity by Contributing Factor (stacked bar) ----
    top_factors = top_n_by_measure(df, "factor_final", "SeverityIndex", top_factor_n)
    v2 = (df[df["factor_final"].isin(top_factors)]
          .groupby(["factor_final", "severity_category"], dropna=False)
          .size()
          .reset_index(name="crash_count"))

    fig2 = px.bar(
        v2, y="factor_final", x="crash_count", color="severity_category",
        barmode="stack", orientation="h", title="Crash Severity by Contributing Factor (Crash Count)",
        labels={"factor_final": "Contributing Factor", "crash_count": "Crash Count", "severity_category": "Severity"}
    )
    fig2.update_layout(legend_title_text="Severity", yaxis={'categoryorder':'total ascending'})
    figs["factor_severity_bar"] = fig2

    # ---- Viz 3: Factor × Severity matrix (counts) ----
    pivot_factor = (df[df["factor_final"].isin(top_factors)]
                    .pivot_table(index="factor_final", columns="severity_category", values="SeverityIndex",
                                 aggfunc="size", fill_value=0))
    # Ensure consistent column order if present
    col_order = [c for c in ["Fatal", "Injury", "Property Damage Only"] if c in pivot_factor.columns]
    pivot_factor = pivot_factor[col_order]

    fig3 = px.imshow(
        pivot_factor.values,
        x=pivot_factor.columns.tolist(),
        y=pivot_factor.index.tolist(),
        text_auto=True,
        aspect="auto",
        title="Factor × Severity Matrix (Crash Count)",
        labels=dict(x="Severity", y="Contributing Factor", color="Crash Count"),
    )
    figs["factor_severity_matrix"] = fig3

    # ---- Viz 4: Vehicle × Severity matrix (counts) ----
    pivot_vehicle = (df[df["vehicle_type_final"].isin(top_vehicles)]
                     .pivot_table(index="vehicle_type_final", columns="severity_category", values="SeverityIndex",
                                  aggfunc="size", fill_value=0))
    col_order = [c for c in ["Fatal", "Injury", "Property Damage Only"] if c in pivot_vehicle.columns]
    pivot_vehicle = pivot_vehicle[col_order]

    fig4 = px.imshow(
        pivot_vehicle.values,
        x=pivot_vehicle.columns.tolist(),
        y=pivot_vehicle.index.tolist(),
        text_auto=True,
        aspect="auto",
        title="Vehicle Type × Severity Matrix (Crash Count)",
        labels=dict(x="Severity", y="Vehicle Type", color="Crash Count"),
    )
    figs["vehicle_severity_matrix"] = fig4

    # ---- Viz 5: Severity Index heatmap (vehicle × factor) ----
    top_hm_vehicles = top_n_by_measure(df, "vehicle_type_final", "SeverityIndex", top_heatmap_vehicle_n)
    top_hm_factors = top_n_by_measure(df, "factor_final", "SeverityIndex", top_heatmap_factor_n)

    hm = (df[df["vehicle_type_final"].isin(top_hm_vehicles) & df["factor_final"].isin(top_hm_factors)]
          .pivot_table(index="vehicle_type_final", columns="factor_final", values="SeverityIndex",
                       aggfunc="sum", fill_value=0))

    fig5 = px.imshow(
        hm.values,
        x=hm.columns.tolist(),
        y=hm.index.tolist(),
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Severity Index Heatmap (Injured + 5×Killed)",
        labels=dict(x="Contributing Factor", y="Vehicle Type", color="Severity Index"),
    )
    # Put numbers on hover (and optionally on cells via text_auto, but that can get crowded)
    figs["severity_index_heatmap"] = fig5

    return figs


def write_dashboard_html(figs: dict[str, go.Figure], out_path: Path, title: str = "NYC Collisions Dashboard") -> None:
    """
    Write a single HTML page with all figures.
    """
    # Build a nice 3-row layout using subplots (2+2+1 wide)
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap", "colspan": 2}, None]],
        subplot_titles=(
            "Crash Severity by Vehicle Type (Count)",
            "Crash Severity by Contributing Factor (Count)",
            "Factor × Severity Matrix (Count)",
            "Vehicle × Severity Matrix (Count)",
            "Severity Index Heatmap (Injured + 5×Killed)"
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.07,
    )

    # Add traces from each figure
    for tr in figs["vehicle_severity_bar"].data:
        fig.add_trace(tr, row=1, col=1)
    for tr in figs["factor_severity_bar"].data:
        fig.add_trace(tr, row=1, col=2)
    for tr in figs["factor_severity_matrix"].data:
        fig.add_trace(tr, row=2, col=1)
    for tr in figs["vehicle_severity_matrix"].data:
        fig.add_trace(tr, row=2, col=2)
    for tr in figs["severity_index_heatmap"].data:
        fig.add_trace(tr, row=3, col=1)

    # Update layout
    fig.update_layout(
        height=1200,
        title_text=title,
        showlegend=True,
        legend_title_text="Severity (bars)",
        margin=dict(l=40, r=30, t=80, b=30),
    )
    # Make bar chart axes labels nicer
    fig.update_xaxes(tickangle=-25, row=1, col=1)

    # Write HTML
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned (or raw) NYC collisions CSV")
    parser.add_argument("--out", default="dashboard.html", help="Output HTML dashboard path")
    parser.add_argument("--top_vehicle_n", type=int, default=12, help="Top N vehicle types to show")
    parser.add_argument("--top_factor_n", type=int, default=15, help="Top N factors to show")
    parser.add_argument("--top_heatmap_vehicle_n", type=int, default=10, help="Top N vehicles in SeverityIndex heatmap")
    parser.add_argument("--top_heatmap_factor_n", type=int, default=12, help="Top N factors in SeverityIndex heatmap")
    parser.add_argument("--title", default="NYC Motor Vehicle Collisions — Severity Dashboard", help="Dashboard title")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(data_path, low_memory=False)
    df = ensure_severity(df)
    df = ensure_dims(df)

    # Drop rows missing key dims
    df = df.dropna(subset=["vehicle_type_final", "factor_final", "severity_category"])

    figs = build_figures(
        df,
        top_vehicle_n=args.top_vehicle_n,
        top_factor_n=args.top_factor_n,
        top_heatmap_vehicle_n=args.top_heatmap_vehicle_n,
        top_heatmap_factor_n=args.top_heatmap_factor_n,
    )

    out_path = Path(args.out)
    write_dashboard_html(figs, out_path, title=args.title)

    print(f"Wrote dashboard: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
