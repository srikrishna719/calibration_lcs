"""Extract 2025 master data, map sensor CSVs by timestamp, and build trends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed_2025_trends"
MASTER_FILE = DATA_DIR / (
    "ambient-CAC-monitoring_automated_processing_v2_hourly_2025-09-06_"
    "analysis_master_000000000000.csv"
)

METRIC_MAP = {
    "PM2.5": ("pm2_5", "PM2.5"),
    "Temperature": ("temp_c", "Temperature"),
    "Humidity": ("rh", "Humidity"),
    "CO": ("co", "CO"),
    "Pressure": ("bp", "Pressure"),
}


def parse_sensor_timestamp(dataframe: pd.DataFrame) -> pd.Series:
    """Parse mixed date formats from sensor files.

    Jan/Feb files use DD-MM-YYYY, while the March file uses YYYY-MM-DD. This
    parser handles both formats row-by-row and floors to the hourly timestamp.
    """
    date_text = dataframe["date"].astype(str).str.strip()
    time_text = dataframe["time"].astype(str).str.strip()
    raw = date_text + " " + time_text
    year_first = date_text.str.match(r"^\d{4}[-/]")

    timestamp = pd.Series(pd.NaT, index=dataframe.index, dtype="datetime64[ns]")
    if year_first.any():
        timestamp.loc[year_first] = pd.to_datetime(raw.loc[year_first], errors="coerce")
    if (~year_first).any():
        timestamp.loc[~year_first] = pd.to_datetime(
            raw.loc[~year_first],
            errors="coerce",
            dayfirst=True,
        )
    return timestamp.dt.floor("h")


def numeric_columns(dataframe: pd.DataFrame, exclude: set[str]) -> list[str]:
    columns: list[str] = []
    for column in dataframe.columns:
        if column in exclude:
            continue
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        if converted.notna().any():
            dataframe[column] = converted
            columns.append(column)
    return columns


def hourly_mean(dataframe: pd.DataFrame, timestamp_col: str, exclude: set[str]) -> pd.DataFrame:
    numeric = numeric_columns(dataframe, exclude)
    if not numeric:
        return pd.DataFrame(columns=[timestamp_col])
    return (
        dataframe.groupby(timestamp_col, as_index=False)[numeric]
        .mean(numeric_only=True)
        .sort_values(timestamp_col)
    )


def save_plot(fig: go.Figure, html_path: Path, png_path: Path | None = None) -> str:
    fig.write_html(html_path, include_plotlyjs="cdn")
    if png_path is None:
        return "html"
    try:
        fig.write_image(png_path, width=1500, height=900, scale=2)
        return "html,png"
    except Exception as exc:
        print(f"PNG export skipped for {png_path.name}: {exc}")
        return "html"


def build_master_chart(master_hourly: pd.DataFrame) -> list[str]:
    metrics = [
        ("PM2.5", "pm2_5"),
        ("Temperature", "temp_c"),
        ("Humidity", "rh"),
        ("CO", "co"),
    ]
    metrics = [(label, column) for label, column in metrics if column in master_hourly and master_hourly[column].notna().any()]
    if not metrics:
        return []

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=[label for label, _ in metrics],
    )
    for row, (label, column) in enumerate(metrics, start=1):
        fig.add_trace(
            go.Scatter(
                x=master_hourly["timestamp"],
                y=master_hourly[column],
                mode="lines",
                name=f"Master mean {label}",
                line={"width": 1.6},
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=label, row=row, col=1)
    fig.update_layout(
        title="Master 2025 hourly trends (mean across sites)",
        height=max(420, 260 * len(metrics)),
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.08},
    )
    return [save_plot(fig, OUTPUT_DIR / "master_2025_trends.html", OUTPUT_DIR / "master_2025_trends.png")]


def build_common_chart(mapped_long: pd.DataFrame) -> list[str]:
    if mapped_long.empty:
        return []

    plot_metrics = [
        metric
        for metric in ["PM2.5", "Temperature", "Humidity", "CO", "Pressure"]
        if metric in set(mapped_long["metric"])
    ]
    if not plot_metrics:
        return []

    fig = make_subplots(
        rows=len(plot_metrics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=plot_metrics,
    )

    for row, metric in enumerate(plot_metrics, start=1):
        metric_df = mapped_long[mapped_long["metric"] == metric].copy()
        master = (
            metric_df.groupby("timestamp", as_index=False)["master_value"]
            .mean()
            .sort_values("timestamp")
        )
        fig.add_trace(
            go.Scatter(
                x=master["timestamp"],
                y=master["master_value"],
                mode="lines",
                name=f"Master {metric}",
                line={"color": "black", "width": 2.2},
            ),
            row=row,
            col=1,
        )
        for source_file, source_df in metric_df.groupby("source_file", sort=True):
            source_df = source_df.sort_values("timestamp")
            fig.add_trace(
                go.Scatter(
                    x=source_df["timestamp"],
                    y=source_df["sensor_value"],
                    mode="lines",
                    name=f"{source_file} {metric}",
                    line={"width": 1.4},
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=metric, row=row, col=1)

    fig.update_layout(
        title="2025 common timestamp trends: master vs sensor CSVs",
        height=max(520, 300 * len(plot_metrics)),
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.08},
    )
    return [save_plot(fig, OUTPUT_DIR / "common_2025_trends.html", OUTPUT_DIR / "common_2025_trends.png")]


def build_mapping_rows(
    master_hourly: pd.DataFrame,
    sensor_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(sensor_file)
    raw["timestamp"] = parse_sensor_timestamp(raw)
    parsed = raw[raw["timestamp"].notna()].copy()
    years = sorted(int(year) for year in parsed["timestamp"].dt.year.dropna().unique())
    sensor_2025 = parsed[parsed["timestamp"].dt.year == 2025].copy()

    summary: dict[str, Any] = {
        "source_file": sensor_file.name,
        "raw_rows": len(raw),
        "parsed_timestamp_rows": len(parsed),
        "years_present": ", ".join(str(year) for year in years),
        "sensor_2025_rows": len(sensor_2025),
        "sensor_2025_hourly_rows": 0,
        "common_timestamps": 0,
        "status": "no 2025 rows",
    }

    if sensor_2025.empty:
        return pd.DataFrame(), pd.DataFrame(), summary

    sensor_hourly = hourly_mean(
        sensor_2025,
        "timestamp",
        exclude={"timestamp", "date", "time", "devId", "db"},
    )
    if "devId" in sensor_2025.columns:
        device_counts = (
            sensor_2025.groupby("timestamp", as_index=False)["devId"]
            .nunique()
            .rename(columns={"devId": "device_count"})
        )
        sensor_hourly = sensor_hourly.merge(device_counts, on="timestamp", how="left")
    sensor_hourly["source_file"] = sensor_file.name
    summary["sensor_2025_hourly_rows"] = len(sensor_hourly)

    joined = master_hourly.merge(
        sensor_hourly,
        on="timestamp",
        how="inner",
        suffixes=("_master", "_sensor"),
    )
    summary["common_timestamps"] = len(joined)
    if joined.empty:
        summary["status"] = "2025 rows but no exact timestamp overlap with master"
        return pd.DataFrame(), pd.DataFrame(), summary

    wide = pd.DataFrame(
        {
            "timestamp": joined["timestamp"],
            "source_file": sensor_file.name,
        }
    )
    long_parts: list[pd.DataFrame] = []
    paired_metric_counts: dict[str, int] = {}
    for metric, (master_col, sensor_col) in METRIC_MAP.items():
        if master_col not in joined.columns or sensor_col not in joined.columns:
            continue
        master_series = joined[master_col]
        sensor_series = joined[sensor_col]
        pair_mask = master_series.notna() & sensor_series.notna()
        paired_count = int(pair_mask.sum())
        wide[f"master_{metric}"] = master_series
        wide[f"sensor_{metric}"] = sensor_series
        wide[f"{metric}_sensor_minus_master"] = sensor_series - master_series
        paired_metric_counts[metric] = paired_count
        if paired_count:
            long_parts.append(
                pd.DataFrame(
                    {
                        "timestamp": joined.loc[pair_mask, "timestamp"],
                        "source_file": sensor_file.name,
                        "metric": metric,
                        "master_value": master_series.loc[pair_mask],
                        "sensor_value": sensor_series.loc[pair_mask],
                        "sensor_minus_master": (sensor_series - master_series).loc[pair_mask],
                    }
                )
            )

    for metric, count in paired_metric_counts.items():
        summary[f"paired_{metric}_rows"] = count
    summary["status"] = "mapped" if any(paired_metric_counts.values()) else "overlap but no comparable non-null metrics"
    mapped_long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    return wide, mapped_long, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_FILE)
    master["timestamp"] = pd.to_datetime(master["hour_ending_IST"], errors="coerce").dt.floor("h")
    master_2025 = master[master["timestamp"].dt.year == 2025].copy()
    master_2025.to_csv(OUTPUT_DIR / "master_2025_rows.csv", index=False)

    master_hourly = hourly_mean(
        master_2025,
        "timestamp",
        exclude={"timestamp", "hour_ending_IST", "site_name"},
    )
    site_counts = (
        master_2025.groupby("timestamp", as_index=False)["site_name"]
        .nunique()
        .rename(columns={"site_name": "site_count"})
    )
    master_hourly = master_hourly.merge(site_counts, on="timestamp", how="left")
    master_hourly.to_csv(OUTPUT_DIR / "master_2025_hourly_mean.csv", index=False)

    sensor_files = sorted(path for path in DATA_DIR.glob("*.csv") if path.resolve() != MASTER_FILE.resolve())
    wide_parts: list[pd.DataFrame] = []
    long_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = [
        {
            "source_file": MASTER_FILE.name,
            "raw_rows": len(master),
            "parsed_timestamp_rows": int(master["timestamp"].notna().sum()),
            "years_present": ", ".join(str(year) for year in sorted(master["timestamp"].dt.year.dropna().astype(int).unique())),
            "sensor_2025_rows": "",
            "sensor_2025_hourly_rows": "",
            "common_timestamps": "",
            "status": "master extracted",
            "master_2025_rows": len(master_2025),
            "master_2025_hourly_rows": len(master_hourly),
        }
    ]

    for sensor_file in sensor_files:
        wide, long, summary = build_mapping_rows(master_hourly, sensor_file)
        summaries.append(summary)
        if not wide.empty:
            wide_parts.append(wide)
        if not long.empty:
            long_parts.append(long)

    mapped_wide = pd.concat(wide_parts, ignore_index=True) if wide_parts else pd.DataFrame()
    mapped_long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    mapped_wide.to_csv(OUTPUT_DIR / "mapped_common_hourly_wide_2025.csv", index=False)
    mapped_long.to_csv(OUTPUT_DIR / "mapped_common_hourly_long_2025.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "mapping_summary.csv", index=False)

    chart_outputs = []
    chart_outputs.extend(build_master_chart(master_hourly))
    chart_outputs.extend(build_common_chart(mapped_long))

    print("Done")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Master 2025 rows: {len(master_2025)}")
    print(f"Master 2025 hourly rows: {len(master_hourly)}")
    print(f"Mapped common wide rows: {len(mapped_wide)}")
    print(f"Mapped common long rows: {len(mapped_long)}")
    print("Summary:")
    print(summary_df.fillna("").to_string(index=False))
    if chart_outputs:
        print(f"Charts written: {', '.join(chart_outputs)}")


if __name__ == "__main__":
    main()
