"""Run the CaliSenseAQ application pipeline on the merged 2025 dataset."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.plots import create_multi_model_metrics_bar, create_scatter_with_fit
from pipeline.run_pipeline import (
    load_config,
    load_input_data,
    run_alignment_stage,
    run_eda_stage,
    run_modeling_stage,
    run_preprocessing_stage,
)


DATA_DIR = ROOT / "data" / "processed_2025_trends"
INPUT_MERGED = DATA_DIR / "CaliSenseAQ_2025_reference_lcs_hourly_merged_recommended.csv"
OUTPUT_DIR = DATA_DIR / "app_test_results"

REFERENCE_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_app_reference_recommended.csv"
LCS_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_app_lcs_recommended.csv"
LEADERBOARD_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_app_test_leaderboard.csv"
CALIBRATED_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_app_test_calibrated_dataset.csv"
VALIDATION_PREDICTIONS_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_app_test_best_validation_predictions.csv"
SUMMARY_JSON = OUTPUT_DIR / "CaliSenseAQ_2025_app_test_summary.json"


def safe_plot_export(fig, html_path: Path, png_path: Path) -> dict[str, str]:
    fig.write_html(html_path, include_plotlyjs="cdn")
    result = {"html": str(html_path), "png": ""}
    try:
        fig.write_image(png_path, width=1400, height=850, scale=2)
        result["png"] = str(png_path)
    except Exception as exc:
        result["png_error"] = str(exc)
    return result


def build_app_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = pd.read_csv(INPUT_MERGED, parse_dates=["timestamp"])
    merged = merged[merged["mapping_confidence"].isin(["high", "medium"])].copy()

    reference = (
        merged.groupby("timestamp", as_index=False)
        .agg(
            pm25=("reference_pm25", "mean"),
            reference_temperature=("reference_temperature", "mean"),
            reference_humidity=("reference_humidity", "mean"),
            reference_pressure=("reference_pressure", "mean"),
        )
        .sort_values("timestamp")
    )

    sensor_agg: dict[str, tuple[str, str]] = {
        "lcs_pm25": ("sensor_pm25", "mean"),
        "lcs_pm10": ("sensor_pm10", "mean"),
        "lcs_pm1": ("sensor_pm1", "mean"),
        "lcs_temperature": ("sensor_temperature", "mean"),
        "lcs_humidity": ("sensor_humidity", "mean"),
        "lcs_no2": ("sensor_no2", "mean"),
    }
    optional_cols = {
        "lcs_co": "sensor_co",
        "lcs_pressure": "sensor_pressure",
        "lcs_so2": "sensor_so2",
        "lcs_o3": "sensor_o3",
        "lcs_tvoc": "sensor_tvoc",
        "lcs_co2": "sensor_co2",
    }
    for output_col, source_col in optional_cols.items():
        if source_col in merged.columns and merged[source_col].notna().any():
            sensor_agg[output_col] = (source_col, "mean")

    lcs = (
        merged.groupby("timestamp", as_index=False)
        .agg(**sensor_agg)
        .sort_values("timestamp")
    )
    return merged, reference, lcs


def build_config() -> dict[str, Any]:
    config = load_config(ROOT / "config" / "default.yaml")
    config["data"]["timestamp_column"] = "timestamp"
    config["data"]["target_column"] = "pm25"
    config["data"]["timezone"] = "UTC"
    config["data"]["reference_prefix"] = "reference"
    config["data"]["sensor_prefix"] = "sensor"

    config["alignment"]["resample_rule"] = "1h"
    config["alignment"]["aggregation"] = "mean"
    config["alignment"]["merge_strategy"] = "inner"
    config["alignment"]["max_lag_steps"] = 0
    config["alignment"]["lag_column"] = "auto"

    config["feature_engineering"]["lag_steps"] = []
    config["feature_engineering"]["rolling_windows"] = []
    config["feature_engineering"]["rolling_std"] = False
    config["feature_engineering"]["add_time_features"] = False
    config["normalization"]["method"] = "none"

    # Keep the app's model families, but reduce RF trees slightly for repeatable
    # local testing without changing the modelling workflow.
    config["training"]["selected_models"] = [
        "ols_regression",
        "multiple_linear_regression",
        "ridge",
        "lasso",
        "random_forest",
        "xgboost",
    ]
    config["training"]["cross_validation_folds"] = 5
    config["training"]["validation_method"] = "timeseriessplit"
    config["training"].setdefault("model_params", {}).setdefault("random_forest", {})["n_estimators"] = 100
    config["training"]["model_params"]["random_forest"]["max_depth"] = 10
    return config


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged, reference, lcs = build_app_inputs()
    reference.to_csv(REFERENCE_CSV, index=False)
    lcs.to_csv(LCS_CSV, index=False)

    config = build_config()
    data_outputs = load_input_data(reference, lcs, config)
    preprocessing_outputs = run_preprocessing_stage(
        data_outputs["reference_raw"],
        data_outputs["sensor_raw"],
        config,
    )
    alignment_outputs = run_alignment_stage(
        preprocessing_outputs["reference_processed"],
        preprocessing_outputs["sensor_processed"],
        config,
    )
    eda_outputs = run_eda_stage(alignment_outputs["merged_data"], config)
    modeling_outputs = run_modeling_stage(alignment_outputs["merged_data"], config)

    leaderboard = modeling_outputs["leaderboard"].copy()
    leaderboard.to_csv(LEADERBOARD_CSV, index=False)
    modeling_outputs["calibrated_dataset"].to_csv(CALIBRATED_CSV, index=False)

    best_name = modeling_outputs["best_model_name"]
    best_result = modeling_outputs["training_results"][best_name]
    validation_predictions = best_result.validation_predictions.copy()
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_CSV, index=False)

    scatter_fig = create_scatter_with_fit(validation_predictions, model_name=best_name)
    scatter_chart = safe_plot_export(
        scatter_fig,
        OUTPUT_DIR / "CaliSenseAQ_2025_app_test_best_scatter.html",
        OUTPUT_DIR / "CaliSenseAQ_2025_app_test_best_scatter.png",
    )
    metrics_fig = create_multi_model_metrics_bar(leaderboard)
    metrics_chart = safe_plot_export(
        metrics_fig,
        OUTPUT_DIR / "CaliSenseAQ_2025_app_test_metrics_bar.html",
        OUTPUT_DIR / "CaliSenseAQ_2025_app_test_metrics_bar.png",
    )

    summary = {
        "input_merged_file": str(INPUT_MERGED),
        "reference_csv": str(REFERENCE_CSV),
        "lcs_csv": str(LCS_CSV),
        "recommended_merged_rows": len(merged),
        "reference_rows_for_app": len(reference),
        "lcs_rows_for_app": len(lcs),
        "date_range": {
            "start": str(reference["timestamp"].min()),
            "end": str(reference["timestamp"].max()),
        },
        "preprocessing": {
            "reference_original_rows": preprocessing_outputs["preprocessing_summary"]["reference"].original_rows,
            "reference_cleaned_rows": preprocessing_outputs["preprocessing_summary"]["reference"].cleaned_rows,
            "sensor_original_rows": preprocessing_outputs["preprocessing_summary"]["sensor"].original_rows,
            "sensor_cleaned_rows": preprocessing_outputs["preprocessing_summary"]["sensor"].cleaned_rows,
        },
        "alignment": alignment_outputs["alignment_metadata"],
        "eda_numeric_columns": eda_outputs.get("numeric_columns", []),
        "best_model_name": best_name,
        "best_model_metrics": best_result.metrics,
        "trained_models": list(modeling_outputs["training_results"].keys()),
        "leaderboard_csv": str(LEADERBOARD_CSV),
        "calibrated_csv": str(CALIBRATED_CSV),
        "validation_predictions_csv": str(VALIDATION_PREDICTIONS_CSV),
        "scatter_chart": scatter_chart,
        "metrics_chart": metrics_chart,
        "note": (
            "This test uses high/medium-confidence inferred device-site mappings only, "
            "then aggregates to one hourly reference CSV and one hourly LCS CSV to match "
            "the current CaliSenseAQ app input contract."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Done")
    print(f"Reference CSV: {REFERENCE_CSV}")
    print(f"LCS CSV: {LCS_CSV}")
    print(f"Leaderboard CSV: {LEADERBOARD_CSV}")
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Rows: merged={len(merged)}, reference={len(reference)}, lcs={len(lcs)}")
    print(f"Aligned rows: {alignment_outputs['alignment_metadata']['merged_rows']}")
    print(f"Best model: {best_name}")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
