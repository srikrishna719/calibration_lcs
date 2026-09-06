"""Reusable demo workflow and run-history helpers for the UI."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from numbers import Number
from typing import Any

import pandas as pd

from modules.feature_engineering import append_time_features, engineer_sensor_features
from modules.normalization import get_normalization_summary, normalize_dataset
from pipeline.run_pipeline import (
    load_input_data,
    run_alignment_stage,
    run_eda_stage,
    run_preprocessing_stage,
    train_on_prepared_dataset,
)


def _reference_target_name(config: dict[str, Any], merged_columns: list[str]) -> str:
    data_cfg = config["data"]
    preferred = f"{data_cfg.get('reference_prefix', 'reference')}_{data_cfg['target_column']}"
    if preferred in merged_columns:
        return preferred
    if data_cfg["target_column"] in merged_columns:
        return str(data_cfg["target_column"])
    numeric_fallbacks = [column for column in merged_columns if column != data_cfg["timestamp_column"]]
    if not numeric_fallbacks:
        raise ValueError("Demo alignment produced no target column candidates.")
    return str(numeric_fallbacks[0])


def build_sample_demo_state(
    reference_source: Any,
    sensor_source: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the bundled sample data through every main workflow stage."""
    config = deepcopy(config)
    ts_col = str(config["data"]["timestamp_column"])

    data_outputs = load_input_data(reference_source, sensor_source, config)
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
    merged = alignment_outputs["merged_data"]
    eda_outputs = run_eda_stage(merged, config)

    numeric_cols = [column for column in merged.select_dtypes(include="number").columns if column != ts_col]
    target_col = _reference_target_name(config, numeric_cols)
    # Match the app's default: reference-instrument columns are not predictors,
    # because a deployed sensor cannot supply them.
    reference_prefix = str(config.get("data", {}).get("reference_prefix", "reference"))
    candidates = [column for column in numeric_cols if column != target_col]
    predictors = [c for c in candidates if not str(c).startswith(f"{reference_prefix}_")] or candidates
    if not predictors:
        raise ValueError("Demo workflow could not find sensor predictor columns.")

    modelling_df = merged[[ts_col, target_col] + predictors].copy()
    feature_config = config.setdefault("feature_engineering", {})
    featured = engineer_sensor_features(
        dataframe=modelling_df,
        timestamp_column=ts_col,
        target_column=target_col,
        config=feature_config,
    )

    norm_method = str(config.get("normalization", {}).get("method", "none")).strip().lower()

    # Training gets the unscaled frame; the scaler is composed into each model
    # so it is fit per fold and exported with it. ``normalized`` is display only.
    # Time features join the matrix first, because the model scales them too.
    modelling_dataset = append_time_features(
        dataframe=featured.copy(),
        timestamp_column=ts_col,
        config=feature_config,
    )
    added_time_columns = [column for column in modelling_dataset.columns if column not in featured.columns]

    norm_cols = [column for column in modelling_dataset.columns if column not in [ts_col, target_col]]
    if norm_method == "none":
        normalized = modelling_dataset.copy()
    else:
        normalized = normalize_dataset(modelling_dataset.copy(), norm_cols, norm_method)
    normalization_summary = get_normalization_summary(modelling_dataset, normalized, norm_cols)
    modeling_outputs = train_on_prepared_dataset(
        modelling_dataset, target_col, config, normalization_method=norm_method
    )

    return {
        "config": config,
        "input_label": "Bundled sample data demo run",
        "data_outputs": data_outputs,
        "preprocessing_outputs": preprocessing_outputs,
        "alignment_outputs": alignment_outputs,
        "eda_outputs": eda_outputs,
        "selected_target": target_col,
        "selected_predictors": predictors,
        "variable_selection_outputs": {"modelling_dataset": modelling_df},
        "feature_engineering_outputs": {"featured_dataset": featured},
        "featured_preview": featured,
        "normalization_outputs": {
            "modelling_dataset": modelling_dataset,
            "normalized_dataset": normalized,
            "method": norm_method,
            "summary": normalization_summary,
            "added_time_columns": added_time_columns,
        },
        "modeling_outputs": modeling_outputs,
        "selected_model_name": modeling_outputs["best_model_name"],
        "selected_features": None,
    }


def _safe_metric(value: Any) -> Any:
    if isinstance(value, Number):
        return float(value)
    return value


def make_run_history_entry(
    modeling_outputs: dict[str, Any],
    config: dict[str, Any],
    source: str,
    selected_features: list[str] | None = None,
) -> dict[str, Any]:
    model_name = modeling_outputs["best_model_name"]
    result = modeling_outputs["training_results"].get(model_name)
    metrics = getattr(result, "metrics", None) or modeling_outputs.get("best_model_metrics", {})
    feature_names = selected_features or getattr(result, "feature_names", []) or []
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": source,
        "model_name": model_name,
        "metrics": {key: _safe_metric(value) for key, value in dict(metrics).items()},
        "selected_features": list(feature_names),
        "config": deepcopy(config),
    }


def history_as_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in history:
        row = {
            "timestamp": entry.get("timestamp"),
            "source": entry.get("source"),
            "model_name": entry.get("model_name"),
            "feature_count": len(entry.get("selected_features") or []),
        }
        row.update(entry.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)
