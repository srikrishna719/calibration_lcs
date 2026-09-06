"""End-to-end calibration pipeline orchestration.

Splits the workflow into discrete stages that can be run independently
from the Streamlit UI or as a single end-to-end call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from evaluation.comparator import create_leaderboard, select_best_model
from models.predict import predict_with_model
from models.train import fitted_scaler, train_models
from modules.alignment import align_and_merge_datasets
from modules.data_loader import load_and_validate_dataset
from modules.drift_analysis import generate_post_analysis_outputs
from modules.eda import generate_eda_outputs
from modules.exporter import (
    export_config_json_bytes,
    export_config_yaml_bytes,
    export_dataframe_csv_bytes,
    export_all_model_metrics_json_bytes,
    export_metadata_json_bytes,
    export_metrics_json_bytes,
    export_model_bytes,
    export_research_report_pdf,
    export_model_summary_report_pdf,
    export_project_run_json_bytes,
)
from modules.feature_engineering import append_time_features, engineer_sensor_features
from modules.leakage import LeakageReport, find_target_encoding_columns
from modules.normalization import get_normalization_summary, normalize_dataset
from modules.preprocessing import preprocess_dataset


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load pipeline configuration from YAML or JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configuration files.")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Config file must be YAML, YML, or JSON.")


# -----------------------------------------------------------------------
# Stage 1 — Data Loading
# -----------------------------------------------------------------------

def load_input_data(
    reference_source: Any,
    sensor_source: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Load and validate the reference and LCS datasets."""
    data_cfg = config["data"]
    ts_col = str(data_cfg["timestamp_column"])
    tz = str(data_cfg.get("timezone", "UTC"))
    duplicate_strategy = str(data_cfg.get("duplicate_timestamps", "error"))
    device_column = data_cfg.get("device_column") or None

    reference_df, reference_duplicates = load_and_validate_dataset(
        source=reference_source,
        timestamp_column=ts_col,
        dataset_name="Reference",
        timezone=tz,
        duplicate_strategy=duplicate_strategy,
        group_column=device_column,
        group_value=data_cfg.get("reference_device") or None,
        return_summary=True,
    )
    sensor_df, sensor_duplicates = load_and_validate_dataset(
        source=sensor_source,
        timestamp_column=ts_col,
        dataset_name="LCS",
        timezone=tz,
        duplicate_strategy=duplicate_strategy,
        group_column=device_column,
        group_value=data_cfg.get("sensor_device") or None,
        return_summary=True,
    )
    return {
        "reference_raw": reference_df,
        "sensor_raw": sensor_df,
        "duplicate_summary": {
            "reference": reference_duplicates,
            "sensor": sensor_duplicates,
        },
    }


# -----------------------------------------------------------------------
# Stage 2 — Preprocessing
# -----------------------------------------------------------------------

def run_preprocessing_stage(
    reference_df: Any,
    sensor_df: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Clean the Reference and LCS datasets with independent settings."""
    data_cfg = config["data"]
    ts_col = str(data_cfg["timestamp_column"])
    target_col = str(data_cfg["target_column"])

    prep_cfg = config["preprocessing"]
    sensor_prep_cfg = dict(prep_cfg.get("sensor", prep_cfg))
    ref_prep_cfg = dict(prep_cfg.get("reference", prep_cfg))

    # Build a reference-specific config — optionally disable outlier removal
    if "reference" not in prep_cfg and not bool(prep_cfg.get("apply_to_reference", False)):
        ref_prep_cfg["outlier_method"] = "none"

    ref_clean, ref_summary = preprocess_dataset(
        dataframe=reference_df,
        timestamp_column=ts_col,
        config=ref_prep_cfg,
        exclude_columns=[target_col],
    )
    sen_clean, sen_summary = preprocess_dataset(
        dataframe=sensor_df,
        timestamp_column=ts_col,
        config=sensor_prep_cfg,
    )
    return {
        "reference_processed": ref_clean,
        "sensor_processed": sen_clean,
        "preprocessing_summary": {
            "reference": ref_summary,
            "sensor": sen_summary,
        },
    }


# -----------------------------------------------------------------------
# Stage 3 — Alignment
# -----------------------------------------------------------------------

def run_alignment_stage(
    reference_df: Any,
    sensor_df: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resample, lag-detect, and merge the two datasets."""
    data_cfg = config["data"]
    ts_col = str(data_cfg["timestamp_column"])
    target_col = str(data_cfg["target_column"])
    sensor_prefix = str(data_cfg.get("sensor_prefix", "sensor"))
    reference_prefix = str(data_cfg.get("reference_prefix", "reference"))

    merged_df, alignment_meta = align_and_merge_datasets(
        reference_df=reference_df,
        sensor_df=sensor_df,
        timestamp_column=ts_col,
        reference_target_column=target_col,
        sensor_prefix=sensor_prefix,
        reference_prefix=reference_prefix,
        config=config["alignment"],
    )
    return {
        "merged_data": merged_df,
        "alignment_metadata": alignment_meta,
    }


# -----------------------------------------------------------------------
# Stage 4 — EDA
# -----------------------------------------------------------------------

def run_eda_stage(
    merged_df: Any,
    config: Dict[str, Any],
    raw_merged_df: Any = None,
) -> Dict[str, Any]:
    """Generate exploratory data analysis outputs."""
    ts_col = str(config["data"]["timestamp_column"])
    return generate_eda_outputs(
        merged_df,
        timestamp_column=ts_col,
        raw_dataframe=raw_merged_df,
    )


# -----------------------------------------------------------------------
# Predictor selection
# -----------------------------------------------------------------------

def screen_target_encoding_columns(
    dataframe: Any,
    timestamp_column: str,
    target_column: str,
    config: Dict[str, Any],
) -> tuple[Any, "LeakageReport"]:
    """Drop predictors that reconstruct the target, before anything derives from them.

    Runs on the merged frame rather than the engineered one: a leaking base
    column would otherwise spawn leaking lag, rolling and interaction columns,
    and the pairwise scan does not scale to a wide engineered frame anyway. Set
    ``training.include_target_encoding_predictors`` to keep them.
    """
    report = LeakageReport()
    if bool(config.get("training", {}).get("include_target_encoding_predictors", False)):
        return dataframe, report

    candidates = [
        c for c in dataframe.select_dtypes(include="number").columns
        if c not in (timestamp_column, target_column)
    ]
    report = find_target_encoding_columns(dataframe, target_column, candidates)
    if not report.excluded:
        return dataframe, report
    return dataframe.drop(columns=report.excluded, errors="ignore"), report


def deployable_feature_subset(
    dataframe: Any,
    timestamp_column: str,
    target_column: str,
    config: Dict[str, Any],
) -> Optional[List[str]]:
    """Numeric feature columns a deployed sensor could actually supply.

    Columns carrying the reference prefix are measurements from the instrument
    being calibrated *against*. A model that depends on them cannot be applied
    to a sensor running on its own, and its metrics flatter what a deployable
    calibration would achieve, so they are excluded by default. Set
    ``training.include_reference_predictors`` to keep them for a co-location
    study where that is the intent.

    Returns ``None`` when every feature should be used, which is the signal
    ``train_models`` expects for "no subset".
    """
    if bool(config.get("training", {}).get("include_reference_predictors", False)):
        return None

    reference_prefix = str(config["data"].get("reference_prefix", "reference"))
    numeric = dataframe.select_dtypes(include="number").columns.tolist()
    candidates = [c for c in numeric if c not in (timestamp_column, target_column)]
    deployable = [c for c in candidates if not str(c).startswith(f"{reference_prefix}_")]

    # Nothing left to model with — fall back rather than fail.
    return deployable or None


# -----------------------------------------------------------------------
# Stage 5 — Feature Engineering + Modelling
# -----------------------------------------------------------------------

def run_modeling_stage(
    merged_df: Any,
    config: Dict[str, Any],
    feature_subset: list | None = None,
) -> Dict[str, Any]:
    """Engineer features, train models, rank them, and produce a calibrated dataset.

    Parameters
    ----------
    feature_subset:
        Optional list of column names to restrict training features.
        When None, all engineered features are used.
    """
    data_cfg = config["data"]
    ts_col = str(data_cfg["timestamp_column"])
    target_col = f"{data_cfg['reference_prefix']}_{data_cfg['target_column']}"

    merged_df, leakage_report = screen_target_encoding_columns(
        merged_df, ts_col, target_col, config
    )

    featured_df = engineer_sensor_features(
        dataframe=merged_df,
        timestamp_column=ts_col,
        target_column=target_col,
        config=config["feature_engineering"],
    )
    if featured_df.empty:
        raise ValueError("Feature engineering removed all rows. Adjust lag or rolling settings.")

    # Time features are ordinary numeric predictors and get scaled with the
    # rest, so they must be part of the frame before the preview is computed.
    featured_df = append_time_features(
        dataframe=featured_df,
        timestamp_column=ts_col,
        config=config["feature_engineering"],
    )

    # Settle the predictor set before previewing, so the preview describes the
    # columns the model will actually scale and nothing else.
    if feature_subset is None:
        feature_subset = deployable_feature_subset(featured_df, ts_col, target_col, config)

    # Normalization is applied *inside* each model (see build_estimator), so the
    # scaler is fit per training fold and ships with the exported model. The
    # frame handed to training therefore stays unscaled; what is computed here
    # is a descriptive preview of what that scaling does.
    norm_cfg = config.get("normalization", {})
    norm_method = str(norm_cfg.get("method", "none"))
    normalization_outputs = None
    if norm_method.strip().lower() != "none":
        norm_cols = feature_subset or [
            c for c in featured_df.select_dtypes(include="number").columns
            if c not in [ts_col, target_col]
        ]
        normalization_outputs = {
            "method": norm_method,
            "columns": norm_cols,
            "summary": get_normalization_summary(
                featured_df,
                normalize_dataset(featured_df, norm_cols, norm_method),
                norm_cols,
            ),
            "preview_only": True,
        }

    results = train_models(
        dataframe=featured_df,
        target_column=target_col,
        timestamp_column=ts_col,
        config=config["training"],
        random_state=int(config.get("app", {}).get("random_state", 42)),
        feature_subset=feature_subset,
        normalization_method=norm_method,
    )
    leaderboard = create_leaderboard(results, config["evaluation"])
    best = select_best_model(results, leaderboard)

    predictions = predict_with_model(
        model=best.model,
        dataframe=featured_df,
        target_column=target_col,
        timestamp_column=ts_col,
        feature_names=best.feature_names,
    )
    calibrated = (
        featured_df[[ts_col, target_col]]
        .merge(predictions, on=ts_col, how="left")
        .rename(columns={target_col: "reference_value", "prediction": "calibrated_value"})
    )

    return {
        "featured_data": featured_df,
        "training_results": {r.model_name: r for r in results},
        "training_results_list": results,
        "leaderboard": leaderboard,
        "best_model_name": best.model_name,
        "best_model": best.model,
        "best_model_metrics": best.metrics,
        "calibrated_dataset": calibrated,
        "scaler": fitted_scaler(best.model),
        "normalization_outputs": normalization_outputs,
        "leakage_report": leakage_report,
    }


def train_on_prepared_dataset(
    prepared_df: Any,
    target_column: str,
    config: Dict[str, Any],
    feature_subset: list | None = None,
    normalization_method: str | None = None,
) -> Dict[str, Any]:
    """Train models on an already-prepared dataset.

    Unlike :func:`run_modeling_stage`, this helper assumes feature engineering
    and normalization have already been applied upstream (via the dedicated UI
    steps). It trains, ranks, and produces a calibrated dataset directly from
    ``prepared_df`` without re-engineering or re-scaling, so the user's choices
    in the Variable Selection / Feature Engineering / Normalization steps are
    the single source of truth.

    Parameters
    ----------
    prepared_df:
        Fully prepared modelling dataset (features + timestamp + target).
    target_column:
        Name of the target column to calibrate against.
    config:
        Pipeline configuration dict.
    feature_subset:
        Optional list of feature columns to restrict training.
    normalization_method:
        Scaler to compose into each model. ``prepared_df`` must be unscaled —
        the scaler is fit per training fold and stored inside the fitted model.
        Defaults to ``config["normalization"]["method"]``.

    Returns
    -------
    Dict[str, Any]
        Same keys as :func:`run_modeling_stage`.
    """
    ts_col = str(config["data"]["timestamp_column"])
    if normalization_method is None:
        normalization_method = str(config.get("normalization", {}).get("method", "none"))

    if prepared_df is None or len(prepared_df) == 0:
        raise ValueError("Prepared dataset is empty. Revisit the preparation steps.")
    if target_column not in prepared_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the prepared dataset."
        )

    results = train_models(
        dataframe=prepared_df,
        target_column=target_column,
        timestamp_column=ts_col,
        config=config["training"],
        random_state=int(config.get("app", {}).get("random_state", 42)),
        feature_subset=feature_subset,
        normalization_method=normalization_method,
    )
    leaderboard = create_leaderboard(results, config["evaluation"])
    best = select_best_model(results, leaderboard)

    predictions = predict_with_model(
        model=best.model,
        dataframe=prepared_df,
        target_column=target_column,
        timestamp_column=ts_col,
        feature_names=best.feature_names,
    )
    calibrated = (
        prepared_df[[ts_col, target_column]]
        .merge(predictions, on=ts_col, how="left")
        .rename(columns={target_column: "reference_value", "prediction": "calibrated_value"})
    )

    return {
        "featured_data": prepared_df,
        "training_results": {r.model_name: r for r in results},
        "training_results_list": results,
        "leaderboard": leaderboard,
        "best_model_name": best.model_name,
        "best_model": best.model,
        "best_model_metrics": best.metrics,
        "calibrated_dataset": calibrated,
        "scaler": fitted_scaler(best.model),
        "normalization_outputs": None,
    }


# -----------------------------------------------------------------------
# Stage 6 — Post-Calibration Analysis
# -----------------------------------------------------------------------

def run_post_analysis_stage(
    predictions_df: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run drift detection and residual analysis."""
    drift_cfg = config.get("drift_analysis", {})
    rolling_window = int(drift_cfg.get("rolling_window", 6))
    drift_threshold = float(drift_cfg.get("drift_threshold", 1.5))
    return generate_post_analysis_outputs(
        predictions_df,
        rolling_window=rolling_window,
        drift_threshold=drift_threshold,
    )


# -----------------------------------------------------------------------
# Stage 7 — Export
# -----------------------------------------------------------------------

def build_export_bundle(
    calibrated_dataset: Any,
    selected_model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: list,
    config: Dict[str, Any],
    full_calibrated_dataset: Any = None,
    coefficient_table: Any = None,
    selected_target: Optional[str] = None,
    selected_predictors: Optional[List[str]] = None,
    modelling_objective: Optional[str] = None,
    leaderboard: Any = None,
    training_results: Optional[Dict[str, Any]] = None,
    prepared_dataset: Any = None,
) -> Dict[str, bytes]:
    """Create exportable artefacts for the selected model."""
    bundle = {
        "calibrated_dataset_csv": export_dataframe_csv_bytes(calibrated_dataset),
        "model_pickle": export_model_bytes(selected_model),
        "metrics_json": export_metrics_json_bytes(metrics),
        "all_model_metrics_json": export_all_model_metrics_json_bytes(
            training_results=training_results,
            leaderboard=leaderboard,
            best_model_name=model_name,
        ),
        "config_json": export_config_json_bytes(config),
        "config_yaml": export_config_yaml_bytes(config),
        "metadata_json": export_metadata_json_bytes(
            model_name=model_name,
            features_used=feature_names,
            metrics=metrics,
            config=config,
        ),
        "project_run_json": export_project_run_json_bytes(
            config=config,
            selected_target=selected_target,
            selected_predictors=selected_predictors,
            model_names=list(training_results.keys()) if training_results else [model_name],
            metrics=metrics,
            modelling_objective=modelling_objective,
        ),
    }
    if full_calibrated_dataset is not None:
        bundle["full_calibrated_dataset_csv"] = export_dataframe_csv_bytes(full_calibrated_dataset)

    # PDF report
    pdf_bytes = export_model_summary_report_pdf(
        model_name=model_name,
        metrics=metrics,
        feature_names=feature_names,
        config=config,
        coefficient_table=coefficient_table,
    )
    if pdf_bytes:
        bundle["model_summary_pdf"] = pdf_bytes

    research_pdf_bytes = export_research_report_pdf(
        model_name=model_name,
        metrics=metrics,
        feature_names=feature_names,
        config=config,
        leaderboard=leaderboard,
        training_results=training_results,
        prepared_dataset=prepared_dataset,
        selected_target=selected_target,
        selected_predictors=selected_predictors,
        modelling_objective=modelling_objective,
    )
    if research_pdf_bytes:
        bundle["research_report_pdf"] = research_pdf_bytes

    return bundle


# -----------------------------------------------------------------------
# Full pipeline (non-interactive)
# -----------------------------------------------------------------------

def run_full_pipeline(
    reference_source: Any,
    sensor_source: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the complete calibration pipeline end to end."""
    data = load_input_data(reference_source, sensor_source, config)
    preprocess = run_preprocessing_stage(
        data["reference_raw"], data["sensor_raw"], config
    )
    alignment = run_alignment_stage(
        preprocess["reference_processed"], preprocess["sensor_processed"], config
    )
    eda = run_eda_stage(alignment["merged_data"], config)
    modeling = run_modeling_stage(alignment["merged_data"], config)

    best_result = modeling["training_results"][modeling["best_model_name"]]
    post = run_post_analysis_stage(best_result.full_predictions, config)

    export = build_export_bundle(
        calibrated_dataset=modeling["calibrated_dataset"],
        selected_model=modeling["best_model"],
        model_name=modeling["best_model_name"],
        metrics=modeling["best_model_metrics"],
        feature_names=best_result.feature_names,
        config=config,
        leaderboard=modeling["leaderboard"],
        training_results=modeling["training_results"],
        prepared_dataset=modeling["featured_data"],
    )
    return {
        **data,
        **preprocess,
        **alignment,
        **eda,
        **modeling,
        **post,
        **export,
    }
