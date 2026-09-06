"""Export helpers for calibration outputs and optional archival metadata.

Supports CSV, JSON, YAML, pickle, and PDF serialization for
datasets, metrics, models, configurations, metadata, and
reproducibility files (project_run.json).
"""

from __future__ import annotations

import io
import json
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.diagnostics import COEFFICIENT_TABLE_COLUMNS

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover
    FPDF = None


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that keeps numeric types and converts only truly unserializable objects to str."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def _nested_config_value(config: Dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    """Read a nested config value without assuming intermediate keys exist."""
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _pdf_safe_text(value: Any, limit: Optional[int] = None) -> str:
    """Return text safe for the built-in PDF fonts."""
    text = str(value)
    if limit is not None:
        text = text[:limit]
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _format_report_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{float(value):.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value) if value else "None"
    if isinstance(value, dict):
        return json.dumps(value, cls=_SafeEncoder)
    return str(value)


def _pdf_section(pdf: "FPDF", title: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)


def _pdf_key_values(pdf: "FPDF", items: List[tuple[str, Any]], label_width: int = 48) -> None:
    pdf.set_font("Helvetica", "", 9)
    value_width = pdf.w - pdf.l_margin - pdf.r_margin - label_width
    for label, value in items:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(label_width, 5, _pdf_safe_text(f"{label}:"))
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(value_width, 5, _pdf_safe_text(_format_report_value(value), limit=145))
        pdf.set_x(pdf.l_margin)


def _pdf_list(pdf: "FPDF", values: List[Any], max_items: int = 40) -> None:
    pdf.set_font("Helvetica", "", 8)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    for value in values[:max_items]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 4, _pdf_safe_text(f"- {value}", limit=160))
    if len(values) > max_items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 4, _pdf_safe_text(f"... and {len(values) - max_items} more"))


def _pdf_simple_table(
    pdf: "FPDF",
    dataframe: pd.DataFrame,
    columns: List[str],
    max_rows: int = 20,
) -> None:
    if dataframe is None or dataframe.empty:
        pdf.multi_cell(0, 5, "No table data available.")
        return

    present = [column for column in columns if column in dataframe.columns]
    if not present:
        pdf.multi_cell(0, 5, "No requested columns available.")
        return

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    first_width = min(42, usable_width * 0.30)
    other_width = (usable_width - first_width) / max(1, len(present) - 1)
    widths = [first_width] + [other_width] * (len(present) - 1)

    pdf.set_font("Helvetica", "B", 7)
    pdf.set_x(pdf.l_margin)
    for width, column in zip(widths, present):
        pdf.cell(width, 5, _pdf_safe_text(column, limit=18), border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for _, row in dataframe[present].head(max_rows).iterrows():
        pdf.set_x(pdf.l_margin)
        for width, column in zip(widths, present):
            pdf.cell(width, 5, _pdf_safe_text(_format_report_value(row[column]), limit=18), border=1)
        pdf.ln()
    if len(dataframe) > max_rows:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 5, _pdf_safe_text(f"... {len(dataframe) - max_rows} more row(s) in exported JSON/CSV."))


def export_dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def export_metrics_json_bytes(metrics: Dict[str, Any]) -> bytes:
    """Serialize metrics to pretty-printed JSON bytes."""
    return json.dumps(metrics, indent=2, cls=_SafeEncoder).encode("utf-8")


def build_all_model_metrics_report(
    training_results: Optional[Dict[str, Any]],
    leaderboard: Optional[pd.DataFrame] = None,
    best_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a machine-readable report for every trained model."""
    report: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": "CaliSenseAQ v5.0",
        "best_model_name": best_model_name,
        "leaderboard": (
            leaderboard.to_dict(orient="records")
            if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty
            else []
        ),
        "models": {},
    }
    if not training_results:
        return report

    for model_name, result in training_results.items():
        report["models"][model_name] = {
            "metrics": dict(getattr(result, "metrics", {}) or {}),
            "validation_method": getattr(result, "validation_method", None),
            "feature_count": len(getattr(result, "feature_names", []) or []),
            "feature_names": list(getattr(result, "feature_names", []) or []),
            "best_params": getattr(result, "best_params", None) or {},
            "intercept": getattr(result, "intercept_value", None),
            "feature_importance": getattr(result, "feature_importance", None) or {},
            "coefficients": getattr(result, "coefficients", None) or {},
        }
    return report


def export_all_model_metrics_json_bytes(
    training_results: Optional[Dict[str, Any]],
    leaderboard: Optional[pd.DataFrame] = None,
    best_model_name: Optional[str] = None,
) -> bytes:
    """Serialize all model metrics and explainability details to JSON bytes."""
    report = build_all_model_metrics_report(training_results, leaderboard, best_model_name)
    return json.dumps(report, indent=2, cls=_SafeEncoder).encode("utf-8")


def export_model_bytes(model: Any) -> bytes:
    """Serialize a fitted model to pickle bytes."""
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.getvalue()


def export_config_json_bytes(config: Dict[str, Any]) -> bytes:
    """Serialize configuration to JSON bytes for reproducibility."""
    return json.dumps(config, indent=2, cls=_SafeEncoder).encode("utf-8")


def export_config_yaml_bytes(config: Dict[str, Any]) -> bytes:
    """Serialize configuration to YAML bytes.

    Falls back to JSON if PyYAML is not installed.
    """
    if yaml is not None:
        return yaml.dump(config, default_flow_style=False, sort_keys=False).encode("utf-8")
    return export_config_json_bytes(config)


def build_metadata(
    model_name: str,
    features_used: List[str],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an archival metadata dictionary suitable for demo/Zenodo use.

    Parameters
    ----------
    model_name:
        Name of the selected model.
    features_used:
        List of feature column names used during training.
    metrics:
        Evaluation metrics for the selected model.
    config:
        Full pipeline configuration.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary with provenance information.
    """
    normalization_method = str(
        _nested_config_value(config, ("normalization", "method"), "none") or "none"
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "features_used": features_used,
        "metrics_summary": {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        },
        "pipeline_config": {
            "target_column": config.get("data", {}).get("target_column"),
            "resample_rule": config.get("alignment", {}).get("resample_rule"),
            "outlier_method": config.get("preprocessing", {}).get("outlier_method"),
            "test_size": config.get("training", {}).get("test_size"),
            "random_state": config.get("app", {}).get("random_state"),
            "normalization_method": normalization_method,
        },
        "model_pickle_contains_scaler": normalization_method != "none",
        "model_input_expectation": (
            "Raw (unscaled) feature columns in features_used order; the pickled "
            "estimator applies the fitted scaler itself."
            if normalization_method != "none"
            else "Feature columns in features_used order."
        ),
        "software": "CaliSenseAQ v5.0",
    }


def export_metadata_json_bytes(
    model_name: str,
    features_used: List[str],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> bytes:
    """Serialize full metadata to JSON bytes."""
    metadata = build_metadata(model_name, features_used, metrics, config)
    return json.dumps(metadata, indent=2, cls=_SafeEncoder).encode("utf-8")


def export_project_run_json(
    config: Dict[str, Any],
    selected_target: Optional[str] = None,
    selected_predictors: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    modelling_objective: Optional[str] = None,
) -> bytes:
    """Generate project_run.json bytes for full reproducibility.

    Parameters
    ----------
    config:
        Full pipeline configuration.
    selected_target:
        Target variable name.
    selected_predictors:
        List of predictor variable names.
    model_names:
        List of model names used.
    metrics:
        Performance metrics for best model.
    modelling_objective:
        User-selected objective ('Interpretability Focused' or 'Prediction Accuracy Focused').

    Returns
    -------
    bytes
        JSON bytes of the project run metadata.
    """
    objective = (
        modelling_objective
        if modelling_objective is not None
        else _nested_config_value(
            config,
            ("modelling", "objective"),
            _nested_config_value(config, ("training", "modelling_objective")),
        )
    )
    validation_method = _nested_config_value(
        config,
        ("validation", "method"),
        _nested_config_value(config, ("training", "validation_method"), "timeseriessplit"),
    )

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "software_version": "v5.0",
        "selected_target_variable": selected_target or config.get("data", {}).get("target_column"),
        "selected_predictors": selected_predictors or [],
        "missing_value_strategy": config.get("preprocessing", {}).get("missing_strategy"),
        "outlier_strategy": config.get("preprocessing", {}).get("outlier_method"),
        "outlier_threshold": config.get("preprocessing", {}).get("outlier_threshold"),
        "alignment_method": config.get("alignment", {}).get("merge_strategy"),
        "resample_frequency": config.get("alignment", {}).get("resample_rule"),
        "max_lag_steps": config.get("alignment", {}).get("max_lag_steps"),
        "feature_engineering": {
            "lag_steps": config.get("feature_engineering", {}).get("lag_steps", []),
            "rolling_windows": config.get("feature_engineering", {}).get("rolling_windows", []),
            "polynomial_degree": config.get("feature_engineering", {}).get("polynomial_degree", 1),
            "polynomial_columns": config.get("feature_engineering", {}).get("polynomial_columns", []),
            "interaction_columns": config.get("feature_engineering", {}).get("interaction_columns", []),
            "time_features_enabled": config.get("feature_engineering", {}).get("add_time_features", False),
        },
        "normalization_method": config.get("normalization", {}).get("method", "none"),
        "modelling_objective": objective,
        "selected_models": model_names or [],
        "validation_strategy": validation_method,
        "test_size": config.get("training", {}).get("test_size"),
        "cross_validation_folds": config.get("training", {}).get("cross_validation_folds"),
        "performance_metrics": metrics or {},
    }
    return json.dumps(run, indent=2, cls=_SafeEncoder).encode("utf-8")


def export_project_run_json_bytes(
    config: Dict[str, Any],
    selected_target: Optional[str] = None,
    selected_predictors: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    modelling_objective: Optional[str] = None,
) -> bytes:
    """Backward-compatible alias for ``export_project_run_json``."""
    return export_project_run_json(
        config=config,
        selected_target=selected_target,
        selected_predictors=selected_predictors,
        model_names=model_names,
        metrics=metrics,
        modelling_objective=modelling_objective,
    )


def export_research_report_pdf(
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: List[str],
    config: Dict[str, Any],
    leaderboard: Optional[pd.DataFrame] = None,
    training_results: Optional[Dict[str, Any]] = None,
    prepared_dataset: Optional[pd.DataFrame] = None,
    selected_target: Optional[str] = None,
    selected_predictors: Optional[List[str]] = None,
    modelling_objective: Optional[str] = None,
) -> Optional[bytes]:
    """Generate a researcher-facing PDF report for all trained models."""
    if FPDF is None:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    data_cfg = config.get("data", {})
    prep_cfg = config.get("preprocessing", {})
    align_cfg = config.get("alignment", {})
    fe_cfg = config.get("feature_engineering", {})
    train_cfg = config.get("training", {})
    norm_cfg = config.get("normalization", {})
    eval_cfg = config.get("evaluation", {})
    ts_col = data_cfg.get("timestamp_column", "timestamp")

    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 10, "CaliSenseAQ Research Calibration Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "All trained model metrics, validation setup, processing choices, and reproducibility details.",
        align="C",
    )

    _pdf_section(pdf, "1. Run Summary")
    rows = cols = "N/A"
    time_start = time_end = "N/A"
    if isinstance(prepared_dataset, pd.DataFrame) and not prepared_dataset.empty:
        rows, cols = len(prepared_dataset), len(prepared_dataset.columns)
        if ts_col in prepared_dataset.columns:
            timestamps = pd.to_datetime(prepared_dataset[ts_col], errors="coerce").dropna()
            if not timestamps.empty:
                time_start = timestamps.min()
                time_end = timestamps.max()
    _pdf_key_values(
        pdf,
        [
            ("Generated", generated_at),
            ("Software", "CaliSenseAQ v5.0"),
            ("Best model", model_name),
            ("Selected target", selected_target or data_cfg.get("target_column")),
            ("Configured target", data_cfg.get("target_column")),
            ("Timestamp column", ts_col),
            ("Timezone", data_cfg.get("timezone")),
            ("Prepared rows", rows),
            ("Prepared columns", cols),
            ("Time span start", time_start),
            ("Time span end", time_end),
            ("Objective", modelling_objective or _nested_config_value(config, ("modelling", "objective"), train_cfg.get("modelling_objective"))),
        ],
    )

    _pdf_section(pdf, "2. Data And Preprocessing")
    _pdf_key_values(
        pdf,
        [
            ("Reference prefix", data_cfg.get("reference_prefix")),
            ("Sensor prefix", data_cfg.get("sensor_prefix")),
            ("Selected predictors", selected_predictors or []),
            ("Reference missing strategy", _nested_config_value(prep_cfg, ("reference", "missing_strategy"), prep_cfg.get("missing_strategy"))),
            ("Reference outlier method", _nested_config_value(prep_cfg, ("reference", "outlier_method"), prep_cfg.get("outlier_method"))),
            ("Sensor missing strategy", _nested_config_value(prep_cfg, ("sensor", "missing_strategy"), prep_cfg.get("missing_strategy"))),
            ("Sensor outlier method", _nested_config_value(prep_cfg, ("sensor", "outlier_method"), prep_cfg.get("outlier_method"))),
            ("Outlier threshold", prep_cfg.get("outlier_threshold")),
        ],
    )

    _pdf_section(pdf, "3. Alignment, Features, And Normalization")
    _pdf_key_values(
        pdf,
        [
            ("Resample rule", align_cfg.get("resample_rule")),
            ("Aggregation", align_cfg.get("aggregation")),
            ("Timestamp matching", align_cfg.get("merge_strategy")),
            ("Max lag steps", align_cfg.get("max_lag_steps")),
            ("Lag column", align_cfg.get("lag_column")),
            ("Lag features", fe_cfg.get("lag_steps", [])),
            ("Rolling windows", fe_cfg.get("rolling_windows", [])),
            ("Rolling std", fe_cfg.get("rolling_std", False)),
            ("Polynomial degree", fe_cfg.get("polynomial_degree", 1)),
            ("Polynomial columns", fe_cfg.get("polynomial_columns", [])),
            ("Interaction columns", fe_cfg.get("interaction_columns", [])),
            ("Time features", fe_cfg.get("add_time_features", False)),
            ("Time feature flags", fe_cfg.get("time_feature_flags", {})),
            ("Normalization", norm_cfg.get("method", "none")),
        ],
    )

    _pdf_section(pdf, "4. Training And Validation")
    _pdf_key_values(
        pdf,
        [
            ("Selected models", train_cfg.get("selected_models", [])),
            ("Validation method", train_cfg.get("validation_method", _nested_config_value(config, ("validation", "method")))),
            ("Test split size", train_cfg.get("test_size")),
            ("Cross-validation folds", train_cfg.get("cross_validation_folds")),
            ("Random state", _nested_config_value(config, ("app", "random_state"))),
            ("Evaluation sort metric", eval_cfg.get("sort_by")),
            ("Evaluation ascending", eval_cfg.get("ascending")),
            ("Tuning config", train_cfg.get("tuning", {})),
        ],
    )
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        4,
        "Leaderboard metrics are calculated from validation predictions: out-of-fold predictions for CV or holdout predictions for holdout validation.",
    )

    _pdf_section(pdf, "5. Full Model Leaderboard")
    leaderboard_columns = [
        "model_name",
        "rank",
        "rmse",
        "mae",
        "r2",
        "mape",
        "bias",
        "pearson_r",
        "slope",
        "intercept",
    ]
    _pdf_simple_table(pdf, leaderboard if leaderboard is not None else pd.DataFrame(), leaderboard_columns, max_rows=25)

    _pdf_section(pdf, "6. Best Model Metrics")
    _pdf_key_values(pdf, [(key.upper().replace("_", " "), value) for key, value in metrics.items()], label_width=42)

    if training_results:
        _pdf_section(pdf, "7. Per-Model Details")
        metric_order = ["rmse", "mae", "mape", "r2", "pearson_r", "bias", "slope", "intercept"]
        for trained_name, result in training_results.items():
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _pdf_safe_text(trained_name), new_x="LMARGIN", new_y="NEXT")
            model_metrics = getattr(result, "metrics", {}) or {}
            metric_items = [(metric, model_metrics.get(metric)) for metric in metric_order if metric in model_metrics]
            metric_items.extend(
                [
                    ("Validation method", getattr(result, "validation_method", None)),
                    ("Feature count", len(getattr(result, "feature_names", []) or [])),
                    ("Best params", getattr(result, "best_params", None) or {}),
                    ("Intercept", getattr(result, "intercept_value", None)),
                ]
            )
            _pdf_key_values(pdf, metric_items, label_width=42)

            importance = getattr(result, "feature_importance", None) or {}
            coefficients = getattr(result, "coefficients", None) or {}
            if importance:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, "Top feature importances", new_x="LMARGIN", new_y="NEXT")
                _pdf_list(pdf, [f"{k}: {_format_report_value(v)}" for k, v in list(importance.items())[:20]], max_items=20)
            if coefficients:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, "Top coefficients", new_x="LMARGIN", new_y="NEXT")
                _pdf_list(pdf, [f"{k}: {_format_report_value(v)}" for k, v in list(coefficients.items())[:20]], max_items=20)

            coefficient_table = getattr(result, "coefficient_table", None)
            if isinstance(coefficient_table, pd.DataFrame) and not coefficient_table.empty:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, "Coefficient table preview", new_x="LMARGIN", new_y="NEXT")
                _pdf_simple_table(
                    pdf,
                    coefficient_table,
                    COEFFICIENT_TABLE_COLUMNS,
                    max_rows=12,
                )

            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, "Features used", new_x="LMARGIN", new_y="NEXT")
            _pdf_list(pdf, list(getattr(result, "feature_names", []) or []), max_items=35)
            pdf.ln(2)

    _pdf_section(pdf, "8. Research Export Notes")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "The export bundle also contains compact/full calibrated CSVs, the selected trained model pickle, config JSON/YAML, metrics JSON, all-model metrics JSON, metadata JSON, and project_run.json for reproducibility.",
    )

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)


def export_model_summary_report_pdf(
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: List[str],
    config: Dict[str, Any],
    coefficient_table: Optional[pd.DataFrame] = None,
) -> Optional[bytes]:
    """Generate a PDF model summary report.

    Parameters
    ----------
    model_name:
        Name of the selected model.
    metrics:
        Evaluation metrics dictionary.
    feature_names:
        List of feature column names.
    config:
        Full pipeline configuration.
    coefficient_table:
        Optional coefficient table (for linear models).

    Returns
    -------
    Optional[bytes]
        PDF file bytes, or None if fpdf2 is not installed.
    """
    if FPDF is None:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Calibration Model Summary Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Metadata
    pdf.set_font("Helvetica", "", 10)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 6, _pdf_safe_text(f"Generated: {generated_at}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Software: CaliSenseAQ v5.0", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Model info
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _pdf_safe_text(f"Model: {model_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Target
    target = config.get("data", {}).get("target_column", "N/A")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _pdf_safe_text(f"Target Variable: {target}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Metrics table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Performance Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metric_keys = ["rmse", "mae", "mape", "r2", "pearson_r", "bias", "slope", "intercept"]
    for key in metric_keys:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.2f}"
        pdf.cell(60, 6, key.upper().replace("_", " "), border=1)
        pdf.cell(60, 6, _pdf_safe_text(val), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Features used
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe_text(f"Features Used ({len(feature_names)})"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for feat in feature_names[:50]:  # limit to 50
        pdf.cell(0, 5, _pdf_safe_text(f"  - {feat}"), new_x="LMARGIN", new_y="NEXT")
    if len(feature_names) > 50:
        pdf.cell(0, 5, _pdf_safe_text(f"  ... and {len(feature_names) - 50} more"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Coefficient table (if available)
    if coefficient_table is not None and not coefficient_table.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Coefficient Table", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [45, 25, 25, 25, 25]
        headers = ["Variable", "Coef", "Std Err", "t-stat", "p-value"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 5, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for _, row in coefficient_table.head(30).iterrows():
            vals = [
                str(row.get("Variable", "")),
                f"{row.get('Coefficient', 0):.4f}",
                f"{row.get('Std Error', float('nan')):.4f}" if pd.notna(row.get("Std Error")) else "N/A",
                f"{row.get('t-statistic', float('nan')):.4f}" if pd.notna(row.get("t-statistic")) else "N/A",
                f"{row.get('p-value', float('nan')):.4f}" if pd.notna(row.get("p-value")) else "N/A",
            ]
            for w, v in zip(col_widths, vals):
                pdf.cell(w, 5, _pdf_safe_text(v, limit=20), border=1)
            pdf.ln()

    # Pipeline config
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Pipeline Configuration", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    config_items = [
        ("Missing Strategy", config.get("preprocessing", {}).get("missing_strategy")),
        ("Outlier Method", config.get("preprocessing", {}).get("outlier_method")),
        ("Resample Rule", config.get("alignment", {}).get("resample_rule")),
        ("Merge Strategy", config.get("alignment", {}).get("merge_strategy")),
        ("Test Size", config.get("training", {}).get("test_size")),
        ("CV Folds", config.get("training", {}).get("cross_validation_folds")),
        ("Normalization", config.get("normalization", {}).get("method", "none")),
        ("Validation", _nested_config_value(config, ("validation", "method"), config.get("training", {}).get("validation_method"))),
        ("Objective", _nested_config_value(config, ("modelling", "objective"), config.get("training", {}).get("modelling_objective"))),
    ]
    for label, val in config_items:
        pdf.cell(0, 5, _pdf_safe_text(f"  {label}: {val}"), new_x="LMARGIN", new_y="NEXT")

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)
