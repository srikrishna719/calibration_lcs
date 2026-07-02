"""Export helpers for calibration outputs — Zenodo-ready.

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


def export_dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def export_metrics_json_bytes(metrics: Dict[str, Any]) -> bytes:
    """Serialize metrics to pretty-printed JSON bytes."""
    return json.dumps(metrics, indent=2, cls=_SafeEncoder).encode("utf-8")


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
    """Build a Zenodo-ready metadata dictionary.

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
        },
        "software": "AirQuality Calibration Lab v5.0",
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
    pdf.cell(0, 6, "Software: Air Quality Calibration Lab v5.0", new_x="LMARGIN", new_y="NEXT")
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
