"""Export helpers for calibration outputs — Zenodo-ready.

Supports CSV, JSON, YAML, and pickle serialization for
datasets, metrics, models, configurations, and metadata.
"""

from __future__ import annotations

import io
import json
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def export_dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def export_metrics_json_bytes(metrics: Dict[str, Any]) -> bytes:
    """Serialize metrics to pretty-printed JSON bytes."""
    return json.dumps(metrics, indent=2, default=str).encode("utf-8")


def export_model_bytes(model: Any) -> bytes:
    """Serialize a fitted model to pickle bytes."""
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.getvalue()


def export_config_json_bytes(config: Dict[str, Any]) -> bytes:
    """Serialize configuration to JSON bytes for reproducibility."""
    return json.dumps(config, indent=2, default=str).encode("utf-8")


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
        "software": "AirQuality Calibration Lab v2.0",
    }


def export_metadata_json_bytes(
    model_name: str,
    features_used: List[str],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> bytes:
    """Serialize full metadata to JSON bytes."""
    metadata = build_metadata(model_name, features_used, metrics, config)
    return json.dumps(metadata, indent=2, default=str).encode("utf-8")
