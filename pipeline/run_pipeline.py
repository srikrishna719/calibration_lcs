"""End-to-end calibration pipeline orchestration."""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from evaluation.comparator import create_leaderboard, select_best_model
from models.predict import predict_with_model
from models.train import train_models
from modules.alignment import align_and_merge_datasets
from modules.data_loader import load_and_validate_dataset
from modules.feature_engineering import engineer_features
from modules.preprocessing import preprocess_dataset


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load pipeline configuration from YAML or JSON.

    Parameters
    ----------
    config_path:
        Path to a YAML or JSON configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configuration files.")
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    raise ValueError("Config file must be YAML, YML, or JSON.")


def export_model_bytes(model: Any) -> bytes:
    """Serialize a trained model to bytes.

    Parameters
    ----------
    model:
        Trained model object.

    Returns
    -------
    bytes
        Serialized model bytes.
    """
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.getvalue()


def run_calibration_pipeline(
    reference_source: Any,
    sensor_source: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full calibration pipeline from raw input to model selection.

    Parameters
    ----------
    reference_source:
        Reference dataset path, file-like object, or DataFrame.
    sensor_source:
        Sensor dataset path, file-like object, or DataFrame.
    config:
        Pipeline configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Pipeline outputs for UI and downstream export.
    """
    data_config = config["data"]
    timestamp_column = str(data_config["timestamp_column"])
    target_column = str(data_config["target_column"])
    sensor_prefix = str(data_config.get("sensor_prefix", "sensor"))
    reference_prefix = str(data_config.get("reference_prefix", "reference"))
    reference_target_column = f"{reference_prefix}_{target_column}"

    reference_df = load_and_validate_dataset(
        source=reference_source,
        timestamp_column=timestamp_column,
        dataset_name="Reference",
    )
    sensor_df = load_and_validate_dataset(
        source=sensor_source,
        timestamp_column=timestamp_column,
        dataset_name="Sensor",
    )

    reference_processed, reference_artifacts = preprocess_dataset(
        dataframe=reference_df,
        timestamp_column=timestamp_column,
        config=config["preprocessing"],
        exclude_columns=[target_column],
    )
    sensor_processed, sensor_artifacts = preprocess_dataset(
        dataframe=sensor_df,
        timestamp_column=timestamp_column,
        config=config["preprocessing"],
    )

    aligned_df, alignment_metadata = align_and_merge_datasets(
        reference_df=reference_processed,
        sensor_df=sensor_processed,
        timestamp_column=timestamp_column,
        reference_target_column=target_column,
        sensor_prefix=sensor_prefix,
        reference_prefix=reference_prefix,
        config=config["alignment"],
    )

    featured_df = engineer_features(
        dataframe=aligned_df,
        timestamp_column=timestamp_column,
        target_column=reference_target_column,
        config=config["feature_engineering"],
    )
    if featured_df.empty:
        raise ValueError("Feature engineering removed all rows. Adjust lag or rolling window settings.")

    training_results = train_models(
        dataframe=featured_df,
        target_column=reference_target_column,
        timestamp_column=timestamp_column,
        config=config["training"],
        random_state=int(config.get("app", {}).get("random_state", 42)),
    )
    leaderboard = create_leaderboard(
        training_results=training_results,
        config=config["evaluation"],
    )
    best_result = select_best_model(training_results=training_results, leaderboard=leaderboard)
    result_by_model = {result.model_name: result for result in training_results}

    calibrated_predictions = predict_with_model(
        model=best_result.model,
        dataframe=featured_df,
        target_column=reference_target_column,
        timestamp_column=timestamp_column,
    )
    calibrated_dataset = featured_df[[timestamp_column, reference_target_column]].merge(
        calibrated_predictions,
        on=timestamp_column,
        how="left",
    ).rename(
        columns={
            reference_target_column: "reference_value",
            "prediction": "calibrated_prediction",
        }
    )

    return {
        "reference_processed": reference_processed,
        "sensor_processed": sensor_processed,
        "aligned_data": aligned_df,
        "featured_data": featured_df,
        "leaderboard": leaderboard,
        "training_results": result_by_model,
        "best_model_name": best_result.model_name,
        "best_model": best_result.model,
        "best_model_metrics": best_result.metrics,
        "best_model_predictions": best_result.predictions,
        "calibrated_dataset": calibrated_dataset,
        "alignment_metadata": alignment_metadata,
        "preprocessing_artifacts": {
            "reference": reference_artifacts,
            "sensor": sensor_artifacts,
        },
        "serialized_model": export_model_bytes(best_result.model),
    }
