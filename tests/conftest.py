"""Shared fixtures for the calibration test suite.

Fixtures that build datasets are session-scoped: alignment and training are the
slow parts, and every module needs the same merged frame.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.run_pipeline import (  # noqa: E402
    load_config,
    load_input_data,
    run_alignment_stage,
    run_preprocessing_stage,
)

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
REAL_DATA_DIR = PROJECT_ROOT / "data" / "processed_2025_trends"
REAL_MERGED_CSV = REAL_DATA_DIR / "CaliSenseAQ_2025_reference_lcs_hourly_merged.csv"

TARGET = "reference_pm25"
TIMESTAMP = "timestamp"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def _base_config() -> dict:
    return load_config(PROJECT_ROOT / "config" / "default.yaml")


@pytest.fixture
def config(_base_config) -> dict:
    """A fresh, cheap-to-train copy of the default config."""
    cfg = deepcopy(_base_config)
    cfg["training"]["selected_models"] = ["ridge"]
    cfg["training"]["model_params"]["random_forest"]["n_estimators"] = 20
    cfg["training"]["model_params"]["random_forest"]["max_depth"] = 5
    return cfg


@pytest.fixture
def config_factory(config):
    """Build a config with training overrides: ``config_factory(["ridge"], test_size=0.3)``."""

    def _make(models: list[str] | None = None, **training_overrides) -> dict:
        cfg = deepcopy(config)
        if models is not None:
            cfg["training"]["selected_models"] = list(models)
        cfg["training"].update(training_overrides)
        return cfg

    return _make


@pytest.fixture(scope="session")
def reference_csv() -> Path:
    return SAMPLE_DIR / "reference_dataset.csv"


@pytest.fixture(scope="session")
def sensor_csv() -> Path:
    return SAMPLE_DIR / "low_cost_sensor_dataset.csv"


@pytest.fixture(scope="session")
def merged(_base_config, reference_csv, sensor_csv) -> pd.DataFrame:
    """Reference + sensor sample data through load, preprocess and align."""
    cfg = deepcopy(_base_config)
    data = load_input_data(reference_csv, sensor_csv, cfg)
    pre = run_preprocessing_stage(data["reference_raw"], data["sensor_raw"], cfg)
    return run_alignment_stage(pre["reference_processed"], pre["sensor_processed"], cfg)["merged_data"]


@pytest.fixture
def merged_copy(merged) -> pd.DataFrame:
    """Mutable copy, so a test cannot disturb the session-scoped frame."""
    return merged.copy()


@pytest.fixture(scope="session")
def real_merged_raw() -> pd.DataFrame:
    """The 2025 co-location dataset, or skip when it is not checked out."""
    if not REAL_MERGED_CSV.exists():
        pytest.skip(f"real dataset not present at {REAL_MERGED_CSV}")
    return pd.read_csv(REAL_MERGED_CSV)


@pytest.fixture(scope="session")
def real_single_device(real_merged_raw) -> pd.DataFrame:
    """One device's rows, so timestamps are unique as the loader expects."""
    device = real_merged_raw["sensor_device_id"].value_counts().idxmax()
    frame = real_merged_raw[real_merged_raw["sensor_device_id"] == device].copy()
    return frame.reset_index(drop=True)
