"""Loading, preprocessing and alignment.

Regression cover for: resampling used to hand text columns to .agg("mean"),
which raised an opaque pandas TypeError on data that had uploaded cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.alignment import (
    align_and_merge_datasets,
    non_numeric_columns,
    resample_timeseries,
)
from modules.data_loader import load_and_validate_dataset, validate_dataset
from pipeline.run_pipeline import run_alignment_stage, run_preprocessing_stage, load_input_data


def _frame(**extra) -> pd.DataFrame:
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="30min"),
        "pm25": [1.0, 2, 3, 4, 5, 6],
    }
    base.update(extra)
    return pd.DataFrame(base)


class TestResampleWithTextColumns:
    def test_text_columns_are_dropped_not_aggregated(self):
        out = resample_timeseries(_frame(site=["A"] * 6, qa=["ok"] * 6), "timestamp", "1h", "mean")
        assert list(out.columns) == ["timestamp", "pm25"]
        assert out["pm25"].tolist() == [1.5, 3.5, 5.5]

    def test_dropped_columns_are_reportable(self):
        frame = _frame(site=["A"] * 6, qa=["ok"] * 6)
        assert non_numeric_columns(frame, "timestamp") == ["site", "qa"]

    def test_timestamp_is_not_reported_as_dropped(self):
        assert non_numeric_columns(_frame(), "timestamp") == []

    def test_all_text_frame_raises_a_readable_error(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "site": ["A", "B", "C"],
        })
        with pytest.raises(ValueError, match="no numeric columns"):
            resample_timeseries(frame, "timestamp", "1h", "mean")

    @pytest.mark.parametrize("aggregation", ["mean", "median", "min", "max", "sum"])
    def test_every_offered_aggregation_survives_text_columns(self, aggregation):
        out = resample_timeseries(_frame(site=["A"] * 6), "timestamp", "1h", aggregation)
        assert list(out.columns) == ["timestamp", "pm25"]


class TestAlignment:
    def test_alignment_reports_dropped_columns_per_dataset(self, _base_config, reference_csv, sensor_csv):
        ref = pd.read_csv(reference_csv)
        sen = pd.read_csv(sensor_csv)
        ref["station_name"] = "CAAQMS-1"
        sen["device_id"] = "LCS-07"

        data = load_input_data(ref, sen, _base_config)
        pre = run_preprocessing_stage(data["reference_raw"], data["sensor_raw"], _base_config)
        out = run_alignment_stage(pre["reference_processed"], pre["sensor_processed"], _base_config)

        dropped = out["alignment_metadata"]["dropped_non_numeric_columns"]
        assert dropped["reference"] == ["station_name"]
        assert dropped["sensor"] == ["device_id"]
        assert len(out["merged_data"]) > 0

    def test_merged_frame_has_prefixed_columns(self, merged):
        assert "reference_pm25" in merged.columns
        assert any(c.startswith("sensor_") for c in merged.columns)

    @pytest.mark.parametrize("sensor_column", ["pm25", "pm25_raw"])
    def test_shared_column_names_do_not_break_lag_detection(self, _base_config, sensor_column):
        """A sensor CSV whose channel is also called "pm25" is the natural case."""
        stamps = pd.date_range("2024-01-01", periods=48, freq="h")
        ref = pd.DataFrame({"timestamp": stamps, "pm25": np.linspace(10, 60, 48)})
        sen = pd.DataFrame({"timestamp": stamps, sensor_column: np.linspace(12, 66, 48)})

        cfg = dict(_base_config["alignment"])
        cfg["max_lag_steps"] = 2
        merged, meta = align_and_merge_datasets(
            reference_df=ref, sensor_df=sen, timestamp_column="timestamp",
            reference_target_column="pm25", sensor_prefix="sensor",
            reference_prefix="reference", config=cfg,
        )
        assert "reference_pm25" in merged.columns
        assert f"sensor_{sensor_column}" in merged.columns
        assert isinstance(meta["lag_steps"], int)

    def test_no_overlap_raises(self, _base_config):
        ref = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "pm25": range(5),
        })
        sen = pd.DataFrame({
            "timestamp": pd.date_range("2030-01-01", periods=5, freq="h"),
            "pm25_raw": range(5),
        })
        with pytest.raises(ValueError, match="No overlapping"):
            align_and_merge_datasets(
                reference_df=ref, sensor_df=sen, timestamp_column="timestamp",
                reference_target_column="pm25", sensor_prefix="sensor",
                reference_prefix="reference", config=_base_config["alignment"],
            )


class TestValidation:
    def test_missing_timestamp_column_is_named_in_the_error(self):
        with pytest.raises(ValueError, match="timestamp column 'timestamp'"):
            validate_dataset(pd.DataFrame({"pm25": [1.0]}), "timestamp", "Reference")

    def test_empty_dataset_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataset(pd.DataFrame(), "timestamp", "Reference")

    def test_dataset_without_numeric_columns_rejected(self):
        frame = pd.DataFrame({"timestamp": ["2024-01-01"], "site": ["A"]})
        with pytest.raises(ValueError, match="at least one numeric"):
            validate_dataset(frame, "timestamp", "Reference")

    def test_duplicate_timestamps_are_collapsed(self):
        frame = pd.DataFrame({
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"],
            "pm25": [1.0, 2.0, 3.0],
        })
        out = load_and_validate_dataset(frame, "timestamp", "Reference", "UTC")
        assert len(out) == 2, "loader keeps one row per timestamp"
