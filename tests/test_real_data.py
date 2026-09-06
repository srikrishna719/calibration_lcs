"""The 2025 co-location dataset: reference station vs six low-cost devices.

Skipped entirely when data/processed_2025_trends is absent, since that folder
is gitignored. It is worth having because the sample data is synthetic and
tidy, while this file carries everything real data does -- text columns,
all-NaN channels, duplicate timestamps across co-located devices, and derived
columns that encode the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.alignment import non_numeric_columns, resample_timeseries
from modules.data_loader import load_and_validate_dataset, summarize_duplicate_timestamps
from models.train import fitted_scaler
from pipeline.run_pipeline import deployable_feature_subset, train_on_prepared_dataset

pytestmark = pytest.mark.realdata

TARGET, TIMESTAMP = "reference_pm25", "timestamp"

# Derived columns of the form sensor_x - reference_x. Any of them hands a model
# the target algebraically; they carry no reference_ prefix so the deployability
# filter does not catch them.
DIFFERENCE_COLUMNS = [
    "pm25_sensor_minus_reference",
    "co_sensor_minus_reference",
    "temperature_sensor_minus_reference",
    "humidity_sensor_minus_reference",
    "pressure_sensor_minus_reference",
]


@pytest.fixture(scope="module")
def device_frame(real_single_device):
    """One device, numeric channels that actually carry data, no NaN rows."""
    frame = real_single_device.copy()
    usable = [
        c for c in frame.select_dtypes(include="number").columns
        if frame[c].notna().mean() > 0.5 and c not in DIFFERENCE_COLUMNS
    ]
    frame = frame[[TIMESTAMP] + usable].dropna().reset_index(drop=True)
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])
    assert len(frame) > 100, "not enough complete rows to model"
    return frame


class TestShapeOfRealData:
    def test_it_carries_the_text_columns_that_used_to_break_resampling(self, real_merged_raw):
        text = non_numeric_columns(real_merged_raw, TIMESTAMP)
        assert len(text) >= 5
        assert "reference_site_name" in text and "sensor_device_id" in text

    def test_resampling_survives_those_columns(self, real_single_device):
        frame = real_single_device.copy()
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])
        out = resample_timeseries(frame, TIMESTAMP, "1h", "mean")
        assert len(out) > 0
        assert not any(c in out.columns for c in ("reference_site_name", "sensor_device_id"))

    def test_all_nan_channels_are_present_and_survive_loading(self, real_merged_raw):
        empty = [c for c in real_merged_raw.columns if real_merged_raw[c].isna().all()]
        assert empty, "expected some instrument channels to carry no data"
        assert "sensor_co2" in empty

    def test_loading_the_whole_file_refuses_rather_than_interleaving_devices(self, real_merged_raw):
        """Six co-located devices share the hourly grid.

        Keeping one row per timestamp would drop ~80% of the file and mix the
        devices into a single incoherent series, so the loader refuses and says
        which column separates them.
        """
        assert real_merged_raw[TIMESTAMP].duplicated().sum() > 0
        with pytest.raises(ValueError, match="repeat a timestamp"):
            load_and_validate_dataset(real_merged_raw, TIMESTAMP, "Merged", "UTC")

    def test_the_device_column_is_identified_automatically(self, real_merged_raw):
        summary = summarize_duplicate_timestamps(real_merged_raw, TIMESTAMP)
        assert "sensor_device_id" in summary.candidate_group_columns

    def test_selecting_a_device_keeps_every_one_of_its_rows(self, real_merged_raw):
        device = real_merged_raw["sensor_device_id"].value_counts().idxmax()
        expected = int((real_merged_raw["sensor_device_id"] == device).sum())

        loaded, summary = load_and_validate_dataset(
            real_merged_raw, TIMESTAMP, "Merged", "UTC",
            duplicate_strategy="first",
            group_column="sensor_device_id", group_value=device,
            return_summary=True,
        )
        assert len(loaded) == expected
        assert loaded[TIMESTAMP].duplicated().sum() == 0
        assert (loaded["sensor_device_id"] == device).all()
        assert f"Kept only sensor_device_id='{device}'" in summary.message()

    def test_averaging_devices_covers_the_whole_grid(self, real_merged_raw):
        loaded, summary = load_and_validate_dataset(
            real_merged_raw, TIMESTAMP, "Merged", "UTC",
            duplicate_strategy="mean", return_summary=True,
        )
        assert len(loaded) == real_merged_raw[TIMESTAMP].nunique()
        assert loaded[TIMESTAMP].duplicated().sum() == 0
        assert "combined with the mean" in summary.message()

    def test_a_single_device_loads_without_loss(self, real_single_device):
        loaded = load_and_validate_dataset(real_single_device, TIMESTAMP, "Merged", "UTC")
        assert len(loaded) == len(real_single_device)


class TestTargetEncodingColumns:
    def test_difference_columns_reconstruct_the_target_exactly(self, real_merged_raw):
        frame = real_merged_raw.dropna(subset=["sensor_pm25", TARGET, "pm25_sensor_minus_reference"])
        implied = frame["sensor_pm25"] - frame["pm25_sensor_minus_reference"]
        assert np.abs(implied - frame[TARGET]).max() < 1e-9

    def test_the_deployability_filter_does_not_catch_them(self, real_merged_raw, config):
        """They carry no reference_ prefix, so a user must exclude them by hand."""
        frame = real_merged_raw[[TIMESTAMP, TARGET, "sensor_pm25", "pm25_sensor_minus_reference"]]
        subset = deployable_feature_subset(frame, TIMESTAMP, TARGET, config)
        assert "pm25_sensor_minus_reference" in subset

    def test_including_one_gives_an_implausibly_perfect_fit(self, real_single_device, config_factory):
        frame = real_single_device[
            [TIMESTAMP, TARGET, "sensor_pm25", "pm25_sensor_minus_reference"]
        ].dropna().reset_index(drop=True)
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])

        leaked = train_on_prepared_dataset(
            frame, TARGET, config_factory(["ridge"]),
            feature_subset=["sensor_pm25", "pm25_sensor_minus_reference"],
        )
        honest = train_on_prepared_dataset(
            frame, TARGET, config_factory(["ridge"]), feature_subset=["sensor_pm25"],
        )
        assert leaked["training_results"]["ridge"].metrics["r2"] > 0.999
        assert honest["training_results"]["ridge"].metrics["r2"] < 0.95


class TestModellingOnRealData:
    def test_reference_channels_are_excluded_by_default(self, device_frame, config):
        subset = deployable_feature_subset(device_frame, TIMESTAMP, TARGET, config)
        assert subset
        assert not any(c.startswith("reference_") for c in subset)
        assert any(c.startswith("sensor_") for c in subset)

    @pytest.mark.parametrize("model_name", ["ols_regression", "ridge", "random_forest"])
    def test_models_train_and_produce_finite_metrics(self, device_frame, config_factory, model_name):
        cfg = config_factory([model_name])
        subset = deployable_feature_subset(device_frame, TIMESTAMP, TARGET, cfg)
        out = train_on_prepared_dataset(device_frame, TARGET, cfg, feature_subset=subset)
        metrics = out["training_results"][model_name].metrics
        assert np.isfinite(metrics["rmse"]) and metrics["rmse"] > 0
        assert np.isfinite(metrics["r2"])
        assert out["calibrated_dataset"]["calibrated_value"].notna().all()

    def test_scaled_training_keeps_the_scaler_in_the_model(self, device_frame, config_factory):
        cfg = config_factory(["ridge"])
        cfg["normalization"]["method"] = "standard"
        subset = deployable_feature_subset(device_frame, TIMESTAMP, TARGET, cfg)
        out = train_on_prepared_dataset(device_frame, TARGET, cfg, feature_subset=subset)

        scaler = fitted_scaler(out["training_results"]["ridge"].model)
        assert scaler is not None
        assert list(scaler.feature_names_in_) == subset

    def test_exported_model_predicts_from_raw_readings(self, device_frame, config_factory):
        import pickle

        from modules.exporter import export_model_bytes

        cfg = config_factory(["ridge"])
        cfg["normalization"]["method"] = "standard"
        subset = deployable_feature_subset(device_frame, TIMESTAMP, TARGET, cfg)
        out = train_on_prepared_dataset(device_frame, TARGET, cfg, feature_subset=subset)
        result = out["training_results"]["ridge"]

        loaded = pickle.loads(export_model_bytes(result.model))
        raw = device_frame[result.feature_names]
        assert np.allclose(loaded.predict(raw), result.model.predict(raw))

    def test_a_real_sensor_channel_beats_predicting_the_mean(self, device_frame, config_factory):
        out = train_on_prepared_dataset(
            device_frame, TARGET, config_factory(["ridge"]), feature_subset=["sensor_pm25"]
        )
        assert out["training_results"]["ridge"].metrics["r2"] > 0.0
