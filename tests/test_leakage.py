"""Detection of predictors that encode the calibration target.

Difference columns such as ``sensor_pm25 - reference_pm25`` reconstruct the
target algebraically. A model given one scores near-perfectly while having
learned nothing transferable, and the metrics carry no hint of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.leakage import (
    DEFAULT_R2_THRESHOLD,
    drop_target_encoding_columns,
    find_target_encoding_columns,
)
from pipeline.run_pipeline import (
    run_modeling_stage,
    screen_target_encoding_columns,
    train_on_prepared_dataset,
)

TARGET, TIMESTAMP = "reference_pm25", "timestamp"


@pytest.fixture
def leaky_frame(merged_copy):
    """The sample data plus the difference column real exports ship."""
    merged_copy["pm25_sensor_minus_reference"] = (
        merged_copy["sensor_pm25_raw"] - merged_copy[TARGET]
    )
    return merged_copy


class TestDetection:
    def test_difference_column_is_found(self, leaky_frame):
        report = find_target_encoding_columns(leaky_frame, TARGET)
        assert report.excluded == ["pm25_sensor_minus_reference"]

    def test_the_legitimate_partner_is_kept(self, leaky_frame):
        report = find_target_encoding_columns(leaky_frame, TARGET)
        assert "sensor_pm25_raw" not in report.excluded

    def test_the_reconstructing_pair_is_reported(self, leaky_frame):
        report = find_target_encoding_columns(leaky_frame, TARGET)
        assert len(report.exact_pairs) == 1
        first, second, score = report.exact_pairs[0]
        assert {first, second} == {"sensor_pm25_raw", "pm25_sensor_minus_reference"}
        assert score > DEFAULT_R2_THRESHOLD

    def test_the_reason_names_the_partner(self, leaky_frame):
        report = find_target_encoding_columns(leaky_frame, TARGET)
        assert "sensor_pm25_raw" in report.reasons["pm25_sensor_minus_reference"]

    def test_correlation_alone_would_not_have_found_it(self, leaky_frame):
        """Why the scan is pairwise: the column looks unremarkable on its own."""
        solo = abs(leaky_frame["pm25_sensor_minus_reference"].corr(leaky_frame[TARGET]))
        assert solo < 0.95
        assert find_target_encoding_columns(leaky_frame, TARGET).excluded

    def test_a_renamed_copy_of_the_target_is_caught_on_its_own(self, merged_copy):
        merged_copy["pm25_ref_duplicate"] = merged_copy[TARGET] * 2.0 + 5.0
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert "pm25_ref_duplicate" in report.excluded
        assert report.solo and report.solo[0][0] == "pm25_ref_duplicate"

    def test_a_sum_column_is_caught_too(self, merged_copy):
        merged_copy["pm25_total"] = merged_copy["sensor_pm25_raw"] + merged_copy[TARGET]
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert "pm25_total" in report.excluded

    def test_an_ambiguous_pair_drops_both_rather_than_guessing(self, merged_copy):
        """Neither name says which is derived, so keeping either could keep the leak."""
        merged_copy["pm25_total"] = merged_copy["sensor_pm25_raw"] + merged_copy[TARGET]
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert set(report.excluded) == {"pm25_total", "sensor_pm25_raw"}
        assert "re-enable the one you know" in report.reasons["sensor_pm25_raw"]

    def test_clean_data_is_left_alone(self, merged):
        report = find_target_encoding_columns(merged, TARGET)
        assert report.excluded == []
        assert report.exact_pairs == []
        assert bool(report) is False

    def test_a_strong_but_honest_predictor_is_not_flagged(self, merged_copy):
        rng = np.random.default_rng(0)
        merged_copy["good_sensor"] = merged_copy[TARGET] * 0.9 + rng.normal(0, 2, len(merged_copy))
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert "good_sensor" not in report.excluded

    def test_naming_decides_which_of_a_pair_is_dropped(self, merged_copy):
        merged_copy["pm25_residual_vs_reference"] = (
            merged_copy["sensor_pm25_raw"] - merged_copy[TARGET]
        )
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert report.excluded == ["pm25_residual_vs_reference"]

    def test_an_unconventionally_named_column_is_still_caught(self, merged_copy):
        """Naming only picks the victim; detection itself is purely numeric.

        With no convention to go on the pair is excluded whole, so the leak
        cannot survive even when the column is named uninformatively.
        """
        merged_copy["zz_channel"] = merged_copy["sensor_pm25_raw"] - merged_copy[TARGET]
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert len(report.exact_pairs) == 1
        assert set(report.excluded) == {"zz_channel", "sensor_pm25_raw"}


class TestGuards:
    def test_missing_target_is_reported_not_raised(self, merged):
        report = find_target_encoding_columns(merged, "absent")
        assert report.excluded == []
        assert "not in the dataset" in report.skipped_reason

    def test_wide_frames_are_refused_rather_than_scanned(self, merged_copy):
        for i in range(70):
            merged_copy[f"extra_{i}"] = float(i)
        report = find_target_encoding_columns(merged_copy, TARGET, max_columns=60)
        assert report.excluded == []
        assert "scan limit" in report.skipped_reason

    def test_too_few_rows_is_reported(self, leaky_frame):
        report = find_target_encoding_columns(leaky_frame.head(5), TARGET, min_rows=10)
        assert report.excluded == []
        assert "complete rows" in report.skipped_reason

    def test_constant_columns_are_ignored(self, merged_copy):
        merged_copy["flat"] = 1.0
        report = find_target_encoding_columns(merged_copy, TARGET)
        assert "flat" not in report.excluded

    def test_candidates_can_be_restricted(self, leaky_frame):
        report = find_target_encoding_columns(
            leaky_frame, TARGET, candidates=["sensor_temperature", "sensor_humidity"]
        )
        assert report.excluded == []

    def test_non_numeric_columns_are_skipped(self, leaky_frame):
        leaky_frame["site"] = "A"
        report = find_target_encoding_columns(leaky_frame, TARGET)
        assert "site" not in report.excluded

    def test_drop_helper_removes_the_columns(self, leaky_frame):
        cleaned, report = drop_target_encoding_columns(leaky_frame, TARGET)
        assert "pm25_sensor_minus_reference" not in cleaned.columns
        assert "sensor_pm25_raw" in cleaned.columns
        assert report.excluded


class TestPipelineIntegration:
    def test_screening_runs_before_feature_engineering(self, leaky_frame, config):
        cleaned, report = screen_target_encoding_columns(leaky_frame, TIMESTAMP, TARGET, config)
        assert "pm25_sensor_minus_reference" not in cleaned.columns
        assert report.excluded

    def test_opt_in_keeps_them(self, leaky_frame, config):
        config["training"]["include_target_encoding_predictors"] = True
        cleaned, report = screen_target_encoding_columns(leaky_frame, TIMESTAMP, TARGET, config)
        assert "pm25_sensor_minus_reference" in cleaned.columns
        assert report.excluded == []

    def test_modelling_stage_excludes_them_and_reports_it(self, leaky_frame, config_factory):
        out = run_modeling_stage(leaky_frame, config_factory(["ridge"]))
        assert "pm25_sensor_minus_reference" not in out["training_results"]["ridge"].feature_names
        assert out["leakage_report"].excluded == ["pm25_sensor_minus_reference"]

    def test_no_engineered_column_derives_from_a_leaking_one(self, leaky_frame, config_factory):
        cfg = config_factory(["ridge"])
        cfg["feature_engineering"]["lag_steps"] = [1, 2]
        cfg["feature_engineering"]["rolling_windows"] = [3]
        cfg["feature_engineering"]["optional_columns"] = []
        out = run_modeling_stage(leaky_frame, cfg)
        assert not any("minus_reference" in c for c in out["featured_data"].columns)

    def test_the_metrics_it_prevents_are_implausible(self, leaky_frame, config_factory):
        """Without the screen a model reports a fit no calibration achieves."""
        cfg = config_factory(["ridge"])
        cfg["training"]["include_target_encoding_predictors"] = True
        leaked = run_modeling_stage(leaky_frame, cfg)
        honest = run_modeling_stage(leaky_frame, config_factory(["ridge"]))

        assert leaked["training_results"]["ridge"].metrics["r2"] > 0.999
        assert honest["training_results"]["ridge"].metrics["r2"] < 0.99

    def test_clean_data_reports_nothing_excluded(self, merged, config_factory):
        out = run_modeling_stage(merged, config_factory(["ridge"]))
        assert out["leakage_report"].excluded == []


@pytest.mark.realdata
class TestOnRealData:
    def test_the_real_difference_column_is_caught(self, real_single_device):
        frame = real_single_device.copy()
        numeric = [
            c for c in frame.select_dtypes(include="number").columns
            if frame[c].notna().mean() > 0.5
        ]
        report = find_target_encoding_columns(frame[numeric], TARGET)
        assert "pm25_sensor_minus_reference" in report.excluded
        assert "sensor_pm25" not in report.excluded

    def test_it_survives_the_full_modelling_stage(self, real_single_device, config_factory):
        frame = real_single_device.copy()
        numeric = [
            c for c in frame.select_dtypes(include="number").columns
            if frame[c].notna().mean() > 0.5
        ]
        frame = frame[[TIMESTAMP] + numeric].dropna().reset_index(drop=True)
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])

        out = run_modeling_stage(frame, config_factory(["ridge"]))
        features = out["training_results"]["ridge"].feature_names
        assert "pm25_sensor_minus_reference" not in features
        assert out["leakage_report"].excluded == ["pm25_sensor_minus_reference"]
        assert out["training_results"]["ridge"].metrics["r2"] < 0.99

    def test_only_the_target_matching_difference_column_is_leakage(self, real_single_device):
        """The other difference columns embed reference met data, which is a
        deployability problem rather than target leakage, and is not this
        check's job -- they do not reconstruct pm25."""
        frame = real_single_device.copy()
        numeric = [
            c for c in frame.select_dtypes(include="number").columns
            if frame[c].notna().mean() > 0.5
        ]
        difference_columns = [c for c in numeric if "minus" in c]
        assert len(difference_columns) > 1

        report = find_target_encoding_columns(frame[numeric], TARGET)
        assert report.excluded == ["pm25_sensor_minus_reference"]
        assert "temperature_sensor_minus_reference" not in report.excluded
