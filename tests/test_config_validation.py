"""Configuration validation and empty-frame diagnosis.

Regression cover for two opaque errors. An uploaded config missing a section
produced a bare ``KeyError`` -- shown in the app as "x 'data'". And any dataset
that ended up empty after feature engineering was blamed on lag and rolling
settings, whatever the real cause.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from config.validation import (
    ConfigError,
    load_config_text,
    parse_config_text,
    unknown_sections,
    validate_config,
)
from pipeline.run_pipeline import diagnose_empty_modelling_frame, load_config, run_modeling_stage

TARGET, TIMESTAMP = "reference_pm25", "timestamp"


class TestDefaulting:
    def test_a_partial_config_is_completed_from_defaults(self):
        config = validate_config({"training": {"selected_models": ["ridge"]}})
        assert config["data"]["timestamp_column"] == "timestamp"
        assert config["training"]["selected_models"] == ["ridge"]
        assert config["alignment"]["resample_rule"]

    def test_nested_overrides_do_not_wipe_their_siblings(self):
        config = validate_config({"training": {"test_size": 0.3}})
        assert config["training"]["test_size"] == 0.3
        assert config["training"]["cross_validation_folds"] == 5

    def test_every_section_the_pipeline_reads_is_present(self):
        config = validate_config({})
        for section in (
            "app", "data", "preprocessing", "alignment",
            "feature_engineering", "normalization", "training", "evaluation",
        ):
            assert section in config, section

    def test_the_packaged_default_validates(self, project_root):
        assert load_config(project_root / "config" / "default.yaml")["data"]["target_column"]


class TestRejection:
    def test_a_list_is_rejected_with_a_readable_message(self):
        with pytest.raises(ConfigError, match="must be a mapping of sections"):
            validate_config(["training"])

    def test_none_is_rejected(self):
        with pytest.raises(ConfigError, match="is empty"):
            validate_config(None)

    def test_a_section_of_the_wrong_type_names_the_section(self):
        with pytest.raises(ConfigError, match="Section 'training' must be a mapping"):
            validate_config({"training": ["ridge"]})

    @pytest.mark.parametrize("key", ["timestamp_column", "target_column"])
    def test_an_empty_column_name_is_rejected_by_name(self, key):
        with pytest.raises(ConfigError, match=f"data.{key}"):
            validate_config({"data": {key: "  "}})

    @pytest.mark.parametrize("value", [0, 1, 1.5, -0.2])
    def test_test_size_must_be_a_proportion(self, value):
        with pytest.raises(ConfigError, match="training.test_size"):
            validate_config({"training": {"test_size": value}})

    def test_test_size_of_the_wrong_type_is_rejected(self):
        with pytest.raises(ConfigError, match="must be a number"):
            validate_config({"training": {"test_size": "0.2"}})

    def test_fold_count_must_be_a_whole_number_of_at_least_two(self):
        with pytest.raises(ConfigError, match="cross_validation_folds"):
            validate_config({"training": {"cross_validation_folds": 1}})
        with pytest.raises(ConfigError, match="whole number"):
            validate_config({"training": {"cross_validation_folds": 2.5}})

    def test_an_empty_model_list_is_rejected(self):
        with pytest.raises(ConfigError, match="selected_models"):
            validate_config({"training": {"selected_models": []}})

    def test_an_unknown_validation_method_is_rejected_with_the_options(self):
        with pytest.raises(ConfigError, match="TimeSeriesSplit, K-Fold or Holdout"):
            validate_config({"training": {"validation_method": "bootstrap"}})

    @pytest.mark.parametrize("method", ["timeseriessplit", "K-Fold", "holdout", "TimeSeriesSplit"])
    def test_accepted_validation_methods(self, method):
        assert validate_config({"training": {"validation_method": method}})


class TestUploadPath:
    def test_json_and_yaml_both_load(self):
        payload = {"training": {"selected_models": ["ridge"]}}
        for text, suffix in ((json.dumps(payload), ".json"), ("training:\n  selected_models: [ridge]\n", ".yaml")):
            config, unknown = load_config_text(text, suffix)
            assert config["training"]["selected_models"] == ["ridge"]
            assert unknown == []

    def test_malformed_text_reports_a_parse_error(self):
        with pytest.raises(ConfigError, match="could not be parsed"):
            parse_config_text("{not valid json", ".json")

    def test_an_unsupported_extension_is_named(self):
        with pytest.raises(ConfigError, match="must be YAML, YML, or JSON"):
            parse_config_text("anything", ".txt")

    def test_a_mistyped_section_is_reported_as_unknown(self):
        assert unknown_sections({"trainig": {}, "data": {}}) == ["trainig"]

    def test_the_legacy_top_level_sections_are_not_flagged(self):
        assert unknown_sections({"validation": {}, "modelling": {}}) == []


class TestEmptyFrameDiagnosis:
    @staticmethod
    def _frame(rows: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            TIMESTAMP: pd.date_range("2024-01-01", periods=rows, freq="h"),
            TARGET: [float(i) for i in range(rows)],
            "sensor_pm25_raw": [float(i) + 1 for i in range(rows)],
        })

    def test_an_all_nan_target_is_named_rather_than_blaming_lags(self, config):
        frame = self._frame()
        frame[TARGET] = float("nan")
        message = diagnose_empty_modelling_frame(frame, TIMESTAMP, TARGET, config)
        assert TARGET in message
        assert "no usable numeric values" in message
        assert "lag" not in message.lower()

    def test_a_missing_target_column_is_named(self, config):
        frame = self._frame().drop(columns=[TARGET])
        message = diagnose_empty_modelling_frame(frame, TIMESTAMP, TARGET, config)
        assert "is not in the aligned dataset" in message

    def test_too_few_target_values_says_how_many(self, config):
        frame = self._frame()
        frame.loc[2:, TARGET] = float("nan")
        message = diagnose_empty_modelling_frame(frame, TIMESTAMP, TARGET, config)
        assert "Only 2 row(s)" in message

    def test_an_empty_predictor_is_named(self, config):
        frame = self._frame()
        frame["sensor_pm25_raw"] = float("nan")
        message = diagnose_empty_modelling_frame(frame, TIMESTAMP, TARGET, config)
        assert "sensor_pm25_raw" in message
        assert "predictor" in message

    def test_oversized_windows_are_blamed_only_when_they_are_the_cause(self, config_factory):
        cfg = config_factory(["ridge"])
        cfg["feature_engineering"]["rolling_windows"] = [500]
        cfg["feature_engineering"]["lag_steps"] = [400]
        message = diagnose_empty_modelling_frame(self._frame(20), TIMESTAMP, TARGET, cfg)
        assert "consumes all of them" in message
        assert "500" in message

    def test_the_message_reaches_the_caller(self, config_factory):
        frame = self._frame()
        frame[TARGET] = float("nan")
        with pytest.raises(ValueError, match="no usable numeric values"):
            run_modeling_stage(frame, config_factory(["ridge"]))
