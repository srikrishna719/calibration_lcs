"""Time features that never repeat, and the validation method that hides them.

Measured on the 2025 co-location data, trained on the first 70% and applied to
the following weeks: adding ``unix_timestamp`` took a random forest from 0.271
to 0.218 R2 and ridge below zero, while *raising* the shuffled K-Fold score
(0.464 -> 0.570). K-Fold interleaves past and future, so the feature it cannot
generalise with looks like the one that helps most.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from modules.feature_engineering import (
    ALWAYS_MONOTONIC_TIME_FEATURES,
    SEASONAL_TIME_FEATURES,
    dataset_span_days,
    monotonic_time_features,
)

COLUMNS = [
    "sensor_pm25", "unix_timestamp", "calendar_date",
    "julian_date", "season", "hour_sin", "hour_cos", "hour_of_day",
]


class TestDetection:
    def test_epoch_features_are_always_flagged(self):
        flagged = monotonic_time_features(COLUMNS, span_days=1100)
        assert set(ALWAYS_MONOTONIC_TIME_FEATURES) <= set(flagged)

    def test_seasonal_features_are_flagged_only_within_a_single_year(self):
        short = monotonic_time_features(COLUMNS, span_days=90)
        long = monotonic_time_features(COLUMNS, span_days=1100)
        assert set(SEASONAL_TIME_FEATURES) <= set(short)
        assert not set(SEASONAL_TIME_FEATURES) & set(long)

    def test_an_unknown_span_is_treated_as_short(self):
        """Assume the cautious case rather than staying silent."""
        assert "julian_date" in monotonic_time_features(COLUMNS, span_days=None)

    @pytest.mark.parametrize("safe", ["hour_sin", "hour_cos", "hour_of_day", "sensor_pm25"])
    def test_repeating_and_sensor_columns_are_never_flagged(self, safe):
        assert safe not in monotonic_time_features(COLUMNS, span_days=90)

    def test_nothing_is_flagged_without_time_features(self):
        assert monotonic_time_features(["sensor_pm25", "sensor_humidity"], 90) == []

    def test_span_is_measured_in_days(self):
        stamps = pd.Series(pd.date_range("2025-01-01", "2025-03-31", freq="h"))
        assert dataset_span_days(stamps) == pytest.approx(89.0, abs=1.0)

    def test_span_of_unparseable_timestamps_is_none(self):
        assert dataset_span_days(pd.Series(["not a date", None])) is None


class TestWarningRendering:
    @staticmethod
    def _frame(*extra_columns: str) -> pd.DataFrame:
        stamps = pd.date_range("2025-01-01", periods=200, freq="h")
        frame = pd.DataFrame({"timestamp": stamps, "sensor_pm25": range(200)})
        for column in extra_columns:
            frame[column] = range(200)
        return frame

    def _render(self, frame, method):
        import ui.app as app

        with mock.patch.object(app, "st") as fake_st:
            app._render_time_feature_validation_warning(frame, "timestamp", method)
        return fake_st

    def test_kfold_with_a_monotonic_feature_warns(self):
        fake_st = self._render(self._frame("unix_timestamp"), "kfold")
        assert fake_st.warning.call_count == 1
        message = fake_st.warning.call_args.args[0]
        assert "unix_timestamp" in message
        assert "K-Fold" in message

    def test_chronological_validation_informs_rather_than_warns(self):
        fake_st = self._render(self._frame("unix_timestamp"), "timeseriessplit")
        assert fake_st.warning.call_count == 0
        assert fake_st.info.call_count == 1
        assert "unix_timestamp" in fake_st.info.call_args.args[0]

    def test_holdout_also_only_informs(self):
        fake_st = self._render(self._frame("calendar_date"), "holdout")
        assert fake_st.warning.call_count == 0
        assert fake_st.info.call_count == 1

    def test_nothing_is_said_without_monotonic_features(self):
        fake_st = self._render(self._frame("hour_sin", "hour_cos"), "kfold")
        assert fake_st.warning.call_count == 0
        assert fake_st.info.call_count == 0

    def test_a_missing_timestamp_column_is_handled(self):
        frame = self._frame("unix_timestamp").drop(columns=["timestamp"])
        fake_st = self._render(frame, "kfold")
        assert fake_st.warning.call_count == 0

    def test_the_span_appears_in_the_message(self):
        fake_st = self._render(self._frame("unix_timestamp"), "kfold")
        assert "days of data" in fake_st.warning.call_args.args[0]

    def test_a_multi_year_dataset_does_not_flag_day_of_year(self):
        stamps = pd.date_range("2021-01-01", periods=1200, freq="D")
        frame = pd.DataFrame({
            "timestamp": stamps, "sensor_pm25": range(1200), "julian_date": range(1200),
        })
        fake_st = self._render(frame, "kfold")
        assert fake_st.warning.call_count == 0

    @pytest.mark.parametrize("columns", [("unix_timestamp",), ("unix_timestamp", "calendar_date")])
    def test_the_messages_read_correctly_for_one_or_several(self, columns):
        """Phrased to avoid subject-verb agreement, which the count would break."""
        warned = self._render(self._frame(*columns), "kfold").warning.call_args.args[0]
        noted = self._render(self._frame(*columns), "timeseriessplit").info.call_args.args[0]
        assert "the reported score goes up" in warned
        assert "values that never repeat" in noted
        for bad in ("this feature raise", "these features raises", "only increase over"):
            assert bad not in warned and bad not in noted
