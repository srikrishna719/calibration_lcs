"""Metric honesty, outlier transparency, and configuration consolidation.

Covers four behaviours that were misleading rather than broken: MAPE exploding
on near-zero references, an outlier screen whose loss grew with column count
without saying so, a duplicated validation/objective setting with no rule for
which won, and a hardcoded sensor prefix that ignored the configured one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from config.validation import validate_config
from evaluation.metrics import (
    MAPE_MIN_DENOMINATOR,
    calculate_regression_metrics,
    fit_slope_intercept,
    mape_excluded_fraction,
    mean_absolute_percentage_error,
)
from modules.exporter import _pdf_safe_text
from modules.feature_engineering import select_feature_columns
from modules.preprocessing import preprocess_dataset


class TestMape:
    def test_near_zero_references_are_excluded(self):
        y = pd.Series([0.0, 0.001, 10.0, 20.0])
        p = pd.Series([1.0, 1.0, 11.0, 21.0])
        # Only the two usable rows count: 10% and 5% error.
        assert mean_absolute_percentage_error(y, p) == pytest.approx(7.5)

    def test_the_excluded_share_is_reported(self):
        y = pd.Series([0.0, 0.5, 10.0, 20.0])
        assert mape_excluded_fraction(y) == pytest.approx(0.5)

    def test_it_appears_in_the_metric_suite(self, merged):
        metrics = calculate_regression_metrics(
            pd.Series([0.1, 5.0, 10.0, 20.0]), pd.Series([1.0, 5.0, 11.0, 21.0])
        )
        assert metrics["mape_excluded_fraction"] == pytest.approx(0.25)
        assert np.isfinite(metrics["mape"])

    def test_all_values_below_the_floor_gives_nan(self):
        y = pd.Series([0.1, 0.2, 0.3])
        assert np.isnan(mean_absolute_percentage_error(y, pd.Series([1.0, 1.0, 1.0])))

    def test_a_realistic_low_concentration_series_stays_interpretable(self):
        rng = np.random.default_rng(0)
        y = pd.Series(np.abs(rng.normal(2.0, 0.8, 500)) + 0.01)
        p = y + rng.normal(0, 2, 500)
        # The old behaviour exceeded 100% here purely from the denominator.
        assert mean_absolute_percentage_error(y, p) < 100

    def test_the_floor_is_documented_as_a_constant(self):
        assert MAPE_MIN_DENOMINATOR > 0


class TestDegenerateSlope:
    def test_a_constant_reference_yields_nan_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            slope, intercept = fit_slope_intercept(
                pd.Series([5.0] * 10), pd.Series(np.linspace(4, 6, 10))
            )
        assert np.isnan(slope) and np.isnan(intercept)

    def test_a_real_relationship_still_fits(self):
        slope, intercept = fit_slope_intercept(
            pd.Series([1.0, 2, 3, 4]), pd.Series([2.0, 4, 6, 8])
        )
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0, abs=1e-9)

    def test_non_finite_values_are_dropped(self):
        slope, _ = fit_slope_intercept(
            pd.Series([1.0, 2.0, np.nan, 4.0]), pd.Series([2.0, 4.0, 5.0, 8.0])
        )
        assert slope == pytest.approx(2.0)


class TestOutlierTransparency:
    @staticmethod
    def _clean_frame(columns: int, rows: int = 2000) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=rows, freq="h")})
        for i in range(columns):
            frame[f"c{i}"] = rng.normal(size=rows)
        return frame

    def test_the_loss_grows_with_column_count_on_clean_data(self):
        """Not a bug, but the reason a breakdown is needed to explain the total."""
        cfg = {"missing_strategy": "none", "outlier_method": "iqr", "outlier_threshold": 1.5}
        narrow = preprocess_dataset(self._clean_frame(1), "timestamp", cfg)[1]
        wide = preprocess_dataset(self._clean_frame(16), "timestamp", cfg)[1]
        assert wide.percentage_removed > 5 * narrow.percentage_removed

    def test_each_column_reports_what_it_would_remove_alone(self):
        cfg = {"missing_strategy": "none", "outlier_method": "iqr", "outlier_threshold": 1.5}
        _, summary = preprocess_dataset(self._clean_frame(8), "timestamp", cfg)
        assert len(summary.outlier_rows_by_column) == 8
        # The union exceeds any single column, which is exactly what surprises people.
        assert summary.rows_removed_as_outliers > max(summary.outlier_rows_by_column.values())

    def test_the_breakdown_is_ordered_worst_first(self):
        cfg = {"missing_strategy": "none", "outlier_method": "iqr", "outlier_threshold": 1.5}
        _, summary = preprocess_dataset(self._clean_frame(6), "timestamp", cfg)
        counts = list(summary.outlier_rows_by_column.values())
        assert counts == sorted(counts, reverse=True)

    def test_the_screen_can_be_narrowed_to_named_columns(self):
        frame = self._clean_frame(8)
        cfg = {
            "missing_strategy": "none", "outlier_method": "iqr", "outlier_threshold": 1.5,
            "outlier_columns": ["c0"],
        }
        _, summary = preprocess_dataset(frame, "timestamp", cfg)
        assert summary.outlier_columns == ["c0"]
        assert list(summary.outlier_rows_by_column) == ["c0"]

        wide = preprocess_dataset(frame, "timestamp", {**cfg, "outlier_columns": []})[1]
        assert summary.rows_removed_as_outliers < wide.rows_removed_as_outliers

    def test_no_outlier_removal_reports_no_breakdown(self):
        cfg = {"missing_strategy": "none", "outlier_method": "none"}
        _, summary = preprocess_dataset(self._clean_frame(4), "timestamp", cfg)
        assert summary.outlier_rows_by_column == {}
        assert summary.rows_removed_as_outliers == 0


class TestConfigConsolidation:
    def test_the_legacy_blocks_fold_into_training(self):
        config = validate_config({"validation": {"method": "holdout"}, "modelling": {"objective": "bias"}})
        assert config["training"]["validation_method"] == "holdout"
        assert config["training"]["modelling_objective"] == "bias"

    def test_the_legacy_blocks_are_removed_so_there_is_one_place_to_look(self):
        config = validate_config({"validation": {"method": "holdout"}})
        assert "validation" not in config
        assert "modelling" not in config

    def test_an_explicit_training_value_wins(self):
        config = validate_config({
            "validation": {"method": "holdout"},
            "training": {"validation_method": "kfold"},
        })
        assert config["training"]["validation_method"] == "kfold"

    def test_the_packaged_default_no_longer_duplicates_them(self):
        from config.validation import load_default_config

        raw = load_default_config()
        assert "validation" not in raw
        assert "modelling" not in raw
        assert raw["training"]["validation_method"]


class TestSensorPrefix:
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame({
            "reference_pm25": [1.0], "lcs_pm25": [1.0], "lcs_temp": [1.0], "other": [1.0],
        })

    def test_the_configured_prefix_is_honoured(self):
        columns = select_feature_columns(
            self._frame(), "reference_pm25", {"optional_columns": ["other"]}, sensor_prefix="lcs"
        )
        assert set(columns) == {"lcs_pm25", "lcs_temp", "other"}

    def test_a_prefix_matching_nothing_falls_back_to_every_feature(self):
        """Narrowing to nothing would silently engineer no features at all."""
        columns = select_feature_columns(
            self._frame(), "reference_pm25", {"optional_columns": ["absent"]}, sensor_prefix="nope"
        )
        assert set(columns) == {"lcs_pm25", "lcs_temp", "other"}

    def test_the_default_prefix_still_works(self, merged):
        columns = select_feature_columns(
            merged, "reference_pm25", {"optional_columns": ["sensor_temperature"]}
        )
        assert all(c.startswith("sensor_") or c == "sensor_temperature" for c in columns)


class TestPdfText:
    @pytest.mark.parametrize("source,expected", [
        ("PM₂.₅", "PM2.5"),
        ("Kolkata — Baranagar", "Kolkata - Baranagar"),
        ("r² ≥ 0.9", "r² >= 0.9"),
        ("temp × humidity", "temp x humidity"),
        ("‘quoted’", "'quoted'"),
    ])
    def test_common_scientific_notation_survives(self, source, expected):
        assert _pdf_safe_text(source) == expected

    def test_latin1_characters_pass_through(self):
        assert _pdf_safe_text("µg/m³") == "µg/m³"

    def test_genuinely_unrepresentable_text_still_degrades(self):
        assert _pdf_safe_text("中文") == "??"

    def test_truncation_still_applies(self):
        assert _pdf_safe_text("sensor_pm25_rolling_mean_24", limit=10) == "sensor_pm2"


class TestOutlierBreakdownRendering:
    """The Preprocessing step's breakdown panel, exercised without a browser."""

    @staticmethod
    def _summary(**overrides):
        from modules.preprocessing import PreprocessingSummary

        defaults = dict(
            original_rows=1000, cleaned_rows=900, rows_removed_as_outliers=100,
            numeric_columns=["a", "b", "c"], percentage_removed=10.0,
            outlier_rows_by_column={"a": 60, "b": 30, "c": 20},
            outlier_columns=["a", "b", "c"],
        )
        defaults.update(overrides)
        return PreprocessingSummary(**defaults)

    def test_it_renders_and_explains_the_union(self):
        from unittest import mock
        import ui.app as app

        with mock.patch.object(app, "st") as fake_st:
            fake_st.expander.return_value.__enter__ = mock.Mock(return_value=None)
            fake_st.expander.return_value.__exit__ = mock.Mock(return_value=False)
            app._render_outlier_breakdown("Sensor", self._summary())

        title = fake_st.expander.call_args.args[0]
        assert "100 rows" in title and "10%" in title
        caption = fake_st.caption.call_args.args[0]
        # The worst single column removes 60; the union removes 100.
        assert "60" in caption and "100" in caption
        assert fake_st.dataframe.call_count == 1

    def test_it_stays_silent_when_nothing_was_removed(self):
        from unittest import mock
        import ui.app as app

        with mock.patch.object(app, "st") as fake_st:
            app._render_outlier_breakdown(
                "Sensor", self._summary(rows_removed_as_outliers=0, outlier_rows_by_column={})
            )
        assert fake_st.expander.call_count == 0

    def test_a_single_screened_column_needs_no_explanation(self):
        from unittest import mock
        import ui.app as app

        with mock.patch.object(app, "st") as fake_st:
            fake_st.expander.return_value.__enter__ = mock.Mock(return_value=None)
            fake_st.expander.return_value.__exit__ = mock.Mock(return_value=False)
            app._render_outlier_breakdown(
                "Sensor",
                self._summary(outlier_rows_by_column={"a": 100}, outlier_columns=["a"]),
            )
        assert fake_st.caption.call_count == 0
        assert fake_st.dataframe.call_count == 1
