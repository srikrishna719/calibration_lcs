"""Feature-matrix construction, the statsmodels wrapper, and prediction.

Regression cover for three defects that shared a cause -- the feature set a
model was fitted on was not carried through to prediction:

* a restricted feature subset crashed training and export, because prediction
  inferred features by dropping target and timestamp;
* a subset matching nothing silently trained on every feature instead;
* the OLS wrapper filled features it was not given with zeros, returning
  plausible numbers from a model that was never fitted -- which is why the
  first defect stayed hidden for the two OLS models.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.predict import predict_with_model, select_model_features
from models.train import StatsmodelsOLSRegressor, prepare_training_matrices, train_models
from pipeline.run_pipeline import train_on_prepared_dataset

TARGET, TIMESTAMP = "reference_pm25", "timestamp"
SUBSET = ["sensor_pm25_raw", "sensor_humidity"]
ALL_MODELS = [
    "ols_regression", "multiple_linear_regression",
    "ridge", "lasso", "random_forest", "xgboost",
]


class TestFeatureSubset:
    def test_subset_is_honoured_in_the_requested_order(self, merged):
        features, _, _ = prepare_training_matrices(
            merged, TARGET, TIMESTAMP, feature_subset=["sensor_voc", "sensor_pm25_raw"]
        )
        assert features.columns.tolist() == ["sensor_voc", "sensor_pm25_raw"]

    @pytest.mark.parametrize("subset", [None, []])
    def test_none_and_empty_both_mean_every_feature(self, merged, subset):
        features, _, _ = prepare_training_matrices(merged, TARGET, TIMESTAMP, feature_subset=subset)
        expected, _, _ = prepare_training_matrices(merged, TARGET, TIMESTAMP)
        assert features.columns.tolist() == expected.columns.tolist()

    def test_unknown_column_raises_instead_of_using_everything(self, merged):
        with pytest.raises(ValueError, match="not found in the modelling dataset: nope"):
            prepare_training_matrices(merged, TARGET, TIMESTAMP, feature_subset=["nope"])

    def test_partially_valid_subset_raises_instead_of_dropping(self, merged):
        with pytest.raises(ValueError, match="not found in the modelling dataset: gone"):
            prepare_training_matrices(
                merged, TARGET, TIMESTAMP, feature_subset=["sensor_pm25_raw", "gone"]
            )

    def test_non_numeric_column_is_reported_distinctly(self, merged_copy):
        merged_copy["site"] = "A"
        with pytest.raises(ValueError, match="present but not numeric: site"):
            prepare_training_matrices(
                merged_copy, TARGET, TIMESTAMP, feature_subset=["sensor_pm25_raw", "site"]
            )

    def test_both_causes_reported_together(self, merged_copy):
        merged_copy["site"] = "A"
        with pytest.raises(ValueError) as exc:
            prepare_training_matrices(merged_copy, TARGET, TIMESTAMP, feature_subset=["site", "nope"])
        assert "not found in the modelling dataset: nope" in str(exc.value)
        assert "present but not numeric: site" in str(exc.value)

    def test_missing_target_is_named(self, merged):
        with pytest.raises(ValueError, match="Target column 'absent'"):
            prepare_training_matrices(merged, "absent", TIMESTAMP)


class TestPredictWithFittedFeatures:
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_subset_trains_and_predicts_for_every_model(self, merged, config_factory, model_name):
        out = train_on_prepared_dataset(
            merged, TARGET, config_factory([model_name]), feature_subset=SUBSET
        )
        result = out["training_results"][model_name]
        assert result.feature_names == SUBSET
        assert out["calibrated_dataset"]["calibrated_value"].notna().all()

    def test_source_column_order_does_not_change_predictions(self, merged, config_factory):
        out = train_on_prepared_dataset(
            merged, TARGET, config_factory(["ridge"]),
            feature_subset=["sensor_humidity", "sensor_pm25_raw"],
        )
        result = out["training_results"]["ridge"]
        frame = out["featured_data"]
        shuffled = frame[list(reversed(frame.columns))]

        a = predict_with_model(result.model, frame, TARGET, TIMESTAMP, feature_names=result.feature_names)
        b = predict_with_model(result.model, shuffled, TARGET, TIMESTAMP, feature_names=result.feature_names)
        assert np.allclose(a["prediction"], b["prediction"])

    def test_missing_feature_column_raises(self, merged, config_factory):
        out = train_on_prepared_dataset(merged, TARGET, config_factory(["ridge"]), feature_subset=SUBSET)
        result = out["training_results"]["ridge"]
        with pytest.raises(ValueError, match="missing feature column"):
            predict_with_model(
                result.model,
                out["featured_data"].drop(columns=["sensor_humidity"]),
                TARGET, TIMESTAMP, feature_names=result.feature_names,
            )

    def test_without_feature_names_the_legacy_behaviour_is_kept(self, merged):
        selected = select_model_features(merged, TARGET, TIMESTAMP, feature_names=None)
        assert TARGET not in selected.columns
        assert TIMESTAMP not in selected.columns


class TestStatsmodelsWrapper:
    @pytest.fixture
    def fitted(self, merged):
        X = merged[["sensor_pm25_raw", "sensor_temperature", "sensor_humidity"]].astype(float)
        y = merged[TARGET].astype(float)
        return StatsmodelsOLSRegressor().fit(X, y), X

    def test_missing_feature_raises_rather_than_substituting_zeros(self, fitted):
        model, X = fitted
        with pytest.raises(ValueError, match="missing feature column"):
            model.predict(X.drop(columns=["sensor_humidity"]))

    def test_zero_substitution_would_have_been_materially_wrong(self, fitted):
        """The old behaviour returned plausible numbers from a different model."""
        import statsmodels.api as sm

        model, X = fitted
        baseline = model.predict(X)
        zeroed = X.copy()
        zeroed["sensor_humidity"] = 0.0
        old = np.asarray(
            model.result_.predict(sm.add_constant(zeroed, has_constant="add")), dtype=float
        )
        assert np.abs(old - baseline).max() > 1.0

    def test_column_reordering_is_handled(self, fitted):
        model, X = fitted
        reordered = X[["sensor_humidity", "sensor_pm25_raw", "sensor_temperature"]]
        assert np.allclose(model.predict(reordered), model.predict(X))

    def test_extra_columns_are_ignored(self, fitted):
        model, X = fitted
        extra = X.copy()
        extra["unused"] = 99.0
        assert np.allclose(model.predict(extra), model.predict(X))

    def test_predict_before_fit_raises(self, merged):
        with pytest.raises(ValueError, match="must be fitted"):
            StatsmodelsOLSRegressor().predict(merged[["sensor_pm25_raw"]])

    def test_is_a_usable_sklearn_estimator(self):
        """Needed so the wrapper can sit inside a scaling Pipeline."""
        from sklearn.base import clone

        model = StatsmodelsOLSRegressor()
        assert model.get_params() == {}
        assert clone(model) is not model
        assert model.__sklearn_is_fitted__() is False


class TestValidationMethods:
    @pytest.mark.parametrize("method", ["timeseriessplit", "kfold", "holdout"])
    def test_each_method_produces_metrics_and_predictions(self, merged, config_factory, method):
        results = train_models(
            merged, TARGET, TIMESTAMP,
            config_factory(["ridge"], validation_method=method)["training"],
        )
        result = results[0]
        assert result.validation_method == method
        assert not result.validation_predictions.empty
        assert np.isfinite(result.metrics["rmse"])

    def test_unknown_method_rejected(self, merged, config_factory):
        with pytest.raises(ValueError, match="Validation method must be"):
            train_models(
                merged, TARGET, TIMESTAMP,
                config_factory(["ridge"], validation_method="bootstrap")["training"],
            )

    def test_too_few_rows_rejected(self, merged, config_factory):
        with pytest.raises(ValueError, match="three complete modelling rows"):
            train_models(merged.head(2), TARGET, TIMESTAMP, config_factory(["ridge"])["training"])
