# Implementation Checklist

## Completed

- ✅ Renamed visible product branding to `calisenseAQ`.
- ✅ Moved saved config upload into advanced import settings.
- ✅ Renamed "Data Configuration" to "Column setup".
- ✅ Clarified shared timestamp wording.
- ✅ Renamed merge strategy UI to timestamp matching with exact/nearest labels.
- ✅ Improved EDA distribution chart color and font sizing.
- ✅ Changed missing-values heatmap to use timestamps when available.
- ✅ Added chart customization to the missing-values chart.
- ✅ Added EDA time-series variable selection.
- ✅ Added plain-language z-score anomaly explanation and threshold control.
- ✅ Moved rolling controls into an advanced feature section.
- ✅ Changed rolling/time feature defaults to off.
- ✅ Separated polynomial powers from pairwise interaction features.
- ✅ Split feature engineering so normalization happens before optional time features.
- ✅ Labeled normalization before/after summary tables and dataset preview.
- ✅ Clarified leaderboard metrics as validation metrics.
- ✅ Added validation vs full fitted prediction scope controls for Results charts.
- ✅ Added date range and chart customization controls to the multi-model time-series overlay.
- ✅ Removed duplicated detailed model comparison from the Modelling step.
- ✅ Improved scatter plot ideal line and OLS fit line contrast.
- ✅ Improved metric comparison chart spacing to reduce title overlap.
- ✅ Improved PNG export failure messaging and verified PNG generation locally.
- ✅ Improved VIF diagnostics with constant-column and unstable-value status.
- ✅ Added VIF scope selection for original predictors vs all trained features.
- ✅ Simplified Residual Analysis by removing plot-style selectors.
- ✅ Renamed main export area to general "Export Outputs".
- ✅ Kept Zenodo/demo archival files as optional export files.
- ✅ Added full calibrated CSV export with all modelling columns plus prediction columns.
- ✅ Verified Python syntax with `compileall`.
- ✅ Verified UI import.
- ✅ Verified sample full-pipeline run.
- ✅ Started Streamlit locally at `http://127.0.0.1:8501`.

## Not Changed Intentionally

- ✅ Did not delete advanced scientific features such as rolling, polynomial, or interaction terms.
- ✅ Did not remove required timestamp/target/timezone setup.
- ✅ Did not make CV folds "empty = use all"; the UI now clarifies the meaning instead.
