# Sreekanth Sir Review - Action Notes

These notes convert the handwritten review comments into implementation tasks, with likely code locations.

## Highest Priority Flow Changes

1. Move normalization before time features.
   - Current state: `ui/app.py` runs Feature Engineering before Normalization (`STEPS` and dispatch around lines 75-99, 2735-2748). `modules/feature_engineering.py` adds time features at the end of `engineer_features()` around lines 324-327, then `ui/app.py` normalizes all feature columns around lines 1557-1620.
   - Change needed: split feature engineering into two phases:
     - sensor-derived features first: lag, rolling, polynomial, interaction;
     - normalization next;
     - time features last, so timestamp-derived columns are not normalized unless explicitly selected.
   - Files: `ui/app.py`, `modules/feature_engineering.py`, `modules/normalization.py`, `config/default.yaml`.

2. Clarify validation metrics vs full-model predictions.
   - Current state: leaderboard metrics come from validation predictions in `models/train.py` around lines 455-481, but charts often use `full_predictions` from a model refit on all rows around lines 486-490.
   - Change needed: label leaderboard as "Validation metrics"; add a short note explaining metric calculation. For scatter/time-series/residual charts, add a selector for "Validation predictions" vs "Full fitted predictions" or default charts to validation predictions for consistency.
   - Files: `models/train.py`, `evaluation/comparator.py`, `ui/app.py`, `modules/plots.py`.

3. Reduce duplication between Modelling comparison and Validation & Results.
   - Current state: model comparison appears inside Modelling (`ui/app.py` around lines 2094-2213) and again in Results (`ui/app.py` around lines 2304-2321).
   - Change needed: keep training controls in Modelling, move model comparison/inspection fully to Validation & Results, or make Modelling show only a compact leaderboard after training.
   - File: `ui/app.py`.

## Upload Data

1. Explain "common timestamp" more clearly.
   - Current state: upload copy mentions common timestamp around `ui/app.py` lines 849-850, and the timestamp input help is around lines 914-920.
   - Change needed: rename UI label to "Shared timestamp column" and explain that the same date/time column must exist in both CSVs so reference and LCS rows can be aligned. If possible, auto-detect common column names after upload.
   - Files: `ui/app.py`, optionally `modules/data_loader.py`.

2. Remove "Optional: Custom Config".
   - Current state: config uploader is in `ui/app.py` lines 877-886, and `resolve_config(cfg_file)` is used in upload.
   - Change needed: remove this from the user-facing Upload step. Keep config import only as an advanced/developer path if still needed.
   - File: `ui/app.py`.

3. Remove or rename "Data Configuration".
   - Current state: `Data Configuration` expander is in `ui/app.py` lines 906-944.
   - Change needed: do not remove required settings blindly. Instead, make timestamp/target/timezone simple visible controls with clearer labels, or auto-detect them and hide advanced naming.
   - File: `ui/app.py`.

## Alignment

1. Simplify "Merge strategy".
   - Current state: merge strategy selectbox is in `ui/app.py` lines 1214-1225; backend merge behavior is in `modules/alignment.py`.
   - Change needed: use reviewer-friendly labels such as "Exact timestamp match" and "Nearest timestamp match"; keep backend values `inner` and `nearest`. Add a one-line result note showing which strategy was used.
   - Files: `ui/app.py`, `modules/alignment.py`.

## EDA

1. Distribution chart color and label size.
   - Current state: distribution chart uses purple in `modules/eda.py` lines 99-111; UI chart rendering is `ui/app.py` lines 1302-1314.
   - Change needed: switch to clearer publication-style colors, increase chart title/axis/tick font sizes, and keep chart customization.
   - Files: `modules/eda.py`, `ui/app.py`.

2. Missing-values chart should use timestamp instead of row index.
   - Current state: `create_missing_value_heatmap()` uses `x=list(range(len(dataframe)))` and `x_label="Row Index"` in `modules/eda.py` lines 33-58.
   - Change needed: pass `timestamp_column` into the heatmap and use timestamps on the x-axis. Change label to "Timestamp"; update colors for present/missing.
   - Files: `modules/eda.py`, `ui/app.py`.

3. Time-series needs variable selection and chart customization.
   - Current state: EDA time-series tab renders a precomputed chart at `ui/app.py` lines 1336-1339. `modules/eda.py` supports optional `columns`, but UI does not expose it.
   - Change needed: add multiselect for numeric variables, then rebuild `create_time_series_figure(merged, ts_col, selected_columns)`. Add title and axis customization.
   - Files: `ui/app.py`, `modules/eda.py`.

4. Explain anomaly detection.
   - Current state: anomaly detection uses absolute z-score > 3 across numeric columns in `modules/eda.py` lines 65-80, but the UI does not explain this in the Anomalies tab around `ui/app.py` lines 1341-1351.
   - Change needed: add a short method note and threshold control, or rename to "Z-score anomaly check".
   - Files: `ui/app.py`, `modules/eda.py`.

## Feature Engineering

1. Reconsider rolling windows.
   - Current state: rolling windows are shown inside the Lag Features expander in `ui/app.py` lines 1443-1469, and defaults are enabled in `config/default.yaml`.
   - Change needed: either remove rolling windows from the main UI or move them to "Advanced". Default to empty/off unless the reviewer confirms they are needed.
   - Files: `ui/app.py`, `modules/feature_engineering.py`, `config/default.yaml`.

2. Clarify polynomial vs interaction features.
   - Current state: UI has both polynomial expansion and pairwise interaction controls in `ui/app.py` lines 1471-1491. Backend polynomial features already include cross-products in `modules/feature_engineering.py` lines 313-322.
   - Change needed: avoid duplicated interaction terms. Either separate "squared/cubic terms" from "pairwise interactions", or use `PolynomialFeatures(..., interaction_only=True)` only when intended.
   - Files: `ui/app.py`, `modules/feature_engineering.py`.

3. Fix time features.
   - Current state: time features are built in `modules/feature_engineering.py` lines 177-259 and appended in lines 324-327.
   - Change needed: move time features after normalization, keep raw ordinal features optional/off by default, prefer cyclic encodings for hour/day/year, and verify generated columns in the preview table.
   - Files: `modules/feature_engineering.py`, `ui/app.py`, `config/default.yaml`.

## Normalization

1. Explain the second table.
   - Current state: Normalization displays "Before" and "After" summary tabs plus the normalized dataset preview in `ui/app.py` lines 1609-1620.
   - Change needed: add headings/captions:
     - "Before normalization summary"
     - "After normalization summary"
     - "Normalized dataset preview"
     Also state that timestamp and target/reference columns are excluded.
   - Files: `ui/app.py`, `modules/normalization.py`.

## Modelling And Leaderboard

1. Clarify CV folds.
   - Current state: `CV folds` is a required number input from 2 to 10 in `ui/app.py` lines 1871-1880; feature subset uses "empty = use all" elsewhere around lines 2044-2061.
   - Change needed: rename to "Number of validation folds"; hide or disable it when validation method is Holdout. Add text: "This is not feature selection; it controls how many chronological validation splits are used."
   - Files: `ui/app.py`, `models/train.py`.

2. Add metric calculation explanation.
   - Current state: leaderboard appears in Modelling around `ui/app.py` lines 2079-2086 and Results around lines 2246-2255. Metrics are calculated in `evaluation/metrics.py` lines 72-105 and selected/ranked in `evaluation/comparator.py` lines 27-68.
   - Change needed: add a compact "How metrics are calculated" expander beside the leaderboard. State: RMSE, MAE, R2, MAPE, Bias, Pearson r, Slope, Intercept are calculated from validation predictions, not the full fitted-model predictions.
   - Files: `ui/app.py`, `evaluation/metrics.py`, `evaluation/comparator.py`.

## Model Comparison Charts

1. Make ideal and fit lines visible, using black/red.
   - Current state: 1:1 ideal line is gold and OLS fit is green in `modules/plots.py` lines 73-91; multi-model scatter uses similar colors around lines 208-222.
   - Change needed: use black for the ideal 1:1 line and red for the OLS fit line, thicken the lines, and ensure lines render above markers. For dark theme, either use a white plot background for these charts or use theme-aware colors.
   - Files: `modules/plots.py`, `ui/app.py`.

2. Fix PNG export.
   - Current state: chart downloads call `fig.to_image()` in `modules/download_helpers.py` lines 21-26, and `kaleido` is listed in `requirements.txt`.
   - Change needed: verify with `.venv\Scripts\python.exe`; if it fails because Kaleido cannot find Chrome, either document/run `plotly_get_chrome` or pin a compatible Kaleido version. Add a clearer UI error instead of only a disabled PNG button.
   - Files: `modules/download_helpers.py`, `requirements.txt`, possibly setup docs.

3. Multi-model time-series overlay needs controls and scope clarity.
   - Current state: overlay is generated by `modules/plots.py` lines 245-275 and displayed in `ui/app.py` lines 2144-2153 / 2314-2321.
   - Change needed: add controls for selected models, selected prediction scope (validation vs full fitted), selected date range, title/axis labels, and line colors. Label the chart explicitly as validation-only or full-data.
   - Files: `ui/app.py`, `modules/plots.py`, `models/train.py`.

4. Fix visual metric comparison title overlap.
   - Current state: grouped metrics chart is in `modules/plots.py` lines 282-314 and displayed in `ui/app.py` lines 2205-2213.
   - Change needed: increase top margin/height, reduce subplot title size, rotate/truncate model labels, and place legend below the chart.
   - Files: `modules/plots.py`, `ui/app.py`.

## Statistical Diagnostics

1. Check VIF table.
   - Current state: VIF is computed from all selected model features in `ui/app.py` lines 2357-2362 using `modules/diagnostics.py` lines 23-52.
   - Change needed: VIF can look wrong when duplicate/derived features are included, when normalized columns include constants, or when rows are dropped due to NaN/inf. Filter constant columns, show row count used, show warning for perfect multicollinearity/infinite VIF, and possibly compute VIF only on original predictor features by default.
   - Files: `modules/diagnostics.py`, `ui/app.py`.

## Residual Analysis

1. Scatter is enough.
   - Current state: residual analysis exposes plot style choices for several tabs in `ui/app.py` lines 2408-2480.
   - Change needed: remove `Plot style` selectboxes and default residual plots to scatter markers. Keep histogram/QQ only if required by diagnostics, otherwise simplify the tab set.
   - Files: `ui/app.py`, `modules/drift_analysis.py`.

## Export / Zenodo Ready Outputs

1. Reconsider "Zenodo-ready" wording.
   - Current state: export title is `Export - Zenodo-Ready Outputs` in `ui/app.py` line 2493; backend exports are in `modules/exporter.py`.
   - Change needed: rename user-facing step to "Export Outputs" unless Zenodo packaging is definitely required. Keep metadata as an optional artifact.
   - Files: `ui/app.py`, `modules/exporter.py`, README.

2. Include all columns in calibrated output.
   - Current state: export `calibrated` dataset only includes timestamp, target/reference value, and calibrated prediction in `ui/app.py` lines 2505-2512.
   - Change needed: export the full aligned/featured dataset plus prediction columns, or add two downloads:
     - compact calibrated output;
     - full calibrated output with all original/selected columns.
   - Files: `ui/app.py`, `modules/exporter.py`, `pipeline/run_pipeline.py`.

## Suggested Implementation Order

1. UX cleanup: Upload, EDA labels/colors, chart customization, metric explanation.
2. Workflow refactor: normalization before time features, simplify feature engineering.
3. Results cleanup: remove duplicated comparison views, clarify validation/full prediction scope.
4. Chart/export reliability: black/red ideal lines, PNG fix, metric title overlap, full-column export.
5. Diagnostics correctness: VIF filtering/warnings and residual-analysis simplification.
