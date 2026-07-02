# v4.0 — UX & Transparency Release

**Date**: 2026-06-13  
**Status**: Previous *(superseded by [v5.0](v5_plan.md))*  
**Conversation**: `e248fc4f-1fae-479f-8014-40824b5f33c8`

---

## Goal

Transform the v3.0 calibration workbench into a **self-explanatory, transparent tool** that a scientist with no ML background can operate confidently — by adding contextual help at every decision point, exposing how each model works under the hood, expanding imputation options, and fixing navigation bugs.

---

## User Requirements (original)

> 1. Small note on everything — tooltips/captions on options and dropdowns explaining what selecting each option does.
> 2. Explain lag features properly — what they are, what they do, how to use them correctly for air quality data.
> 3. After model training, a dropdown for comparing model results with each other.
> 4. Missing value strategies should include `ffill`, `interpolate`, and `bfill` as well.
> 5. User can input lag steps freely (large lags like 3h don't make sense for AQ data).
> 6. Model comparison should support 2, 3, 4 or more models simultaneously.
> 7. Show what hyperparameters are available, what inputs each model needs, and how each model works — with formulas where applicable.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `help=` tooltips everywhere | Streamlit's native tooltip system — shows on hover, non-intrusive, no extra UI space |
| `st.caption()` at expander tops | Context-setting sentence before the user sees any widget — reduces cognitive load |
| LaTeX formulas in Model Reference Guide | `st.latex()` renders properly in Streamlit; makes the maths accessible without leaving the app |
| Lag warning threshold: `max_lag × 20 > n_rows` | Conservative heuristic: at minimum you want 20 usable rows per lag step to avoid a degenerate dataset |
| Multi-model comparison via multiselect (not selectboxes) | Scales to any number of models without code changes; defaults to all trained models |
| Next button guard: `disabled=(out is None)` | Prevents users from proceeding before training completes; button is always visible (not conditional) |
| 4 imputation strategies exposed in UI | `interpolate_ffill` (recommended) + `ffill` + `bfill` + `interpolate`; all validated in backend |

---

## Changes Made

### `modules/preprocessing.py`

#### `clean_missing_values()`
- Added `"interpolate"` strategy — linear interpolation only, no fill fallback
- Added `"bfill"` strategy — backward-fill then forward-fill for edge NaNs
- Updated docstring to document all 4 strategies with caveats

---

### `ui/app.py`

#### Step 1 — Upload Data
- All 3 `st.file_uploader()` widgets now have `help=` explaining CSV format, column requirements, and timestamp format
- `st.checkbox("Use bundled sample datasets")` has tooltip explaining when to uncheck
- Data Configuration expander: `st.caption()` added; timestamp col, target col, timezone all have tooltips
- Load button has tooltip

#### Step 2 — Preprocessing
- Added `st.caption()` to Sensor and Reference expanders
- Missing value strategy dropdown: all 4 options (`interpolate_ffill`, `interpolate`, `ffill`, `bfill`) exposed with descriptive `help=`
- Outlier method and threshold inputs annotated
- Reference preprocessing: `st.info()` warning about rarely needing outlier removal

#### Step 3 — Alignment
- Added `st.caption()` explaining alignment purpose
- Resample frequency, aggregation, merge strategy, max lag all have detailed `help=` tooltips

#### Step 5 — Modelling (Feature Engineering expanders)

| Expander | What was added |
|----------|---------------|
| **Lag & Rolling** | Full caption explaining "N steps ago" with hourly PM2.5 examples; `help=` on all 3 inputs; dynamic `st.warning()` when `max_lag × 20 > n_rows` |
| **Polynomial & Interaction** | Caption explaining when linear models benefit; `help=` on degree, poly cols, interaction cols |
| **Time & Date Features** | Caption on diurnal/seasonal patterns; `help=` on every checkbox (cyclical, season, unix, etc.) |
| **Training Settings** | Caption; `help=` on test split (chronological note), CV folds (TimeSeriesSplit note), model list (description of each) |
| **Manual Hyperparameters** | Caption; `help=` on all 7 parameters (alpha, n_estimators, max_depth, learning_rate) with typical ranges |
| **Auto-Tuning** | Expanded caption + `st.info()` on n_iter guidance; `help=` on each tune checkbox and n_iter input |

#### Step 5 — New: 📚 Model Reference Guide
New collapsible expander with **5 tabs** (one per model), each containing:
- Plain-English description of the algorithm
- `st.latex()` formula (prediction equation + objective/loss)
- `st.info()` block: what inputs it needs, scaling requirements, PM calibration guidance
- Markdown table: **all** hyperparameters with default, UI-exposure status (✅/❌), and description

#### Step 5 — Enhanced: 🔀 Compare Models Side-by-Side
Replaced fixed 2-model comparator with multiselect of N models:
- **Scatter Grid tab**: panels in rows of 2, dynamically sized
- **Time-Series Overlay tab**: all selected models on one chart
- **Metrics Table tab**: rows = metrics with ↑/↓/→ target direction labels; best value per row highlighted green; bar chart below table

---

## Bug Fixes

### `KeyError: 'model'` in comparison leaderboard filter
- **Root cause**: Leaderboard column is `model_name` (defined in `evaluation/comparator.py` `LEADERBOARD_COLUMNS`), not `model`
- **Fix**: `out["leaderboard"]["model_name"].isin(...)` in comparison metrics bar chart

### Modelling Next button loops back
- **Root cause**: `if st.button("Next ➡️", ...)` was indented 8 spaces — inside `if out is not None:` — so `go_next()` → `st.rerun()` re-entered the same conditional block and never advanced `current_step`
- **Fix**: Moved button to function body (4-space indent), outside the conditional; added `disabled=(out is None)` guard

### Upload page icon text overlap
- **Root cause**: Streamlit file uploader uses Material Icons font for the upload arrow; when the font fails to load (common on Windows), the ligature text `"upload"` renders as literal characters overlapping the button label
- **Fix**: CSS targeting `[data-testid="stFileUploader"] button span span span { font-size: 0 !important; }` + `::before { content: '⬆'; }` replacement

---

## Smoke Test Results (2026-06-13)

```
SMOKE TEST REPORT
=======================================================
  PASS  Imports OK
  PASS  ref=168 rows  lcs=168 rows
  PASS  All 4 impute strategies pass
  PASS  S1 load OK
  PASS  S2 preprocess OK
  PASS  S3 alignment OK  n=165 rows
  PASS  S4 EDA OK
  PASS  S5 train OK  best=lasso
  PASS  Leaderboard cols: rank, model_name, rmse, mae, r2, ...
  PASS  model_name column OK
  PASS  Leaderboard filter OK

TOTAL: 11 passed  0 failed
```

---

## Files Modified

| File | Type | Summary |
|------|------|---------|
| `ui/app.py` | Modified | All UX additions: tooltips, captions, model reference guide, multi-model comparator, Next button fix, CSS icon fix |
| `modules/preprocessing.py` | Modified | Added `bfill` and `interpolate` strategies to `clean_missing_values()` |

---

*Previous version plan: [v3_plan.md](v3_plan.md)*
