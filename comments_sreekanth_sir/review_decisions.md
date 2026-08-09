# Review Decisions - Sreekanth Sir Feedback

Purpose: treat the handwritten comments as reviewer/user-experience signals, not literal technical instructions. The right response is to keep the scientifically necessary parts, simplify confusing wording, and make outputs clearer.

Decision labels:
- Accept: implement the feedback directly.
- Adapt: the feedback points to a real issue, but the implementation should be different.
- Defer: needs confirmation or user testing before changing.
- Do not implement: likely wrong or harmful as stated.

## Overall Decision

Do not blindly remove technical controls. Most comments show that the UI is exposing advanced pipeline concepts without enough context. The main product direction should be:

1. Make the default workflow simpler.
2. Move advanced controls into clearly named advanced sections.
3. Label metrics and plots by their data scope: validation predictions vs full fitted predictions.
4. Preserve reproducibility controls, but avoid making first-time users configure them.
5. Export both compact and full datasets.

## Decision Register

| Senior comment | Decision | Product interpretation | Recommended change |
|---|---:|---|---|
| "Calisense AQ - A complete solution..." | Accept | Product should be branded as `calisenseAQ`. | Update visible app branding, README title, page title, sidebar, and export metadata. |
| "What is this common timestamp?" | Accept | The upload copy is unclear to non-technical users. | Rename to "Shared timestamp column" and explain that both CSVs need the same date/time column for matching rows. Add auto-detection if feasible. |
| Remove "custom config"? | Adapt | Config upload is advanced and confusing in the first step. | Hide under "Advanced import settings" or remove from main Upload UI. Keep backend config support for reproducibility/export. |
| Remove "Data Configuration"? | Adapt | Required fields are presented like technical config. | Keep timestamp, target, and timezone controls, but rename the section to "Column setup" or auto-detected "Confirm columns". Do not remove required inputs entirely. |
| "Merge Strategy" | Adapt | The term is technical. | Rename choices to "Exact timestamp match" and "Nearest timestamp match"; keep backend values `inner` and `nearest`. |
| Distribution - change color, increase label size | Accept | Visual readability issue. | Use higher-contrast publication-style colors and larger title/axis/tick fonts. |
| Missing values - change color, row index -> timestamp | Accept | Current heatmap x-axis is not meaningful enough. | Use timestamp on x-axis and clearer present/missing colors. |
| Time series - option to choose variable + chart customization | Accept | Reviewer expects interactive selection. | Add multiselect for plotted variables and reuse chart title/axis customization. |
| What is "Anomaly detection"? | Adapt | Method is unexplained. | Rename to "Z-score anomaly check", show method summary and threshold, optionally expose threshold control. |
| Remove "Rolling windows"? | Adapt | Rolling features are useful but too advanced/noisy by default. | Move rolling controls under Advanced Feature Engineering. Consider defaulting rolling windows to empty/off. Do not delete backend support. |
| Polynomial expansion of interaction | Adapt | Current UI can create confusing/duplicated feature families. | Separate polynomial terms from interaction terms. Ensure polynomial expansion does not silently duplicate pairwise interaction columns. |
| Time features are not correct | Accept investigation, adapt implementation | Current raw time features can be scientifically questionable and are normalized before modelling. | Make time features optional/advanced, prefer cyclic encodings, move them after normalization, and preview generated columns clearly. |
| Normalization should be before time features | Accept | This is a valid pipeline-order issue. | Normalize sensor/engineered numeric features before adding timestamp-derived time features. Never normalize timestamp or target. |
| What is the second table? | Accept | Normalization output labels are unclear. | Label tables as "Before normalization summary", "After normalization summary", and "Normalized dataset preview". |
| CV folds. Empty = use all? | Do not implement as stated | CV folds cannot mean "use all"; that applies to feature selection, not validation splitting. | Rename to "Number of validation folds"; hide/disable it when Holdout validation is selected. Add explanatory help text. |
| Performance metrics calculation? Full model or validation results? | Accept | This is currently ambiguous. | Label leaderboard as validation metrics. Add "How metrics are calculated" section. Add validation/full fitted prediction toggle for plots where useful. |
| Ideal line is not seen. Use black/red. | Accept with theme handling | Line contrast is poor. | Use a high-contrast ideal line and red fit line. If using black, use a white chart background for those plots; otherwise use theme-aware contrast. |
| PNG not working | Accept | Export reliability issue. | Verify Kaleido/Chrome runtime. Show a clearer error and fix dependency/version/runtime setup. |
| Multi-model time-series overlay: customization, option to choose, full or validation? | Accept | Scope and controls are unclear. | Add model selection, date-range filtering, title/axis customization, and validation/full fitted scope label. |
| Visual metric comparison title overlapping | Accept | Layout bug. | Increase top margin/height, reduce subplot title size, move legend below, and handle long model names. |
| Validation & Results seems duplication | Accept | Results overlap with Modelling comparison. | Keep Modelling focused on training. Move detailed comparison to Validation & Results. |
| VIF table seems wrong | Accept investigation | VIF on engineered/duplicated features can produce misleading values. | Filter constants, warn on infinite VIF, show row count used, and default VIF to original predictors unless user selects engineered features. |
| Residual Analysis plot type: scatter is enough | Adapt | User does not need style choices. | Default residual plots to scatter and remove line/dotted style controls. Keep histogram/QQ as optional diagnostic tabs if needed. |
| Zenodo ready outputs needed? | Adapt | Zenodo is for demo/archival packaging, not the primary export workflow. | Rename to "Export Outputs"; keep Zenodo/demo metadata as optional archival export. |
| All columns please | Accept | Compact export drops useful context. | Add "Full calibrated dataset" download with all aligned/featured/original useful columns plus prediction columns. Keep compact export too. |

## Recommended Implementation Order

1. Low-risk UX cleanup: upload labels, merge labels, EDA colors/labels, normalization table captions, anomaly explanation.
2. Results clarity: leaderboard metric explanation, validation/full prediction labeling, remove duplicated comparison views.
3. Export and chart reliability: full-column export, PNG fix, metric chart overlap, ideal/fit line contrast.
4. Pipeline-order refactor: normalization before time features and time-feature defaults. This needs regression testing.
5. Diagnostics cleanup: VIF defaults/warnings and residual-analysis simplification.

## Decisions To Avoid

1. Do not remove timestamp/target/timezone setup completely; the pipeline needs those values.
2. Do not make CV folds "empty = use all"; that is conceptually wrong.
3. Do not remove rolling/polynomial/interaction code entirely unless the app is no longer intended for research use.
4. Do not hide whether charts use validation predictions or full fitted predictions.
5. Do not export only the compact calibrated file; users need source columns for review and reuse.
