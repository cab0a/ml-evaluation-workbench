# Changelog

All notable changes to this project are documented in this file.

## [0.8.0] - 2026-07-28

### Added

- Fixed 25-row summary of representative paired contrasts across six existing
  experiment families
- Direction-aligned preferred effects that preserve the raw paired difference
  while accounting for higher- and lower-is-better metrics
- Macro-F1-only cross-experiment figure linked to the complete source
  artifacts
- Machine-readable interface contract for the CLI, Python API, report schema,
  supported Python versions, and generated artifacts

### Changed

- Evaluation report schema updated to version 8
- Reproduction manifest expanded to twenty-seven reference artifacts
- Documentation reorganized around evidence navigation, interface review, and
  cross-metric interpretation boundaries
- Generated-artifact inventory centralized so the CLI, demo verification, and
  interface contract share one source of truth

## [0.7.0] - 2026-07-28

### Added

- Controlled downsampling of the globally least frequent class at 100%, 75%,
  50%, and 25% retention inside each outer training fold
- Deterministic retained-row samples shared by logistic regression and fixed
  KNN while validation data and labels remain unchanged
- Fold-level seeds, retained-row signatures, before/after class shares,
  post-sampling class counts, aggregate metrics, and per-class recall
- Class-imbalance summary, diagnostics, and sensitivity figure covering macro
  F1 and target-class recall

### Changed

- Evaluation report schema updated to version 7
- Reproduction manifest expanded to twenty-four reference artifacts
- Documentation expanded with training-prevalence sensitivity results,
  non-monotonic observations, and mitigation and deployment boundaries

## [0.6.0] - 2026-07-27

### Added

- Validation-only missing-value injection at 0%, 10%, 25%, and 50% of
  observed feature cells
- Validation-only zero-mean Gaussian noise at 0, 0.25, 0.5, and 1.0 times
  each outer training fold's feature standard deviation
- Shared perturbations across logistic regression and fixed KNN on the
  existing five outer folds
- Fold-level cell accounting, deterministic perturbation seeds, actual noise
  scales, aggregate metrics, and paired differences from unperturbed scores
- Robustness summary, diagnostics, and macro-F1 sensitivity figure

### Changed

- Evaluation report schema updated to version 6
- Reproduction manifest expanded to twenty reference artifacts
- Test evaluation results reused through a session fixture to keep the
  expanded deterministic suite efficient
- Documentation expanded with controlled-sensitivity results and a clear
  boundary between synthetic perturbations and deployment robustness

## [0.5.0] - 2026-07-27

### Added

- Shared-outer-fold comparison of uncalibrated and sigmoid-calibrated
  probabilities for logistic regression and the fixed KNN comparator
- Training-only three-fold calibration inside every outer training partition
- Fold-level accuracy, multiclass log loss, multiclass Brier score, and
  ten-bin top-label expected calibration error
- Cross-fitted class probabilities with source-row references
- Reliability-bin evidence, compact summaries, paired method differences, and
  a top-label reliability figure

### Changed

- Evaluation report schema updated to version 5
- Reproduction manifest expanded to sixteen reference artifacts
- CLI extended with an explicit inner calibration-fold parameter
- Documentation expanded with metric-dependent calibration results and
  probability-quality limitations

## [0.4.0] - 2026-07-23

### Added

- Shared-fold feature ablation for bill length only, bill depth only, and both
  measurements across logistic regression and KNN
- Fold-level and summary ablation CSV artifacts with paired differences from
  the two-feature reference
- Feature-ablation macro-F1 visualization
- Train/validation overlap, validation coverage, and partition-size checks
- Within-fold shuffled-training-label negative control with observed-versus-
  shuffled score differences
- Focused leakage-diagnostic JSON and fold-level CSV artifacts

### Changed

- Evaluation report schema updated to version 4
- Reproduction manifest expanded to eleven reference artifacts
- Documentation expanded with diagnostic limitations and a non-causal
  interpretation boundary

## [0.3.0] - 2026-07-23

### Added

- Fixed 5-nearest-neighbors nonlinear comparator using the same features,
  preprocessing, holdout, and cross-validation folds as existing models
- Model configuration metadata and paired holdout and fold-level differences
  for all controlled comparisons
- Compact `model_comparison.csv` summary spanning holdout and
  cross-validation metrics
- KNN holdout predictions, correctness flags, fold-level scores, tests, and
  CLI summaries

### Changed

- Evaluation report schema updated to version 3
- Cross-validation figure expanded to show all three models
- Documentation expanded with a no-tuning boundary and fold-level comparison
  interpretation

## [0.2.0] - 2026-07-22

### Added

- Five-fold stratified cross-validation shared by the dummy and
  logistic-regression pipelines
- Fold-level CSV evidence with aggregate metrics, per-class recall, and
  partition sizes
- Cross-validation summaries with mean, population standard deviation,
  minimum, maximum, and paired model gains
- Cross-validation score visualization and checksum verification
- CLI control for the fold count and tests for deterministic fold evidence

### Changed

- Evaluation report schema updated to version 2
- Documentation expanded to separate fold variability from confidence
  intervals and generalization claims

## [0.1.0] - 2026-07-22

### Added

- Pinned Palmer Penguins dataset with CC0 provenance and SHA-256 verification
- Deterministic stratified holdout evaluation using bill length and bill depth
- Majority-class dummy baseline and logistic-regression prototype
- Training-only median imputation and standardization in scikit-learn pipelines
- Accuracy, balanced accuracy, macro F1, per-class recall, predictions, and
  confusion-matrix artifacts
- CLI, focused tests, reproducible demo, checksum manifest, and CI
