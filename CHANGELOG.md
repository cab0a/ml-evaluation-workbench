# Changelog

All notable changes to this project are documented in this file.

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
