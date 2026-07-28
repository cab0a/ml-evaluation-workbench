# Configuration and Interfaces

This document defines the CLI commands, evaluation defaults, fixed model and
experiment settings, Python API, and exit behavior for ML Evaluation Workbench
1.0.0. The [README](../README.md) contains the minimal runnable workflow.

## Requirements

- Python 3.10 or later
- `matplotlib>=3.9`
- `numpy>=1.26`
- `pandas>=2.2`
- `scikit-learn>=1.5`

The byte-exact reference environment uses Python 3.12 on Ubuntu with
[`requirements-reproducibility.txt`](../requirements-reproducibility.txt).
The declared dependency ranges are exercised separately on Python 3.10 through
3.14.

## CLI

```text
ml-evaluation-workbench evaluate DATASET [--output-dir DIR]
                                         [--random-state INTEGER]
                                         [--test-size FRACTION]
                                         [--cv-folds INTEGER]
                                         [--calibration-folds INTEGER]

ml-evaluation-workbench verify ARTIFACT_DIR
```

### `evaluate`

```bash
ml-evaluation-workbench evaluate data/penguins.csv \
  --output-dir results \
  --random-state 42 \
  --test-size 0.25 \
  --cv-folds 5 \
  --calibration-folds 3
```

| Argument | Default | Meaning |
| --- | ---: | --- |
| `DATASET` | required | Palmer Penguins CSV path |
| `--output-dir` | `results` | Directory that receives the generated report files |
| `--random-state` | `42` | Holdout, outer-fold, shuffle, perturbation, and sampling seed basis |
| `--test-size` | `0.25` | Stratified holdout fraction; must be between 0 and 1 |
| `--cv-folds` | `5` | Stratified outer folds; at least 2 and no greater than the smallest class count |
| `--calibration-folds` | `3` | Inner training-only calibration folds; at least 2 and feasible in every outer training partition |

The command writes 27 documented evaluation artifacts followed by
`checksums.sha256`. It never modifies the source dataset.

### `verify`

```bash
ml-evaluation-workbench verify results
```

`verify` requires the complete documented artifact set, rejects missing or
unexpected manifest entries, recalculates every SHA-256, and returns the
verified artifact count.

Expected summary:

```text
Verified: 27 artifacts
Manifest: results/checksums.sha256
```

### Exit Codes

| Code | Meaning |
| ---: | --- |
| `0` | Evaluation or verification completed successfully |
| `2` | Argument, input, validation, manifest, or output error prevented completion |

## Fixed Model Configuration

| Model | Role | Configuration |
| --- | --- | --- |
| Majority-class dummy | Reference baseline | `DummyClassifier(strategy="most_frequent")` |
| Logistic regression | Linear baseline | `LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)` |
| 5-nearest neighbors | Nonlinear comparator | `n_neighbors=5`, uniform weights, `algorithm="auto"`, `leaf_size=30`, Minkowski `p=2` |

Every model is wrapped in a pipeline with median imputation and standard
scaling. No parameter tuning or model selection is performed.

## Fixed Experiment Settings

| Setting | Values |
| --- | --- |
| Features | `bill_length_mm`, `bill_depth_mm` |
| Target | `species` |
| Feature sets | Bill length only, bill depth only, both measurements |
| Aggregate metrics | Accuracy, balanced accuracy, macro F1 |
| Probability methods | Uncalibrated, sigmoid |
| Probability metrics | Accuracy, log loss, multiclass Brier, top-label ECE |
| Reliability bins | 10 equal-width top-label confidence bins |
| Missing-value rates | 0%, 10%, 25%, 50% |
| Noise multipliers | 0, 0.25, 0.5, 1.0 training-fold standard deviations |
| Class-retention fractions | 100%, 75%, 50%, 25% |
| Cross-experiment schema | Version 1 fixed representative policy |

## Python API

The supported package-root API is:

```python
from ml_evaluation_workbench import (
    DATASET_SHA256,
    DATASET_URL,
    EvaluationResult,
    __version__,
    download_dataset,
    evaluate_dataset,
    load_dataset,
    sha256_file,
    verify_artifact_manifest,
    verify_dataset,
)
```

The main evaluation call is:

```python
result = evaluate_dataset(
    frame,
    random_state=42,
    test_size=0.25,
    cv_folds=5,
    calibration_folds=3,
)
```

`frame` is the only positional parameter. The four evaluation settings are
keyword-only. `verify_artifact_manifest(output_dir)` returns the number of
verified artifacts.

`EvaluationResult` exposes:

- `metrics`
- `predictions`
- `cross_validation_folds`
- `model_comparison`
- `feature_ablation_folds`
- `feature_ablation_summary`
- `leakage_diagnostic_folds`
- `leakage_diagnostics`
- `probability_calibration_folds`
- `probability_calibration_summary`
- `probability_calibration_predictions`
- `probability_calibration_bins`
- `robustness_folds`
- `robustness_summary`
- `robustness_diagnostics`
- `class_imbalance_folds`
- `class_imbalance_summary`
- `class_imbalance_diagnostics`
- `cross_experiment_summary`
- `confusion`
- `labels`

The package exports, required parameters, defaults, and result fields are
checked by stable-interface regression tests and recorded in
[`interface_contract.json`](../results/interface_contract.json).

## Controlled CLI Summary

The committed input and default settings report:

```text
Dataset rows: 344
Training rows: 258
Test rows: 86
Dummy accuracy: 0.442
Logistic regression accuracy: 0.942
Logistic regression macro F1: 0.924
KNN accuracy: 0.965
KNN macro F1: 0.960
Cross-validation folds: 5
Logistic regression CV macro F1 mean: 0.928
Logistic regression CV macro F1 std: 0.041
KNN CV macro F1 mean: 0.949
KNN CV macro F1 std: 0.018
Logistic regression shuffled-label macro F1 mean: 0.254
KNN shuffled-label macro F1 mean: 0.298
Split integrity check: passed
logistic_regression CV log loss, uncalibrated: 0.139
logistic_regression CV log loss, sigmoid calibrated: 0.236
knn CV log loss, uncalibrated: 0.280
knn CV log loss, sigmoid calibrated: 0.133
Cross-experiment contrasts: 25
```

The command then prints the path of every generated artifact. Console wording
is explanatory; automated consumers should use the JSON/CSV artifacts and
machine-readable interface contract.
