# ML Evaluation Workbench

## 日本語概要

このリポジトリは、固定した公開データと共通splitを使い、dummy、logistic
regression、5-nearest neighborsを比較するML評価プロジェクトです。モデル評価の
設計や結果の監査方法を確認したいMLエンジニアに役立ちます。

fold-level metrics、row-level predictions、feature ablation、split integrity、
shuffled-label negative controlに加え、probability calibration、validation-onlyの
missing-value injection、Gaussian feature noiseによる感度評価を含みます。前処理は
各training partition内でfitされ、数値成果物はSHA-256で検証できます。結果の適用範囲と
制約の詳細は英語本文を参照してください。

---

[![CI](https://github.com/cab0a/ml-evaluation-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/cab0a/ml-evaluation-workbench/actions/workflows/ci.yml)

Compare machine-learning baselines through shared splits, leakage-aware
pipelines, inspectable predictions, and reproducible evaluation artifacts.

## Overview

ML Evaluation Workbench demonstrates a small but complete evaluation cycle:

**Question → Controlled Comparison → Diagnostics → Interpretation**

Version 0.6.0 adds controlled sensitivity experiments for injected missing
values and Gaussian feature noise. Validation features are perturbed on the
same outer folds while training data, labels, model settings, and preprocessing
policy remain fixed.

The repository emphasizes evaluation design and reproducibility rather than
model complexity or leaderboard performance.

It is intended for ML engineers and reviewers who want a compact example of
how to structure a controlled model comparison. Unlike the image-processing
experiments elsewhere in this portfolio, this repository focuses on tabular
classification evidence: baselines, split policy, fold variability, feature
ablation, negative controls, and row-level errors.

## Representative Result

The same five stratified folds are used to compare each selected feature set.
Both substantive models lose at least 0.284 mean macro F1 when either bill
measurement is removed from the shared two-feature reference.

![Feature-ablation macro F1](results/feature_ablation_scores.png)

The per-fold values and paired differences are available in
[`feature_ablation_folds.csv`](results/feature_ablation_folds.csv) and
[`feature_ablation_summary.csv`](results/feature_ablation_summary.csv). This
supports a bounded complementary-signal claim for the fixed dataset, models,
and folds—not a causal feature-importance claim.

## Problem

A model score has little meaning without a reference baseline, a clearly
defined split, leakage-aware preprocessing, class-sensitive metrics, and
inspectable predictions. Small demonstrations often omit one or more of these
elements and make the result difficult to reproduce or interpret.

This project keeps those decisions explicit:

- The dataset bytes and upstream revision are fixed.
- The dummy model establishes a minimum reference.
- Imputation and scaling are fitted only on the training partition.
- Accuracy is reported alongside balanced accuracy, macro F1, and per-class
  recall.
- Holdout metrics, fold-level evidence, row-level predictions, and figures are
  committed as reviewable artifacts.

## Key Features

- Pinned, checksum-verified public dataset
- Deterministic stratified train/test split
- Five-fold stratified cross-validation shared by all three models
- Majority-class `DummyClassifier` baseline
- Median imputation and standardization inside a scikit-learn `Pipeline`
- Logistic-regression classifier using two interpretable measurements
- Fixed 5-nearest-neighbors nonlinear comparator without parameter tuning
- Compact holdout and cross-validation model-comparison table
- Three-way feature ablation on shared folds
- Per-fold and summary ablation artifacts with paired differences
- Train/validation overlap and validation-coverage diagnostics
- Within-fold shuffled-training-label negative control
- Cross-fitted uncalibrated and sigmoid-calibrated probabilities
- Log loss, multiclass Brier score, and top-label calibration error
- Reliability-bin CSV evidence and a calibration figure
- Validation-only missing-value injection at four fixed rates
- Gaussian feature noise scaled from each outer training fold
- Fold-level perturbation accounting and robustness summaries
- Accuracy, balanced accuracy, macro F1, and per-class recall
- Cross-validation mean, population standard deviation, minimum, and maximum
- Row-level holdout predictions with source-row references
- Fold-level CSV evidence and a cross-validation score figure
- Confusion-matrix image for the deterministic holdout
- CLI with explicit split parameters
- Atomic JSON, CSV, and PNG artifact replacement
- Focused tests and GitHub Actions for Python 3.10 through 3.14

## Quick Start

Python 3.10 or later is required.
On Debian or Ubuntu, install `python3-venv` if `venv` reports that `ensurepip`
is unavailable.

```bash
git clone https://github.com/cab0a/ml-evaluation-workbench.git
cd ml-evaluation-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the committed dataset and run the evaluation:

```bash
python examples/download_penguins.py --check
ml-evaluation-workbench evaluate data/penguins.csv \
  --output-dir output/quickstart
```

The evaluation writes twenty artifacts under `output/quickstart/`. Start with
`model_comparison.csv` for a compact model-level view,
`feature_ablation_scores.png` for the representative comparison, and
`predictions.csv` for row-level error inspection. Checksum-fixed reference
copies are committed under `results/`.

## Usage

```text
ml-evaluation-workbench evaluate DATASET [--output-dir DIR]
                                         [--random-state INTEGER]
                                         [--test-size FRACTION]
                                         [--cv-folds INTEGER]
                                         [--calibration-folds INTEGER]
```

The documented v0.6 command is:

```bash
ml-evaluation-workbench evaluate data/penguins.csv \
  --output-dir results \
  --random-state 42 \
  --test-size 0.25 \
  --cv-folds 5 \
  --calibration-folds 3
```

Expected summary:

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
logistic_regression macro F1 at 50% injected missingness: 0.624
logistic_regression macro F1 at 1.0x feature noise: 0.692
knn CV log loss, uncalibrated: 0.280
knn CV log loss, sigmoid calibrated: 0.133
knn macro F1 at 50% injected missingness: 0.625
knn macro F1 at 1.0x feature noise: 0.673
Metrics: results/metrics.json
Predictions: results/predictions.csv
Confusion matrix: results/confusion_matrix.png
Cross-validation fold scores: results/cross_validation_folds.csv
Cross-validation scores: results/cross_validation_scores.png
Model comparison: results/model_comparison.csv
Feature ablation folds: results/feature_ablation_folds.csv
Feature ablation summary: results/feature_ablation_summary.csv
Feature ablation scores: results/feature_ablation_scores.png
Leakage diagnostic folds: results/leakage_diagnostic_folds.csv
Leakage diagnostics: results/leakage_diagnostics.json
Probability calibration folds: results/probability_calibration_folds.csv
Probability calibration summary: results/probability_calibration_summary.csv
Probability calibration predictions: results/probability_calibration_predictions.csv
Probability calibration bins: results/probability_calibration_bins.csv
Probability calibration plot: results/probability_calibration.png
Robustness folds: results/robustness_folds.csv
Robustness summary: results/robustness_summary.csv
Robustness diagnostics: results/robustness_diagnostics.json
Robustness plot: results/robustness.png
```

## Technical Design

### Dataset

The repository includes the simplified Palmer Penguins dataset maintained by
Allison Horst, Alison Hill, and Kristen Gorman. It contains 344 rows and eight
columns. The data are available under CC0 1.0.

- [Dataset documentation](https://allisonhorst.github.io/palmerpenguins/)
- [Dataset license](https://allisonhorst.github.io/palmerpenguins/LICENSE.html)
- [Pinned provenance and checksum](data/README.md)

Only `bill_length_mm` and `bill_depth_mm` are used as model inputs. Island,
sex, body mass, flipper length, and observation year are intentionally excluded
from v0.6. This keeps the question interpretable and avoids relying on
location-specific correlations that can make a random holdout unnecessarily
easy.

## Evaluation Methodology

1. Load the pinned CSV and preserve its source-row index.
2. Create a 75/25 stratified holdout split with random state 42.
3. Fit median imputation on the training bill measurements.
4. Fit standardization on the imputed training measurements.
5. Train a majority-class dummy, logistic regression, and 5-nearest-neighbors
   classifier using equivalent input rows.
6. Keep the KNN configuration fixed at five neighbors, uniform weighting, and
   Euclidean distance; do not select it from these evaluation scores.
7. Evaluate all three models on the untouched holdout partition.
8. Run five-fold stratified cross-validation, using the same shuffled folds
   for all models and refitting each complete pipeline inside every fold.
9. Summarize each metric with its mean, population standard deviation,
   minimum, and maximum across the five observed folds.
10. Evaluate bill length alone, bill depth alone, and both measurements for
    logistic regression and KNN on the same folds.
11. Verify zero train/validation row overlap and exactly-once validation
    coverage across the folds.
12. Shuffle only the training labels inside each fold, refit both substantive
    models, and compare the negative-control scores with observed scores.
13. Compare uncalibrated probabilities with sigmoid calibration on the same
    outer folds. Fit each calibrator with three stratified folds drawn only
    from the corresponding outer training partition.
14. Record fold-level log loss, multiclass Brier score, top-label expected
    calibration error, cross-fitted probabilities, and reliability bins.
15. Inject missing values into 0%, 10%, 25%, and 50% of previously observed
    validation feature cells using deterministic seeds shared by both models.
16. Add zero-mean Gaussian noise to observed validation cells at 0, 0.25,
    0.5, and 1.0 times each outer training fold's feature standard deviation.
17. Compare every perturbed score with the unperturbed score on the same fold.
18. Save aggregate metrics, fold-level evidence, diagnostic summaries,
    predictions, and evaluation figures.

The preprocessing steps are part of each scikit-learn `Pipeline`, so their
statistics are not estimated from the holdout partition or a fold's validation
partition.

## Generated Artifacts

`metrics.json` contains:

- `report_version` and `project_version`
- dataset path, SHA-256, row count, classes, selected features, and missing
  feature-cell count
- split strategy, seed, fraction, and train/test row counts
- classifier configuration, accuracy, balanced accuracy, macro F1, and
  per-class recall for each model
- holdout differences for logistic regression versus dummy, KNN versus dummy,
  and KNN versus logistic regression
- cross-validation strategy and fold count
- per-model cross-validation mean, population standard deviation, minimum, and
  maximum
- paired fold-level differences for all three controlled comparisons
- feature-ablation configuration, summaries, and paired differences from the
  two-feature reference
- split-integrity checks and shuffled-training-label negative-control summaries
- probability-calibration design, metric definitions, method summaries, and
  paired fold-level differences
- robustness protocol, perturbation definitions, seed rules, cell accounting,
  condition summaries, and paired differences from unperturbed scores

`predictions.csv` contains:

- original one-based CSV source row, including the header offset
- actual class
- dummy, logistic-regression, and KNN predictions
- correctness flags for each model

`cross_validation_folds.csv` contains one row per model and fold, including
train and validation row counts, the three aggregate metrics, and per-class
recall.

`model_comparison.csv` provides one compact row per model with its evaluation
role, holdout metrics, and cross-validation means and standard deviations.

`feature_ablation_folds.csv` records 30 model-feature-fold evaluations.
`feature_ablation_summary.csv` provides one row per model and feature set,
including paired mean differences from the two-feature reference.
`feature_ablation_scores.png` visualizes macro F1 and observed fold
variation.

`leakage_diagnostic_folds.csv` records the observed and shuffled-label scores
for each substantive model and fold. `leakage_diagnostics.json` records split
integrity, validation coverage, negative-control summaries, and the diagnostic
interpretation boundary.

`probability_calibration_folds.csv` records accuracy, log loss, multiclass
Brier score, and top-label expected calibration error for each model, method,
and outer fold. `probability_calibration_summary.csv` provides compact means
and fold standard deviations. `probability_calibration_predictions.csv`
contains one cross-fitted probability vector per source row, model, and
method. `probability_calibration_bins.csv` records the ten equal-width
top-label reliability bins used by `probability_calibration.png`.

`robustness_folds.csv` records one row per perturbation, severity, fold, and
model, including affected-cell counts, realized fractions, noise scales, and
classification metrics. `robustness_summary.csv` provides fold summaries and
paired mean differences from the unperturbed condition.
`robustness_diagnostics.json` fixes the perturbation protocol and
interpretation boundary. `robustness.png` visualizes macro-F1 sensitivity.

`confusion_matrix.png` visualizes the logistic-regression holdout errors.
`cross_validation_scores.png` shows all three models' aggregate scores in
each fold. `checksums.sha256` fixes the bytes of all twenty reference
artifacts.

## Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.442 | 0.333 | 0.204 |
| Logistic regression | 0.942 | 0.920 | 0.924 |
| 5-nearest neighbors | 0.965 | 0.959 | 0.960 |

KNN correctly classifies 83 of 86 holdout rows, compared with 81 for logistic
regression. KNN macro F1 is 0.036 higher on this holdout. Its three errors are
one Chinstrap predicted as Gentoo and two Gentoo observations predicted as
Adelie and Chinstrap.

The logistic-regression confusion matrix is retained for continuity with the
earlier baseline; model-specific correctness is available in
`predictions.csv`.

![Logistic regression confusion matrix](results/confusion_matrix.png)

### Five-Fold Cross-Validation

| Model | Accuracy, mean ± std | Balanced Accuracy, mean ± std | Macro F1, mean ± std |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.442 ± 0.006 | 0.333 ± 0.000 | 0.204 ± 0.002 |
| Logistic regression | 0.945 ± 0.034 | 0.924 ± 0.043 | 0.928 ± 0.041 |
| 5-nearest neighbors | 0.959 ± 0.011 | 0.948 ± 0.023 | 0.949 ± 0.018 |

KNN's mean paired macro-F1 difference from logistic regression is +0.021, but
the fold-level range is -0.011 to +0.108. KNN is more stable in these five
folds and avoids logistic regression's fold-4 drop, while logistic regression
slightly leads in another fold. The controlled result supports a useful
nonlinear comparison, not universal KNN superiority.

![Cross-validation fold scores](results/cross_validation_scores.png)

### Feature Ablation

| Feature set | Logistic Macro F1 | KNN Macro F1 |
| --- | ---: | ---: |
| Bill length only | 0.575 ± 0.012 | 0.647 ± 0.073 |
| Bill depth only | 0.569 ± 0.024 | 0.665 ± 0.050 |
| Both measurements | 0.928 ± 0.041 | 0.949 ± 0.018 |

Removing either measurement reduces mean macro F1 by at least 0.284 for KNN
and 0.353 for logistic regression relative to the shared two-feature
reference. Under these fixed models and folds, the measurements provide
complementary predictive signal.

### Leakage Diagnostics

- Maximum train/validation overlap: 0 rows
- Validation coverage: exactly once for every row
- Logistic-regression shuffled-label macro F1: 0.254 ± 0.052
- KNN shuffled-label macro F1: 0.298 ± 0.054
- Observed-minus-shuffled macro-F1 mean: 0.674 for logistic regression and
  0.651 for KNN

The negative-control scores are substantially below the observed scores. This
is consistent with the models using the intended feature-label association,
but it does not prove that every possible source of leakage is absent.

### Probability Calibration

| Model and method | Accuracy | Log loss | Multiclass Brier | Top-label ECE |
| --- | ---: | ---: | ---: | ---: |
| Logistic, uncalibrated | 0.945 | 0.139 | 0.072 | 0.064 |
| Logistic, sigmoid | 0.936 | 0.236 | 0.117 | 0.114 |
| KNN, uncalibrated | 0.959 | 0.280 | 0.056 | 0.030 |
| KNN, sigmoid | 0.965 | 0.133 | 0.057 | 0.071 |

Sigmoid calibration reduces KNN mean log loss by 0.148 but slightly increases
its Brier score and increases top-label ECE. For logistic regression, all
three recorded probability-quality metrics worsen. The methods therefore
produce a metric-dependent result rather than evidence that calibration is
uniformly beneficial. The reliability points are descriptive summaries of
the 344 cross-fitted predictions per model and method.

![Top-label reliability diagram](results/probability_calibration.png)

### Missing-Value and Noise Robustness

| Perturbation | Severity | Logistic Macro F1 | KNN Macro F1 |
| --- | ---: | ---: | ---: |
| Injected missing values | 0% | 0.928 | 0.949 |
| Injected missing values | 10% | 0.845 | 0.870 |
| Injected missing values | 25% | 0.770 | 0.759 |
| Injected missing values | 50% | 0.624 | 0.625 |
| Gaussian feature noise | 0.00x | 0.928 | 0.949 |
| Gaussian feature noise | 0.25x | 0.910 | 0.920 |
| Gaussian feature noise | 0.50x | 0.860 | 0.839 |
| Gaussian feature noise | 1.00x | 0.692 | 0.673 |

At the highest tested severities, logistic regression loses 0.304 macro F1
under missingness and 0.237 under noise; KNN loses 0.324 and 0.277. These are
paired mean differences under fixed synthetic perturbations, not estimates of
performance under a real acquisition failure or production drift.

![Missing-value and noise robustness](results/robustness.png)

See [results/README.md](results/README.md) for interpretation and the boundary
between this controlled result and a general performance claim.

## Limitations

- Version 0.6.0 evaluates one small dataset with one deterministic holdout and
  one five-fold stratified cross-validation run.
- A random row split does not measure transfer across islands, years, field
  conditions, or independent collection programs.
- The simplified dataset does not provide a grouping identifier suitable for
  testing repeated observations of the same individual.
- Four selected feature cells are missing and use training-partition median
  imputation. Other missingness mechanisms are not evaluated.
- The two-feature design is intentionally constrained and does not establish
  an optimal feature set.
- The three fixed models are controlled comparators, not a comprehensive model
  search.
- The KNN configuration is chosen in advance and is not claimed to be optimal.
  Different neighbor counts, weights, or distance metrics are not evaluated.
- Ablation differences are descriptive for the two selected measurements and
  do not establish causal feature importance.
- Reusing one cross-validation partition limits sensitivity analysis across
  alternative split seeds.
- Split-integrity checks and a shuffled-label negative control can reveal some
  implementation failures but cannot prove the absence of all leakage.
- Sigmoid is the only calibration method evaluated. Its three inner folds are
  fixed in advance and are not compared with isotonic or other approaches.
- Top-label ECE depends on ten fixed equal-width bins and can hide class-level
  or within-bin calibration behavior. Empty bins provide no evidence.
- Calibration estimates are based on a small dataset. Differences between
  metrics and folds should not be treated as deployment guarantees.
- Missing-value injection is independent across observed feature cells and
  does not represent a measured missingness mechanism.
- Gaussian perturbations are independent, zero mean, and scaled from training
  folds. They do not model bias, outliers, correlated sensor noise, or drift.
- Training-time corruption, structured missingness, label noise, and combined
  perturbations are outside the v0.6 scope.
- The five fold scores are correlated because their training partitions
  overlap. Their standard deviation is descriptive and is not a confidence
  interval.
- The committed results are specific to this dataset revision, split, feature
  selection, preprocessing, and dependency behavior.
- Species labels and measurements are treated as given; label uncertainty and
  measurement error are not modeled.
- The result is not intended for field identification, ecological inference,
  or decisions affecting wildlife.

## Reproducibility

The dataset bytes, source revision, split parameters, model configurations,
feature sets, and checksum manifest are committed. Regenerate the complete
reference set in place and then verify it:

```bash
python examples/run_demo.py
python examples/run_demo.py --verify-only
```

For a non-destructive check, select another directory:

```bash
python examples/run_demo.py --output-dir .work/generated
python examples/run_demo.py --output-dir .work/generated --verify-only
```

JSON and CSV files are replaced atomically. Exact numeric artifacts are fixed
by `results/checksums.sha256`; the figures are generated from the same recorded
evaluation values.

## Development and Testing

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Tests cover dataset validation, holdout and cross-validation behavior,
feature-ablation accounting, leakage diagnostics, report schemas, CLI
arguments and errors, deterministic outputs, and checksum verification.
GitHub Actions checks the installed CLI and dataset, runs the tests and a
controlled evaluation on Python 3.10 through 3.14, and independently requires
a Python 3.12 regeneration to match committed `results/`.

## Compatibility

Python 3.10 through 3.14 are exercised in CI. Version 0.6.0 is an alpha
evaluation project, so consumers should treat the documented command,
`metrics.json` report version, and CSV columns as versioned interfaces rather
than as a 1.x stability guarantee. Release changes are recorded in
[`CHANGELOG.md`](CHANGELOG.md).

## Project Structure

```text
ml-evaluation-workbench/
├── .github/workflows/ci.yml
├── data/
│   ├── README.md
│   └── penguins.csv
├── examples/
│   ├── download_penguins.py
│   └── run_demo.py
├── results/
│   ├── README.md
│   ├── checksums.sha256
│   ├── confusion_matrix.png
│   ├── cross_validation_folds.csv
│   ├── cross_validation_scores.png
│   ├── feature_ablation_folds.csv
│   ├── feature_ablation_scores.png
│   ├── feature_ablation_summary.csv
│   ├── leakage_diagnostic_folds.csv
│   ├── leakage_diagnostics.json
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── predictions.csv
│   ├── probability_calibration.png
│   ├── probability_calibration_bins.csv
│   ├── probability_calibration_folds.csv
│   ├── probability_calibration_predictions.csv
│   ├── probability_calibration_summary.csv
│   ├── robustness.png
│   ├── robustness_diagnostics.json
│   ├── robustness_folds.csv
│   └── robustness_summary.csv
├── src/ml_evaluation_workbench/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── dataset.py
│   ├── evaluation.py
│   └── reporting.py
├── tests/
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
├── README.md
└── pyproject.toml
```

## Roadmap

- **v0.2:** Stratified cross-validation and fold-level evidence
- **v0.3:** Controlled model comparison
- **v0.4:** Feature ablation and leakage diagnostics
- **v0.5:** Probability calibration
- **v0.6 (current):** Missing-value and noise robustness
- **v0.7:** Class-imbalance sensitivity
- **v0.8:** Cross-experiment summaries and interface review
- **v0.9:** Documentation and reproducibility review
- **v1.0:** Stable portfolio release

## License

Project code is licensed under the MIT License. See [LICENSE](LICENSE). The
Palmer Penguins data are provided separately under CC0 1.0; see
[data/README.md](data/README.md).
