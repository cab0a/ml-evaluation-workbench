# ML Evaluation Workbench

## 日本語概要

このリポジトリは、固定した公開データと共通splitを使い、dummy、logistic
regression、5-nearest neighborsを比較するML評価プロジェクトです。モデル評価の
設計や結果の監査方法を確認したいMLエンジニアに役立ちます。

fold-level metrics、row-level predictions、feature ablation、split integrity、
shuffled-label negative control、probability calibration、validation robustness、
class-imbalance sensitivityと、6種類の実験から固定方針で選んだ25件の代表比較を
含みます。v1.0.0では数値実験を変更せず、CLI、Python API、27個の成果物名、
15種類のCSV列順、5種類のJSONトップレベルキーを1.x安定インターフェースとして
固定しました。前処理は各training partition内でfitされ、成果物はSHA-256と
Python 3.12参照環境で再現性を検証できます。安定性の範囲、評価結果の適用限界、
再現手順の詳細は英語本文を参照してください。

---

[![CI](https://github.com/cab0a/ml-evaluation-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/cab0a/ml-evaluation-workbench/actions/workflows/ci.yml)

Compare machine-learning baselines through shared splits, leakage-aware
pipelines, inspectable predictions, and reproducible evaluation artifacts.

## Overview

ML Evaluation Workbench demonstrates a small but complete evaluation cycle:

**Question → Controlled Comparison → Diagnostics → Interpretation**

Version 1.0.0 is the stable portfolio release. It preserves the v0.8 numerical
evaluation and the v0.9 reproducibility workflow while defining the CLI,
Python API, artifact inventory, CSV column order, and JSON top-level keys as
backward-compatible 1.x interfaces. Regression tests and a generated
machine-readable contract guard those surfaces against accidental drift.

The repository emphasizes evaluation design and reproducibility rather than
model complexity or leaderboard performance.

It is intended for ML engineers and reviewers who want a compact example of
how to structure a controlled model comparison. Unlike the image-processing
experiments elsewhere in this portfolio, this repository focuses on tabular
classification evidence: baselines, split policy, fold variability, feature
ablation, negative controls, and row-level errors.

## Representative Result

The cross-experiment summary contains 25 representative paired contrasts
selected by a fixed policy from model comparison, feature ablation,
shuffled-label control, probability calibration, validation robustness, and
class-imbalance sensitivity. It provides one navigation layer without
discarding the complete source artifacts.

![Representative cross-experiment macro-F1 contrasts](results/cross_experiment_summary.png)

Positive `preferred_effect_mean` values favor the condition after accounting
for whether a metric is higher- or lower-is-better. The figure intentionally
shows only the 15 macro-F1 contrasts; all 25 contrasts, including probability
metrics and target-class recall, are available in
[`cross_experiment_summary.csv`](results/cross_experiment_summary.csv).
Effects from different metric scales must not be compared by magnitude.

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
- Deterministic downsampling of the least frequent training class
- Shared class-retention samples at 100%, 75%, 50%, and 25%
- Target-class recall, class-count evidence, and paired sensitivity summaries
- Fixed representative contrasts across six experiment families
- Direction-aligned effect values with preserved paired fold variation
- Machine-readable CLI, Python API, report, and artifact interface contract
- Stable 1.x contract for CLI behavior, Python API, and artifact schemas
- Regression checks for 15 CSV schemas and five JSON top-level key sets
- Public checksum verification through `ml-evaluation-workbench verify`
- Automatic manifest generation after every successful evaluation
- Python 3.12 reference constraints for byte-exact artifact regeneration
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
ml-evaluation-workbench verify output/quickstart
```

The evaluation writes twenty-seven report artifacts and
`checksums.sha256` under `output/quickstart/`; the next command verifies the
complete documented set.
Start with `cross_experiment_summary.csv` and
`cross_experiment_summary.png` for a compact evidence map, then use
`interface_contract.json` to inspect the documented interfaces. Complete
fold-level and row-level evidence remains available in the experiment-specific
artifacts. Checksum-fixed reference copies are committed under `results/`.

## Usage

```text
ml-evaluation-workbench evaluate DATASET [--output-dir DIR]
                                         [--random-state INTEGER]
                                         [--test-size FRACTION]
                                         [--cv-folds INTEGER]
                                         [--calibration-folds INTEGER]

ml-evaluation-workbench verify ARTIFACT_DIR
```

The documented v1.0 evaluation command is:

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
logistic_regression macro F1 at 25% Chinstrap retention: 0.885
logistic_regression Chinstrap recall at 25% retention: 0.619
knn CV log loss, uncalibrated: 0.280
knn CV log loss, sigmoid calibrated: 0.133
knn macro F1 at 50% injected missingness: 0.625
knn macro F1 at 1.0x feature noise: 0.673
knn macro F1 at 25% Chinstrap retention: 0.922
knn Chinstrap recall at 25% retention: 0.737
Cross-experiment contrasts: 25
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
Class-imbalance folds: results/class_imbalance_folds.csv
Class-imbalance summary: results/class_imbalance_summary.csv
Class-imbalance diagnostics: results/class_imbalance_diagnostics.json
Class-imbalance plot: results/class_imbalance.png
Cross-experiment summary: results/cross_experiment_summary.csv
Cross-experiment plot: results/cross_experiment_summary.png
Interface contract: results/interface_contract.json
Checksums: results/checksums.sha256
```

Verify the generated files independently:

```bash
ml-evaluation-workbench verify results
```

Expected verification summary:

```text
Verified: 27 artifacts
Manifest: results/checksums.sha256
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
from v1.0. This keeps the question interpretable and avoids relying on
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
18. Within each outer training fold, retain 100%, 75%, 50%, or 25% of the
    globally least frequent class using deterministic samples shared by both
    substantive models.
19. Keep every validation partition unchanged, and record retained source-row
    hashes, class counts, target-class recall, and paired differences from full
    retention.
20. Select a fixed representative set of comparisons from the six experiment
    families. Preserve same-fold differences and align their signs so a
    positive preferred effect always favors the condition.
21. Record the installed CLI, Python API, report schema, supported Python
    versions, and generated-artifact inventory in a machine-readable interface
    contract.
22. Save aggregate metrics, fold-level evidence, diagnostic summaries,
    predictions, and evaluation figures.
23. Atomically write a SHA-256 manifest covering the 27 documented report
    artifacts.

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
- class-imbalance target selection, retention levels, sampling policy,
  condition summaries, and paired differences from full retention
- cross-experiment selection policy, metric-direction rules, source artifacts,
  and the number and scope of representative contrasts

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

`class_imbalance_folds.csv` records one row per retention level, fold, and
model, including the deterministic seed, retained-row signature, before/after
class counts, aggregate metrics, and per-class recall.
`class_imbalance_summary.csv` provides fold summaries and paired mean
differences from full retention. `class_imbalance_diagnostics.json` fixes the
target-class selection and sampling rules. `class_imbalance.png` visualizes
macro F1 and Chinstrap-recall sensitivity.

`cross_experiment_summary.csv` contains 25 fixed representative contrasts
across six experiment families. Every row records its reference and condition,
metric direction, fold means and population standard deviations, paired
condition-minus-reference differences, direction-aligned preferred effect,
source artifact, and interpretation. `cross_experiment_summary.png` visualizes
the 15 macro-F1 contrasts only, avoiding magnitude comparisons across
incompatible metric scales.

`interface_contract.json` records interface-contract schema version 3, the
project and report versions, Python compatibility, both CLI commands and exit
codes, package exports, public function signatures, `EvaluationResult` fields,
generated artifacts, all 15 CSV column sequences, top-level keys for all five
JSON artifacts, and the reference-environment policy.

`confusion_matrix.png` visualizes the logistic-regression holdout errors.
`cross_validation_scores.png` shows all three models' aggregate scores in
each fold. `checksums.sha256` is written automatically after evaluation and
fixes the bytes of all twenty-seven reference artifacts.

## Results

### Cross-Experiment Summary

The fixed cross-experiment policy selects 25 contrasts from six existing
experiment families. Selection is based on experiment role and tested
severity, not on effect size. Representative rows include:

| Experiment | Condition vs Reference | Metric | Preferred Effect |
| --- | --- | --- | ---: |
| Model comparison | KNN vs majority dummy | Macro F1 | +0.745 |
| Feature ablation | Bill depth only vs both, logistic | Macro F1 | -0.359 |
| Negative control | Shuffled vs observed, logistic | Macro F1 | -0.674 |
| Calibration | Sigmoid vs uncalibrated, KNN | Log loss | +0.148 |
| Validation robustness | 50% missing vs unperturbed, KNN | Macro F1 | -0.324 |
| Class imbalance | 25% vs 100% Chinstrap, logistic | Chinstrap recall | -0.203 |

For lower-is-better metrics such as log loss, `preferred_effect_mean` reverses
the raw condition-minus-reference sign. This makes direction consistent
within the navigation table, but it does not put accuracy, loss, calibration,
recall, and macro F1 on a common scale. Each row links back to its complete
source artifact.

![Representative cross-experiment macro-F1 contrasts](results/cross_experiment_summary.png)

### Holdout Comparison

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

### Class-Imbalance Sensitivity

Chinstrap is the least frequent class in the fixed dataset, with 68 of 344
rows. The experiment downsamples only its rows inside each outer training
fold. All validation rows and labels remain unchanged.

| Chinstrap Retention | Mean Training Share | Logistic Macro F1 | Logistic Recall | KNN Macro F1 | KNN Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 19.8% | 0.928 | 0.822 | 0.949 | 0.896 |
| 75% | 15.5% | 0.934 | 0.823 | 0.945 | 0.867 |
| 50% | 11.0% | 0.928 | 0.765 | 0.941 | 0.838 |
| 25% | 6.0% | 0.885 | 0.619 | 0.922 | 0.737 |

At 25% retention, logistic regression loses 0.043 mean macro F1 and 0.203
mean Chinstrap recall relative to full retention. KNN loses 0.027 macro F1 and
0.158 Chinstrap recall. The 75% logistic-regression result is slightly above
its full-retention mean, so the observed response is not strictly monotonic at
every level.

![Class-imbalance sensitivity](results/class_imbalance.png)

See [results/README.md](results/README.md) for interpretation and the boundary
between this controlled result and a general performance claim.

## Limitations

- Version 1.0.0 evaluates one small dataset with one deterministic holdout and
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
- Structured missingness, label noise, combined perturbations, and changes to
  non-target training classes are outside the v1.0 scope.
- The class-imbalance experiment downsamples only Chinstrap, selected because
  it is globally least frequent in this dataset. It does not evaluate other
  target classes, oversampling, class weighting, threshold changes, or
  mitigation strategies.
- Downsampling changes both class prevalence and training-set size. The
  experiment does not isolate those two effects or model population-prior
  shift in validation data.
- Class-retention samples use one deterministic seed per fold and condition.
  Alternative retained rows can produce different sensitivity curves.
- The cross-experiment summary is a selective navigation artifact, not a
  statistical meta-analysis. Its fixed policy omits intermediate robustness
  and class-retention severities.
- Direction alignment makes favorable signs consistent, but preferred effects
  from different metrics are not standardized or comparable by magnitude.
- The stable contract covers documented software interfaces and the
  reproduction workflow. It does not guarantee model quality, dependency
  behavior outside the supported ranges, or byte-identical output outside the
  reference environment.
- The Python 3.12 constraints define the byte-exact reference environment, not
  the only supported environment. Other supported Python and dependency
  combinations are checked for successful, self-consistent generation rather
  than equality with committed PNG bytes.
- The five fold scores are correlated because their training partitions
  overlap. Their standard deviation is descriptive and is not a confidence
  interval.
- The committed results are specific to this dataset revision, split, feature
  selection, preprocessing, and dependency behavior.
- Species labels and measurements are treated as given; label uncertainty and
  measurement error are not modeled.
- The result is not intended for field identification, ecological inference,
  or decisions affecting wildlife.

## Stable Interfaces

Version 1.0.0 defines the documented surfaces that downstream users can depend
on throughout the 1.x release line:

- CLI: `evaluate` with the five documented options and `verify` with an
  artifact directory
- Exit code `0` for success and `2` for argument, input, or output errors
- Python exports including `evaluate_dataset`, `EvaluationResult`, and
  `verify_artifact_manifest`
- `metrics.json` report schema version 8 and its top-level sections
- Interface-contract schema version 3
- Twenty-seven generated artifact names, their roles, and the checksum
  manifest
- Ordered columns for all 15 CSV artifacts
- Ordered top-level keys for all five JSON artifacts
- Python 3.10 through 3.14 compatibility exercised in CI
- Python 3.12 plus `requirements-reproducibility.txt` as the byte-exact
  reference environment

[`interface_contract.json`](results/interface_contract.json) is generated with
each evaluation. Its artifact inventory and `EvaluationResult` fields come
from implementation-level definitions, while the documented CLI and Python
surfaces are explicitly reviewed for this release. Backward-compatible
additions and corrections can occur in later 1.x releases; removing or
changing a documented stable interface requires a new major version.

This promise concerns software interfaces and reproducible evaluation
artifacts. It does not make the fitted examples production-ready or guarantee
their performance on new data.

## Reproducibility

The dataset bytes, source revision, split parameters, model configurations,
feature sets, reference constraints, and checksum manifest are committed.
For byte-exact regeneration, use Python 3.12 on the documented Ubuntu
environment and install the reference constraints:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-reproducibility.txt -e .
python examples/download_penguins.py --check
python examples/run_demo.py
ml-evaluation-workbench verify results
git diff --exit-code -- results/
```

For a non-destructive compatibility check, generate into another directory:

```bash
python examples/run_demo.py --output-dir .work/generated
ml-evaluation-workbench verify .work/generated
```

The Python 3.10 through 3.14 matrix installs the declared dependency ranges and
requires successful tests, generation, and verification of each newly
generated manifest. A separate Python 3.12 job installs
`requirements-reproducibility.txt` and requires an exact diff match with
committed `results/`. This distinguishes broad runtime compatibility from
byte-exact reference reproduction.

JSON, CSV, PNG, and manifest files are replaced atomically. Exact reference
bytes are fixed by `results/checksums.sha256`; figures are generated from the
same recorded evaluation values.

## Development and Testing

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Tests cover stable interface regression, dataset validation, holdout and
cross-validation behavior,
feature-ablation accounting, leakage diagnostics, robustness and
class-imbalance protocols, cross-experiment selection and metric direction,
interface-contract contents, report schemas, CLI arguments and errors,
deterministic outputs, manifest generation, checksum verification, and direct
dependency constraints.
GitHub Actions checks the installed CLI and dataset, runs the tests and a
controlled evaluation on Python 3.10 through 3.14, and independently requires
a Python 3.12 regeneration to match committed `results/`.

## Compatibility

Python 3.10 through 3.14 are exercised in CI. Version 1.0.0 marks the
documented CLI, Python API, artifact names, CSV column order, and JSON
top-level keys as stable within 1.x. This software stability statement does
not imply that the example models are suitable for production deployment.
Release changes are recorded in [`CHANGELOG.md`](CHANGELOG.md).

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
│   ├── class_imbalance.png
│   ├── class_imbalance_diagnostics.json
│   ├── class_imbalance_folds.csv
│   ├── class_imbalance_summary.csv
│   ├── confusion_matrix.png
│   ├── cross_experiment_summary.csv
│   ├── cross_experiment_summary.png
│   ├── cross_validation_folds.csv
│   ├── cross_validation_scores.png
│   ├── feature_ablation_folds.csv
│   ├── feature_ablation_scores.png
│   ├── feature_ablation_summary.csv
│   ├── interface_contract.json
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
│   ├── interface.py
│   ├── reproducibility.py
│   └── reporting.py
├── tests/
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── requirements-reproducibility.txt
└── pyproject.toml
```

## Roadmap

- **v0.2:** Stratified cross-validation and fold-level evidence
- **v0.3:** Controlled model comparison
- **v0.4:** Feature ablation and leakage diagnostics
- **v0.5:** Probability calibration
- **v0.6:** Missing-value and noise robustness
- **v0.7:** Class-imbalance sensitivity
- **v0.8:** Cross-experiment summaries and interface review
- **v0.9:** Documentation and reproducibility review
- **v1.0 (current):** Stable portfolio release and 1.x interface contract
- **v1.x maintenance:** Backward-compatible fixes, dependency review, and
  documentation improvements

## License

Project code is licensed under the MIT License. See [LICENSE](LICENSE). The
Palmer Penguins data are provided separately under CC0 1.0; see
[data/README.md](data/README.md).
