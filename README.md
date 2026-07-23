# ML Evaluation Workbench

[![CI](https://github.com/cab0a/ml-evaluation-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/cab0a/ml-evaluation-workbench/actions/workflows/ci.yml)

Reproducible machine-learning baselines, evaluation artifacts, and concise
interpretation using public datasets.

## Overview

ML Evaluation Workbench demonstrates a small but complete evaluation cycle:

**Question → Baseline → Controlled Comparison → Evaluation → Error Review**

Version 0.3.0 asks a deliberately narrow question: how does a fixed local
nonlinear classifier compare with a linear baseline when both use only bill
length and bill depth to separate three Palmer penguin species? It evaluates a
majority-class dummy, multinomial logistic regression, and 5-nearest neighbors
under one deterministic holdout and five shared stratified folds.

The repository emphasizes evaluation design and reproducibility rather than
model complexity or leaderboard performance.

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

## Features

- Pinned, checksum-verified public dataset
- Deterministic stratified train/test split
- Five-fold stratified cross-validation shared by all three models
- Majority-class `DummyClassifier` baseline
- Median imputation and standardization inside a scikit-learn `Pipeline`
- Logistic-regression classifier using two interpretable measurements
- Fixed 5-nearest-neighbors nonlinear comparator without parameter tuning
- Compact holdout and cross-validation model-comparison table
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

```bash
git clone https://github.com/cab0a/ml-evaluation-workbench.git
cd ml-evaluation-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the committed dataset and run the evaluation:

```bash
python examples/download_penguins.py --check
ml-evaluation-workbench evaluate data/penguins.csv --output-dir results
```

Run the tests:

```bash
python -m pytest
```

Regenerate and verify the committed sample artifacts:

```bash
python examples/run_demo.py
python examples/run_demo.py --verify-only
```

## Usage

```text
ml-evaluation-workbench evaluate DATASET [--output-dir DIR]
                                         [--random-state INTEGER]
                                         [--test-size FRACTION]
                                         [--cv-folds INTEGER]
```

The documented v0.3 command is:

```bash
ml-evaluation-workbench evaluate data/penguins.csv \
  --output-dir results \
  --random-state 42 \
  --test-size 0.25 \
  --cv-folds 5
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
Metrics: results/metrics.json
Predictions: results/predictions.csv
Confusion matrix: results/confusion_matrix.png
Cross-validation fold scores: results/cross_validation_folds.csv
Cross-validation scores: results/cross_validation_scores.png
Model comparison: results/model_comparison.csv
```

## Dataset

The repository includes the simplified Palmer Penguins dataset maintained by
Allison Horst, Alison Hill, and Kristen Gorman. It contains 344 rows and eight
columns. The data are available under CC0 1.0.

- [Dataset documentation](https://allisonhorst.github.io/palmerpenguins/)
- [Dataset license](https://allisonhorst.github.io/palmerpenguins/LICENSE.html)
- [Pinned provenance and checksum](data/README.md)

Only `bill_length_mm` and `bill_depth_mm` are used as model inputs. Island,
sex, body mass, flipper length, and observation year are intentionally excluded
from v0.3. This keeps the question interpretable and avoids relying on
location-specific correlations that can make a random holdout unnecessarily
easy.

## Methodology

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
10. Save aggregate metrics, paired differences, fold-level scores, a compact
    model-comparison table, sorted holdout predictions, and evaluation figures.

The preprocessing steps are part of each scikit-learn `Pipeline`, so their
statistics are not estimated from the holdout partition or a fold's validation
partition.

## Output Schema

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

`confusion_matrix.png` visualizes the logistic-regression holdout errors.
`cross_validation_scores.png` shows all three models' aggregate scores in
each fold. `checksums.sha256` fixes the bytes of all six reference artifacts.

## Evaluation

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

See [results/README.md](results/README.md) for interpretation and the boundary
between this controlled result and a general performance claim.

## Limitations

- Version 0.3.0 evaluates one small dataset with one deterministic holdout and
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
- The five fold scores are correlated because their training partitions
  overlap. Their standard deviation is descriptive and is not a confidence
  interval.
- One shuffled cross-validation partition does not measure sensitivity to
  alternative partition seeds.
- The committed score is specific to this dataset revision, split, feature
  selection, preprocessing, and dependency behavior.
- Species labels and measurements are treated as given; label uncertainty and
  measurement error are not modeled.
- The result is not intended for field identification, ecological inference,
  or decisions affecting wildlife.

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
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── predictions.csv
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
- **v0.3 (current):** Controlled model comparison
- **v0.4:** Feature ablation and leakage diagnostics
- **v0.5:** Probability calibration
- **v0.6:** Missing-value and noise robustness
- **v0.7:** Class-imbalance sensitivity
- **v0.8:** Cross-experiment summaries and interface review
- **v0.9:** Documentation and reproducibility review
- **v1.0:** Stable portfolio release

## License

Project code is licensed under the MIT License. See [LICENSE](LICENSE). The
Palmer Penguins data are provided separately under CC0 1.0; see
[data/README.md](data/README.md).
