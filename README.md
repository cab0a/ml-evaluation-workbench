# ML Evaluation Workbench

## 日本語概要

このリポジトリは、固定した公開データと共通のデータ分割を使い、単純な基準モデル、ロジスティック回帰、5近傍法を比較する機械学習評価プロジェクトです。

6種類の実験では、分割ごとの評価指標と行単位の予測、特徴量除去、負の対照実験、確率校正、検証方式とクラス不均衡への感度を記録し、25件の代表比較を出力します。

v1.0.0では数値実験を変えず、コマンドライン操作、Python API、成果物名と列順を1.x安定インターフェースとして固定しています。評価結果の適用限界と再現手順は、以下の英語本文を参照してください。

---

Compare fixed machine-learning baselines through shared splits, leakage-aware
pipelines, row-level evidence, controlled diagnostics, and reproducible report
artifacts.

## Overview

ML Evaluation Workbench is a compact tabular-classification evaluation that
exposes the decisions behind a model comparison. It emphasizes evaluation
design and claim boundaries rather than model complexity or leaderboard
performance.

The project uses a checksum-pinned Palmer Penguins dataset and asks:

**How do fixed linear and nonlinear classifiers compare with a majority-class
baseline, and how does their observed behavior change under controlled
diagnostics?**

| At a glance | Evidence |
| --- | --- |
| Dataset | 344 rows, three species, pinned upstream revision and SHA-256 |
| Models | Majority-class dummy, logistic regression, and fixed 5-nearest neighbors |
| Partitions | Deterministic 75/25 holdout plus five shared stratified folds |
| Leakage control | Imputation and scaling fitted inside each training partition |
| Diagnostics | Feature ablation, shuffled labels, calibration, validation perturbations, and class downsampling |
| Outputs | 27 report artifacts plus an automatically generated SHA-256 manifest |
| Stable boundary | CLI, package-root API, artifact names, 15 CSV schemas, and five JSON top-level schemas |

## Representative Comparison

The primary five-fold comparison reports accuracy, balanced accuracy, macro
F1, and per-class recall on the same validation partitions:

| Model | Role | Accuracy, mean ± std | Macro F1, mean ± std |
| --- | --- | ---: | ---: |
| Majority-class dummy | Reference baseline | 0.442 ± 0.006 | 0.204 ± 0.002 |
| Logistic regression | Linear baseline | 0.945 ± 0.034 | 0.928 ± 0.041 |
| 5-nearest neighbors | Nonlinear comparator | 0.959 ± 0.011 | 0.949 ± 0.018 |

KNN's paired macro-F1 difference from logistic regression is +0.021 on
average, but individual folds range from -0.011 to +0.108. This supports a
useful nonlinear comparison under the fixed setup, not a claim that KNN is
universally better.

On the deterministic holdout, KNN correctly classifies 83 of 86 rows and
logistic regression classifies 81. Row-level outcomes remain inspectable in
[`predictions.csv`](results/predictions.csv).

The cross-experiment summary then selects 25 predefined contrasts from six
experiment families without selecting rows by effect size:

![Representative cross-experiment macro-F1 contrasts](results/cross_experiment_summary.png)

The figure shows only 15 macro-F1 contrasts so the horizontal scale has one
meaning. The complete
[`cross_experiment_summary.csv`](results/cross_experiment_summary.csv) also
contains probability metrics and target-class recall. Their effect magnitudes
must not be compared across metric scales.

## Evaluation Boundaries

- The numerical evidence comes from one small public dataset, one holdout
  split, and one five-fold partition.
- The three fixed models are controlled comparators; no hyperparameter search
  or model selection is performed.
- Fold standard deviations describe the five observed folds and are not
  confidence intervals.
- Split-integrity checks and shuffled labels test specific failure modes but
  cannot prove the absence of every form of leakage.
- Missing values, Gaussian noise, and class downsampling are synthetic
  sensitivity tests, not measured production drift.
- Stable software and artifact interfaces do not imply deployment suitability
  or performance on independent data.

The complete method-specific limits are documented in
[Limitations and Claim Boundaries](docs/limitations.md).

## Quick Start

Python 3.10 or later is required. On Debian or Ubuntu, install
`python3-venv` if `venv` reports that `ensurepip` is unavailable.

```bash
git clone https://github.com/cab0a/ml-evaluation-workbench.git
cd ml-evaluation-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python examples/download_penguins.py --check
ml-evaluation-workbench evaluate data/penguins.csv \
  --output-dir output/quickstart
ml-evaluation-workbench verify output/quickstart
```

The evaluation writes 27 report artifacts and
`output/quickstart/checksums.sha256`. Verification should report:

```text
Verified: 27 artifacts
Manifest: output/quickstart/checksums.sha256
```

Start with:

1. `cross_experiment_summary.png` for the comparison map;
2. `cross_experiment_summary.csv` for the 25 selected contrasts;
3. `model_comparison.csv` for the primary baselines; and
4. `metrics.json` and the fold-level CSV files for complete evidence.

Checksum-fixed reference copies are committed under [`results/`](results/).

## Generated Artifacts

| Area | Main artifacts | What to inspect |
| --- | --- | --- |
| Complete evaluation | `metrics.json` | Dataset identity, settings, metrics, experiment summaries, and report version |
| Primary comparison | `model_comparison.csv`, `cross_validation_folds.csv`, `predictions.csv` | Baseline roles, fold variability, paired results, and row-level holdout errors |
| Feature ablation | `feature_ablation_summary.csv`, `feature_ablation_folds.csv` | Shared-fold differences from the two-feature reference |
| Leakage diagnostics | `leakage_diagnostics.json`, `leakage_diagnostic_folds.csv` | Split integrity and shuffled-label negative control |
| Calibration | `probability_calibration_summary.csv`, `probability_calibration_predictions.csv`, `probability_calibration_bins.csv` | Cross-fitted probabilities, probability metrics, and reliability evidence |
| Robustness | `robustness_summary.csv`, `robustness_folds.csv`, `robustness_diagnostics.json` | Perturbation scope, affected cells, metrics, and paired effects |
| Class imbalance | `class_imbalance_summary.csv`, `class_imbalance_folds.csv`, `class_imbalance_diagnostics.json` | Retained rows, class counts, target recall, and paired effects |
| Navigation | `cross_experiment_summary.csv`, `cross_experiment_summary.png` | Fixed representative contrasts and their source artifacts |
| Interface review | `interface_contract.json`, `checksums.sha256` | Stable CLI/API/schema inventory and byte verification |

The complete 27-artifact inventory and schema policy are documented in
[Artifact Schema](docs/artifact-schema.md). Detailed numerical interpretation
is maintained in [Reference Results](results/README.md).

## Key Features

- Pinned public dataset with offline SHA-256 verification
- Shared holdout and cross-validation partitions across controlled comparisons
- Training-partition-only median imputation and standardization
- Majority-class reference, fixed linear baseline, and fixed nonlinear
  comparator
- Paired fold-level comparisons and row-level holdout predictions
- Split-integrity checks and a within-fold shuffled-label negative control
- Cross-fitted probability calibration with saved probability vectors and bins
- Deterministic validation perturbations with affected-cell accounting
- Deterministic class downsampling with retained-row signatures
- Fixed cross-experiment selection policy with source-artifact traceability
- Atomic per-file writes and a manifest covering all 27 report artifacts
- Stable CLI, package-root API, artifact schemas, and Python 3.10–3.14 CI

## Evaluation Design

The fixed workflow combines:

1. deterministic holdout and shared stratified outer folds;
2. primary model comparison;
3. three-way feature ablation;
4. split-integrity and shuffled-label diagnostics;
5. uncalibrated versus sigmoid probability evaluation;
6. validation-only missing-value and Gaussian-noise perturbations;
7. controlled least-frequent-class retention; and
8. a predefined cross-experiment navigation summary.

The same-fold design preserves paired differences. Preprocessing and
calibration remain inside their permitted training partitions. Exact models,
metrics, seeds, severities, sampling rules, and the full evaluation sequence
are documented in [Evaluation Design](docs/evaluation-design.md). CLI defaults,
validation rules, and the public Python API are documented in
[Configuration and Interfaces](docs/configuration.md).

## Results

| Experiment | Selected result | Interpretation boundary |
| --- | --- | --- |
| Primary comparison | KNN CV macro F1 `0.949 ± 0.018`; logistic `0.928 ± 0.041`; dummy `0.204 ± 0.002` | Fixed models and folds; not universal ranking |
| Feature ablation | Removing either measurement reduces mean macro F1 by at least `0.284` for KNN and `0.353` for logistic | Descriptive signal, not causal importance |
| Shuffled labels | Macro F1 falls to `0.254 ± 0.052` for logistic and `0.298 ± 0.054` for KNN | Negative control does not exclude all leakage |
| Calibration | Sigmoid lowers KNN log loss by `0.148` but worsens its top-label ECE; all three recorded probability metrics worsen for logistic | Method and metric dependent |
| 50% injected missingness | Macro F1 falls by `0.304` for logistic and `0.324` for KNN | Synthetic validation perturbation |
| 25% Chinstrap retention | Chinstrap recall falls by `0.203` for logistic and `0.158` for KNN | Downsampling changes prevalence and sample size |

The committed artifacts also include confusion matrix, fold-score,
feature-ablation, calibration, robustness, and class-imbalance figures. See
[Reference Results](results/README.md) for complete tables, protocol-specific
interpretation, and continuity with earlier releases.

## Reproducibility

Verify the committed dataset and artifacts without changing them:

```bash
python examples/download_penguins.py --check
ml-evaluation-workbench verify results
```

Run a non-destructive compatibility evaluation:

```bash
python examples/run_demo.py --output-dir .work/generated
ml-evaluation-workbench verify .work/generated
```

Regenerate the byte-exact reference set with Python 3.12 on the documented
Ubuntu environment:

```bash
python -m pip install -c requirements-reproducibility.txt -e .
python examples/run_demo.py
ml-evaluation-workbench verify results
git diff --exit-code -- results/
```

The Python 3.10 through 3.14 matrix requires successful tests, generation, and
self-consistent manifest verification. A separate Python 3.12 job uses pinned
constraints and requires regenerated `results/` to match the committed bytes.

## Development and Testing

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Tests cover dataset provenance, deterministic evaluation, split behavior,
feature ablation, leakage diagnostics, probability outputs, perturbation
accounting, class-retention sampling, summary selection, CLI errors, public API
signatures, artifact schemas, manifest verification, and reference
constraints.

GitHub Actions runs the suite on Python 3.10 through 3.14. On Python 3.12, it
also executes the README Quick Start, verifies the generated checksum manifest,
and requires the primary metrics and comparison artifacts to be non-empty. A
separate Python 3.12 job reproduces the committed reference artifacts and
requires an empty result diff.

## Compatibility

Version 1.0.0 keeps the documented CLI and exit meanings, package-root Python
API, `EvaluationResult` fields, 27 artifact names, 15 CSV column sequences,
five JSON top-level key sequences, report version 8, and interface-contract
version 3 stable throughout 1.x.

Python 3.10 through 3.14 compatibility and Python 3.12 byte-exact reproduction
are separate claims. See [Compatibility](docs/compatibility.md) for compatible
extensions, breaking-change criteria, environment scope, and exclusions.

## Documentation

| Document | Contents |
| --- | --- |
| [Evaluation Design](docs/evaluation-design.md) | Dataset scope, partitions, preprocessing, six experiment families, and selection policy |
| [Configuration and Interfaces](docs/configuration.md) | CLI, defaults, fixed settings, Python API, and exit codes |
| [Artifact Schema](docs/artifact-schema.md) | 27-artifact inventory, report structure, CSV/JSON boundaries, and manifest behavior |
| [Compatibility](docs/compatibility.md) | Stable 1.x surfaces, breaking changes, and environment distinctions |
| [Limitations](docs/limitations.md) | Complete numerical, methodological, and claim boundaries |
| [Reference Results](results/README.md) | Full tables, figures, interpretation, and reproduction review |
| [Dataset Provenance](data/README.md) | Upstream revision, license, checksum, and verification |
| [Changelog](CHANGELOG.md) | Versioned project history |

## Project Layout

| Path | Role |
| --- | --- |
| `src/ml_evaluation_workbench/` | Dataset checks, evaluation, reporting, interfaces, CLI, and manifest verification |
| `data/` | Pinned public dataset and provenance |
| `examples/` | Dataset verification and complete reproduction entry points |
| `results/` | Committed JSON, CSV, PNG, and checksum evidence |
| `tests/` | Evaluation, interface, CLI, schema, and reproducibility regression tests |
| `docs/` | Evaluation design, configuration, artifact, compatibility, and limitation references |
| `.github/workflows/ci.yml` | Python compatibility and byte-exact reference jobs |

## License

Project code is licensed under the [MIT License](LICENSE). Palmer Penguins data
are provided separately under CC0 1.0; see
[Dataset Provenance](data/README.md).
