# Artifact Schema

## 日本語概要

本書は、評価処理が生成する27件の成果物と`checksums.sha256`を、目的・形式・検証方法ごとに定義します。行単位の予測、分割単位の評価、確率校正、感度分析、横断集約、公開インターフェースの各CSV・JSON・画像を追跡できます。

列順、JSON項目、原子的な書き込みの詳細は以下の英語本文を参照してください。

---

ML Evaluation Workbench writes 27 documented report artifacts and one
`checksums.sha256` manifest. This document explains their roles, schema
boundaries, and verification policy. Exact ordered schemas are generated in
[`interface_contract.json`](../results/interface_contract.json) and defined by
`src/ml_evaluation_workbench/interface.py`.

## Artifact Groups

| Evaluation area | Artifacts | Evidence |
| --- | --- | --- |
| Complete report | `metrics.json` | Dataset identity, settings, model results, experiment summaries, and report version |
| Holdout | `predictions.csv`, `confusion_matrix.png` | Row-level predictions, correctness flags, and logistic-regression errors |
| Model comparison | `model_comparison.csv`, `cross_validation_folds.csv`, `cross_validation_scores.png` | Holdout metrics, fold metrics, paired comparisons, and observed variation |
| Feature ablation | `feature_ablation_folds.csv`, `feature_ablation_summary.csv`, `feature_ablation_scores.png` | Shared-fold feature-set comparisons and differences from both measurements |
| Leakage diagnostics | `leakage_diagnostic_folds.csv`, `leakage_diagnostics.json` | Split integrity, coverage, shuffled-label scores, and claim boundary |
| Probability calibration | `probability_calibration_folds.csv`, `probability_calibration_summary.csv`, `probability_calibration_predictions.csv`, `probability_calibration_bins.csv`, `probability_calibration.png` | Cross-fitted probabilities, probability metrics, bin evidence, and reliability diagram |
| Validation robustness | `robustness_folds.csv`, `robustness_summary.csv`, `robustness_diagnostics.json`, `robustness.png` | Perturbation accounting, fold metrics, paired effects, protocol, and sensitivity plot |
| Class imbalance | `class_imbalance_folds.csv`, `class_imbalance_summary.csv`, `class_imbalance_diagnostics.json`, `class_imbalance.png` | Retained rows, class counts, recall, paired effects, protocol, and sensitivity plot |
| Navigation | `cross_experiment_summary.csv`, `cross_experiment_summary.png` | Fixed representative contrasts and macro-F1-only overview |
| Interface review | `interface_contract.json` | CLI, Python API, versions, artifact inventory, and ordered schemas |

## `metrics.json`

`metrics.json` uses `report_version: 8` and the following stable top-level
order:

1. `report_version`
2. `project_version`
3. `dataset`
4. `split`
5. `models`
6. `comparison`
7. `cross_validation`
8. `feature_ablation`
9. `leakage_diagnostics`
10. `probability_calibration`
11. `robustness`
12. `class_imbalance`
13. `cross_experiment_summary`

The report contains:

- dataset path, SHA-256, rows, classes, selected features, and missing cells;
- holdout strategy, random state, fraction, and train/test counts;
- model configuration and holdout metrics;
- paired holdout and fold-level model differences;
- cross-validation statistics and per-class recall;
- feature-ablation summaries and differences from the two-feature reference;
- split-integrity and shuffled-label negative-control summaries;
- probability-calibration definitions, scores, and paired method differences;
- validation-perturbation definitions, cell accounting, and paired effects;
- class-retention sampling, counts, recall, and paired effects; and
- cross-experiment selection and metric-direction policies.

## CSV Evidence

The 15 CSV files have stable ordered column sequences in the 1.x series.

### `predictions.csv`

One holdout row per source record, including the original one-based CSV row,
actual class, three model predictions, and correctness flags.

### `model_comparison.csv`

One row per model with evaluation role, holdout metrics, and cross-validation
means and population standard deviations.

### Fold-Level Files

- `cross_validation_folds.csv` records model, fold, partition sizes, aggregate
  metrics, and per-class recall.
- `feature_ablation_folds.csv` adds feature-set identity to the same shared-fold
  evidence.
- `leakage_diagnostic_folds.csv` records observed and shuffled-label scores,
  overlap counts, and observed-minus-shuffled differences.
- `probability_calibration_folds.csv` records model, probability method,
  partition sizes, accuracy, log loss, Brier score, and top-label ECE.
- `robustness_folds.csv` records perturbation, severity, seed, eligible and
  affected cells, realized fraction, feature noise scales, and metrics.
- `class_imbalance_folds.csv` records retention, seed, row counts, target
  shares, retained-row SHA-256, metrics, class counts, and per-class recall.

### Summary Files

- `feature_ablation_summary.csv` records fold means, population standard
  deviations, and paired differences from both measurements.
- `probability_calibration_summary.csv` records method means and population
  standard deviations.
- `robustness_summary.csv` records severity summaries and paired differences
  from the unperturbed condition.
- `class_imbalance_summary.csv` records retained counts, target shares,
  aggregate metrics, target recall, and differences from full retention.
- `cross_experiment_summary.csv` records 25 fixed contrasts with metric
  direction, reference and condition fold statistics, raw paired difference,
  preferred effect, source artifact, and interpretation.

### Probability Rows and Bins

- `probability_calibration_predictions.csv` contains 1,376 cross-fitted
  model-method predictions with class probabilities.
- `probability_calibration_bins.csv` records all ten equal-width top-label bins,
  including empty bins.

## JSON Diagnostic Files

Five JSON artifacts have stable ordered top-level key sequences:

- `metrics.json`
- `interface_contract.json`
- `leakage_diagnostics.json`
- `robustness_diagnostics.json`
- `class_imbalance_diagnostics.json`

Diagnostic JSON files retain experiment configuration, seed rules, data-scope
boundaries, summaries, and interpretation limits. They are intended to keep
protocol details beside the numerical outputs.

## Interface Contract

`interface_contract.json` uses contract schema version 3 and records:

- project and report versions;
- `stable_1_x` compatibility status;
- supported Python versions;
- `evaluate` and `verify` commands, arguments, defaults, and exit codes;
- package exports and public function parameters;
- `EvaluationResult` fields;
- all 27 artifact names, media types, and roles;
- ordered columns for all 15 CSV artifacts;
- ordered top-level keys for all five JSON artifacts; and
- byte-exact reference and compatibility-matrix expectations.

The contract is regenerated during every evaluation. Regression tests compare
the committed files with the schema registry and explicitly check public
function signatures, CLI defaults, report versions, artifact names, CSV
columns, and JSON key order.

## Manifest and Atomic Writes

`evaluate` atomically replaces each JSON, CSV, and PNG artifact, then writes
`checksums.sha256` atomically. The manifest covers all 27 documented report
artifacts and does not include itself.

```bash
ml-evaluation-workbench verify results
```

Verification requires exactly the documented manifest entries, rejects
duplicates and malformed digests, checks that every file exists, and compares
each SHA-256 with the recorded value.

The 28 files are not committed as one filesystem transaction. An interrupted
run can therefore leave a mixture of old and new individual artifacts before
the manifest is replaced. Run `verify` after generation before consuming or
sharing the directory.
