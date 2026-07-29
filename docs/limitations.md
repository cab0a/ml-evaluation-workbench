# Limitations and Claim Boundaries

## 日本語概要

本書は、一つの小規模データセット、一つの固定分割、限られた特徴量と分類器から得た結果の適用範囲を定義します。分割間の相関、ラベル入れ替え、確率校正、欠損・ノイズ、クラス不均衡、参照環境に関する制約を分けて記録しています。

評価結果から主張できない内容は以下の英語本文を参照してください。

---

## English Summary

These constraints define what the committed evaluation can and cannot support.
The [README](../README.md) keeps the most decision-relevant boundaries near the
representative result.

## Dataset and Split Scope

- Version 1.0.0 evaluates one small dataset with one deterministic holdout and
  one five-fold stratified cross-validation run.
- A random row split does not measure transfer across islands, years, field
  conditions, or independent collection programs.
- The simplified dataset does not provide a grouping identifier suitable for
  testing repeated observations of the same individual.
- Reusing one cross-validation partition limits sensitivity analysis across
  alternative split seeds.
- The five fold scores are correlated because their training partitions
  overlap. Their standard deviation is descriptive and is not a confidence
  interval.
- The committed results are specific to this dataset revision, split, feature
  selection, preprocessing, and dependency behavior.
- Species labels and measurements are treated as given; label uncertainty and
  measurement error are not modeled.

## Model and Feature Scope

- Four selected feature cells are missing and use training-partition median
  imputation. Other naturally occurring missingness mechanisms are not
  evaluated.
- The two-feature design is intentionally constrained and does not establish an
  optimal feature set.
- The three fixed models are controlled comparators, not a comprehensive model
  search.
- The KNN configuration is chosen in advance and is not claimed to be optimal.
  Different neighbor counts, weights, or distance metrics are not evaluated.
- Ablation differences are descriptive for the two selected measurements and
  do not establish causal feature importance.

## Leakage and Calibration Scope

- Split-integrity checks and a shuffled-label negative control can reveal some
  implementation failures but cannot prove the absence of all leakage.
- Sigmoid is the only calibration method evaluated. Its three inner folds are
  fixed in advance and are not compared with isotonic or other approaches.
- Top-label ECE depends on ten fixed equal-width bins and can hide class-level
  or within-bin calibration behavior. Empty bins provide no evidence.
- Calibration estimates are based on a small dataset. Differences between
  metrics and folds should not be treated as deployment guarantees.

## Robustness Scope

- Missing-value injection is independent across observed feature cells and
  does not represent a measured missingness mechanism.
- Gaussian perturbations are independent, zero mean, and scaled from training
  folds. They do not model bias, outliers, correlated sensor noise, or drift.
- Structured missingness, label noise, combined perturbations, and changes to
  non-target training classes are outside the v1.0 scope.

## Class-Imbalance Scope

- The class-imbalance experiment downsamples only Chinstrap, selected because
  it is globally least frequent in this dataset.
- The experiment does not evaluate other target classes, oversampling, class
  weighting, threshold changes, or mitigation strategies.
- Downsampling changes both class prevalence and training-set size. The
  experiment does not isolate those two effects or model population-prior shift
  in validation data.
- Class-retention samples use one deterministic seed per fold and condition.
  Alternative retained rows can produce different sensitivity curves.

## Summary and Reproducibility Scope

- The cross-experiment summary is a selective navigation artifact, not a
  statistical meta-analysis. Its fixed policy omits intermediate robustness
  and class-retention severities.
- Direction alignment makes favorable signs consistent, but preferred effects
  from different metrics are not standardized or comparable by magnitude.
- The stable contract covers documented software interfaces and the
  reproduction workflow. It does not guarantee model quality, dependency
  behavior outside the supported ranges, or byte-identical output outside the
  reference environment.
- Python 3.12 constraints define the byte-exact reference environment, not the
  only supported environment. Other supported combinations are checked for
  successful, self-consistent generation rather than equality with committed
  PNG bytes.
- The result is not intended for field identification, ecological inference,
  deployment guarantees, or decisions affecting wildlife.
