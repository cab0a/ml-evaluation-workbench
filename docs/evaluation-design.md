# Evaluation Design

## 日本語概要

本書は、固定した公開データと共通の分割を使うモデル比較について、問い、前処理境界、基準モデル、特徴量除去、負の対照実験、確率校正、検証データの摂動、クラス不均衡を評価する方法を定義します。探索や最良モデル選定ではなく、条件を固定した比較設計を記録しています。

評価順序と各実験の統制条件は以下の英語本文を参照してください。

---

## English Summary

This document defines the controlled question, dataset scope, split policy,
preprocessing boundaries, and six diagnostic experiment families used by ML
Evaluation Workbench 1.0.0. See the [README](../README.md) for the shortest
path from installation to the representative result.

## Evaluation Question

The workbench asks whether fixed linear and nonlinear classifiers improve on a
majority-class reference under shared data partitions, and how their observed
behavior changes under feature removal, shuffled labels, probability
calibration, validation perturbations, and controlled class downsampling.

The repository demonstrates evaluation structure rather than model search. No
model, feature set, calibration method, or perturbation severity is selected
from the reported scores.

## Dataset and Feature Scope

The committed input is the simplified Palmer Penguins dataset maintained by
Allison Horst, Alison Hill, and Kristen Gorman:

- 344 rows and three species
- pinned upstream revision and SHA-256
- CC0 1.0 data license
- two model inputs: `bill_length_mm` and `bill_depth_mm`
- one target: `species`

Island, sex, body mass, flipper length, and observation year are intentionally
excluded. The two-feature question remains interpretable and does not rely on
location-specific correlations that can make a random holdout unnecessarily
easy. Dataset provenance and verification are documented in
[`data/README.md`](../data/README.md).

## Shared Controls

| Control | Fixed design |
| --- | --- |
| Random state | `42` |
| Holdout | Stratified 75/25 split |
| Cross-validation | Five shuffled stratified folds |
| Preprocessing | Median imputation and standardization inside each training partition |
| Models | Majority-class dummy, logistic regression, and 5-nearest neighbors |
| Aggregate metrics | Accuracy, balanced accuracy, and macro F1 |
| Class-sensitive evidence | Per-class recall |
| Fold summary | Mean, population standard deviation, minimum, and maximum |

The complete scikit-learn pipeline is refitted inside every fold. Holdout and
validation partitions do not contribute imputation or scaling statistics.

## Primary Model Comparison

The majority-class `DummyClassifier` provides a minimum reference. Logistic
regression is the fixed linear baseline. Five-nearest neighbors is a fixed
nonlinear comparator using uniform weights and Euclidean distance. KNN is not
chosen or tuned from the reported results.

All three models use the same holdout rows and outer cross-validation folds.
The evaluation records holdout metrics, paired fold-level differences,
row-level holdout predictions, per-class recall, and a logistic-regression
confusion matrix.

## Feature Ablation

Logistic regression and KNN are evaluated with:

1. `bill_length_mm` only;
2. `bill_depth_mm` only; and
3. both measurements.

The same five outer folds are used for every model and feature set. Differences
are paired within each fold against the two-feature reference. The majority
dummy is excluded because it does not use input features.

These comparisons describe signal available to the fixed models under the
controlled partitions. They do not establish causal importance or an optimal
feature subset.

## Split Integrity and Negative Control

The structural checks require:

- zero train/validation row-index overlap;
- exactly-once validation coverage;
- training plus validation rows equal to the dataset size in every fold; and
- complete preprocessing fit on the training partition.

The negative control shuffles training labels within each fold, preserves
training-label counts, leaves validation labels unchanged, and refits both
substantive models. It tests whether the observed scores depend on the intended
feature-label association.

A passed split check and a low shuffled-label score do not prove the absence of
all leakage. External provenance, duplicated entities, time leakage, or target
encoding outside the implemented row-index checks can remain undetected.

## Probability Calibration

Uncalibrated and sigmoid-calibrated probabilities are compared on the same
five outer folds. Each calibrator uses three stratified inner folds drawn only
from its outer training partition.

Recorded metrics are:

- accuracy;
- multiclass log loss;
- multiclass Brier score; and
- ten-bin top-label expected calibration error.

The workbench saves cross-fitted probability vectors for every source row,
fold-level metrics, paired method differences, and the reliability-bin values
shown in the calibration figure.

Only sigmoid calibration and top-label calibration are evaluated. The design
does not compare isotonic calibration, repeated partitions, classwise
calibration, or an independent deployment sample.

## Validation Robustness

The models are evaluated on validation-only perturbations while their training
data and validation labels remain unchanged.

### Missing Values

Observed validation feature cells are selected without replacement at rates of
0%, 10%, 25%, and 50%. Existing missing cells remain missing and do not count
as injected cells.

### Gaussian Noise

Zero-mean noise is added to observed validation cells at 0, 0.25, 0.5, and 1.0
times each outer training fold's population standard deviation per feature.
Both models receive the same perturbed validation matrix.

Every condition is compared with its unperturbed score on the same fold.
Affected-cell counts, realized fractions, seeds, feature scales, metrics, and
paired differences are retained.

These synthetic perturbations are sensitivity checks. They do not model a
measured sensor, missingness mechanism, acquisition pipeline, or production
drift.

## Class-Imbalance Sensitivity

Chinstrap is selected as the globally least frequent class. Inside each outer
training fold, the experiment retains 100%, 75%, 50%, or 25% of its rows
without replacement and keeps every Adelie and Gentoo row. The same retained
sample is used for both models.

Validation rows, validation labels, and validation class counts do not change.
The artifacts record:

- original and retained target counts;
- before/after training shares;
- deterministic resampling seeds;
- retained-row SHA-256 signatures;
- aggregate metrics and per-class recall; and
- paired differences from full retention.

Downsampling changes both prevalence and training-set size. This experiment
does not isolate those effects or evaluate mitigation strategies.

## Cross-Experiment Summary

A fixed policy selects 25 representative contrasts across:

1. primary model comparison;
2. feature ablation;
3. shuffled-label control;
4. probability calibration;
5. validation robustness; and
6. class-imbalance sensitivity.

Every row retains reference and condition fold statistics, the raw
condition-minus-reference difference, a direction-aligned preferred effect,
the source artifact, and an interpretation boundary. Positive preferred
effects favor the condition for both higher- and lower-is-better metrics.

The summary is a navigation artifact, not a statistical meta-analysis.
Preferred effects are not standardized and must not be compared across
different metric scales.

## Evaluation Sequence

1. Load the checksum-verified CSV and preserve its source-row index.
2. Create the stratified holdout and shared outer folds.
3. Fit median imputation and standardization inside each training partition.
4. Evaluate the dummy, logistic-regression, and KNN models on the holdout.
5. Record row-level predictions and the confusion matrix.
6. Evaluate all models on the shared cross-validation folds.
7. Evaluate the three feature sets on the same folds.
8. Check split overlap and validation coverage.
9. Run the within-fold shuffled-training-label negative control.
10. Compare uncalibrated and sigmoid-calibrated probabilities.
11. Save cross-fitted probabilities and reliability-bin evidence.
12. Apply deterministic missing-value and Gaussian-noise perturbations to
    validation features only.
13. Downsample the least frequent training class at four fixed retention
    levels.
14. Calculate paired fold-level differences for every controlled comparison.
15. Select the fixed cross-experiment navigation summary.
16. Record the installed CLI, Python API, report schemas, compatibility, and
    artifact inventory.
17. Atomically write JSON, CSV, and PNG artifacts.
18. Atomically write the SHA-256 manifest for all 27 documented artifacts.
