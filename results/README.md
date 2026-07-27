# Version 0.5 Evaluation Result

## Question

How do sigmoid-calibrated probabilities compare with uncalibrated probabilities
for the fixed logistic-regression and KNN classifiers under shared outer
folds?

## Controlled Setup

- Dataset rows: 344
- Candidate features: `bill_length_mm`, `bill_depth_mm`
- Feature sets: bill length only, bill depth only, and both measurements
- Models: fixed logistic regression and 5-nearest neighbors
- Cross-validation: the same five shuffled stratified folds for every feature
  set, model, and diagnostic
- Preprocessing: median imputation and standardization fitted inside each
  training partition
- Selection policy: diagnostic only; no feature set or model is selected from
  these scores
- Negative control: training labels shuffled within each fold while validation
  labels remain unchanged
- Probability methods: uncalibrated and sigmoid calibrated
- Calibration: three stratified inner folds drawn only from each outer
  training partition
- Probability metrics: multiclass log loss, multiclass Brier score, and
  ten-bin top-label expected calibration error
- Random state: 42; label-shuffle seed is 42 plus the one-based fold number

The majority-class dummy remains in the primary comparison but is excluded
from feature ablation because it does not use input features.

The v0.4 feature-ablation and leakage evidence is retained as context. The
v0.5 comparison does not tune the models or select a calibration method from
the reported scores.

## Primary Comparison Reference

| Model | Accuracy, mean ± std | Balanced Accuracy, mean ± std | Macro F1, mean ± std |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.441858 ± 0.006490 | 0.333333 ± 0.000000 | 0.204291 ± 0.002081 |
| Logistic regression | 0.944800 ± 0.033510 | 0.923659 ± 0.043207 | 0.928288 ± 0.041342 |
| 5-nearest neighbors | 0.959292 ± 0.010882 | 0.947829 ± 0.022578 | 0.949112 ± 0.017756 |

These two-feature scores are the reference for the ablation differences below.

## Feature Ablation

| Feature set | Model | Accuracy | Balanced Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Bill length only | Logistic regression | 0.744288 ± 0.022538 | 0.622724 ± 0.011612 | 0.574987 ± 0.012112 |
| Bill length only | 5-nearest neighbors | 0.732822 ± 0.049770 | 0.653446 ± 0.061832 | 0.647485 ± 0.072626 |
| Bill depth only | Logistic regression | 0.755754 ± 0.031414 | 0.627516 ± 0.024334 | 0.568900 ± 0.024325 |
| Bill depth only | 5-nearest neighbors | 0.741176 ± 0.050273 | 0.664400 ± 0.042212 | 0.664812 ± 0.049604 |
| Both measurements | Logistic regression | 0.944800 ± 0.033510 | 0.923659 ± 0.043207 | 0.928288 ± 0.041342 |
| Both measurements | 5-nearest neighbors | 0.959292 ± 0.010882 | 0.947829 ± 0.022578 | 0.949112 ± 0.017756 |

The paired macro-F1 differences relative to both measurements are:

| Removed measurement | Logistic Difference | KNN Difference |
| --- | ---: | ---: |
| Bill depth | -0.353300 | -0.301627 |
| Bill length | -0.359387 | -0.284300 |

Both single-feature configurations are substantially below the two-feature
reference for both models. Under these fixed folds and configurations, bill
length and bill depth provide complementary predictive signal. The result does
not establish causal importance, measurement value outside this dataset, or
an optimal feature subset.

![Feature-ablation macro F1](feature_ablation_scores.png)

## Split-Integrity Diagnostics

All five folds satisfy the implemented structural checks:

- Maximum train/validation overlap: 0 rows
- Validation coverage minimum: 1
- Validation coverage maximum: 1
- Training plus validation rows per fold: 344
- Complete pipeline fit scope: training partition

These checks verify the row-index properties of this splitter output. They do
not inspect external provenance, repeated biological individuals, or hidden
relationships not represented by a row index.

## Shuffled-Training-Label Negative Control

| Model | Observed Macro F1 | Shuffled Macro F1 | Observed − Shuffled |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.928288 | 0.253790 ± 0.052453 | 0.674498 |
| 5-nearest neighbors | 0.949112 | 0.297675 ± 0.053507 | 0.651437 |

The negative control destroys the association between features and training
labels inside each fold while preserving the training-label counts and leaving
validation labels unchanged. Both shuffled-label means are substantially below
the observed means, and the difference is positive in every fold.

This outcome is consistent with the primary scores depending on the intended
feature-label association. It is not proof that the project is free from every
form of leakage. A shuffled-label control can miss duplicated entities,
provenance leakage, time leakage, or a feature that directly encodes the target
in a way not exercised by this small dataset.

## Probability Calibration

All values below are means across the same five outer validation folds.
Multiclass Brier score is the mean summed squared error across all class
probabilities. Top-label ECE uses ten equal-width confidence bins.

| Model and method | Accuracy | Log Loss | Multiclass Brier | Top-Label ECE |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression, uncalibrated | 0.944800 | 0.138743 | 0.072342 | 0.063897 |
| Logistic regression, sigmoid | 0.936104 | 0.236341 | 0.116650 | 0.113640 |
| 5-nearest neighbors, uncalibrated | 0.959292 | 0.280329 | 0.056488 | 0.030205 |
| 5-nearest neighbors, sigmoid | 0.965089 | 0.132767 | 0.057062 | 0.071316 |

For KNN, sigmoid calibration reduces mean log loss by 0.147563 and slightly
increases mean accuracy, while its Brier score changes by +0.000574 and
top-label ECE increases by 0.041112. For logistic regression, sigmoid
calibration increases log loss by 0.097598 and also worsens the two other
recorded probability-quality metrics.

This mixed outcome is the main result: calibration quality depends on the
model, metric, sample, and calibration design. The KNN log-loss reduction
should not be generalized into a claim that sigmoid calibration is uniformly
better.

![Top-label reliability diagram](probability_calibration.png)

`probability_calibration_predictions.csv` provides all 1,376 cross-fitted
model-method predictions. `probability_calibration_bins.csv` records the
points shown above, including empty bins. Fold-level scores and paired
differences are available in `probability_calibration_folds.csv`,
`probability_calibration_summary.csv`, and `metrics.json`.

## Continuity Artifacts

The deterministic holdout, three-model comparison, fold-level scores,
row-level predictions, and logistic-regression confusion matrix from v0.3 are
retained alongside the v0.4 diagnostics and regenerated under report schema
version 5.

![Cross-validation fold scores](cross_validation_scores.png)

![Logistic regression confusion matrix](confusion_matrix.png)

## Interpretation Boundary

Version 0.5 adds calibration evidence, not a production probability-quality
guarantee. The calibration comparison, ablation, and negative control reuse one
five-fold outer partition and one public dataset revision. Their standard
deviations describe observed fold variation and are not confidence intervals.

The reliability diagram uses top-label confidence rather than classwise
calibration and depends on ten fixed bins. Empty or sparsely populated bins
carry little evidence. Only sigmoid calibration with three inner folds is
evaluated; isotonic calibration, repeated splits, independent datasets, and
deployment shift are outside this version's scope.

The committed artifacts demonstrate a deterministic implementation of feature
ablation, split-integrity checks, a negative control, and cross-fitted
probability diagnostics. They are not a benchmark claim, causal
feature-importance result, deployment guarantee, or ecological conclusion.

## Reproduction

From the repository root:

```bash
python examples/run_demo.py
python examples/run_demo.py --verify-only
```
