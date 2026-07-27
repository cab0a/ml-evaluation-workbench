# Version 0.6 Evaluation Result

## Question

How sensitive are the fixed logistic-regression and KNN classifiers to
controlled missing-value injection and Gaussian feature noise under shared
outer folds?

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
- Missing-value rates: 0%, 10%, 25%, and 50% of observed validation feature
  cells, selected without replacement
- Gaussian-noise levels: 0, 0.25, 0.5, and 1.0 times each outer training
  fold's population standard deviation per feature
- Perturbation scope: validation features only; training data and validation
  labels remain unchanged
- Random state: 42; label-shuffle seed is 42 plus the one-based fold number

The majority-class dummy remains in the primary comparison but is excluded
from feature ablation because it does not use input features.

The v0.4 feature-ablation and leakage evidence and the v0.5 calibration
evidence are retained as context. The v0.6 comparison does not tune the models
or select perturbation levels from the reported scores.

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

## Missing-Value and Noise Robustness

All values below are mean macro F1 across the same five outer validation
folds. The unperturbed rows reproduce the primary cross-validation scores.

| Perturbation | Severity | Logistic Regression | 5-Nearest Neighbors |
| --- | ---: | ---: | ---: |
| Injected missing values | 0% | 0.928 | 0.949 |
| Injected missing values | 10% | 0.845 | 0.870 |
| Injected missing values | 25% | 0.770 | 0.759 |
| Injected missing values | 50% | 0.624 | 0.625 |
| Gaussian feature noise | 0.00x | 0.928 | 0.949 |
| Gaussian feature noise | 0.25x | 0.910 | 0.920 |
| Gaussian feature noise | 0.50x | 0.860 | 0.839 |
| Gaussian feature noise | 1.00x | 0.692 | 0.673 |

At 50% injected missingness, paired mean macro-F1 differences from the
unperturbed scores are -0.304 for logistic regression and -0.324 for KNN.
At 1.0x Gaussian noise, the corresponding differences are -0.237 and -0.277.
Both models show progressively lower mean scores under these fixed severity
levels, with no consistent robustness advantage across the two perturbation
types.

![Missing-value and noise robustness](robustness.png)

The missing-value experiment injects an exact nearest-integer cell count into
previously observed validation cells. Existing missing cells remain missing
and are not counted as injected. Gaussian noise is applied only to observed
validation cells, and its per-feature scale is calculated from the
corresponding outer training fold. Each perturbed validation matrix is shared
by both models.

`robustness_folds.csv` records seeds, eligible and affected cells, realized
fractions, noise scales, and fold metrics. `robustness_summary.csv` records
means, fold standard deviations, and paired differences from the unperturbed
condition. `robustness_diagnostics.json` fixes the protocol and interpretation
boundary.

## Continuity Artifacts

The deterministic holdout, three-model comparison, fold-level scores,
row-level predictions, and logistic-regression confusion matrix from v0.3 are
retained alongside the v0.4 diagnostics and v0.5 calibration evidence, then
regenerated under report schema version 6.

![Cross-validation fold scores](cross_validation_scores.png)

![Logistic regression confusion matrix](confusion_matrix.png)

## Interpretation Boundary

Version 0.6 adds controlled sensitivity evidence, not a deployment-robustness
guarantee. The robustness and calibration comparisons, ablation, and negative
control reuse one five-fold outer partition and one public dataset revision.
Their standard deviations describe observed fold variation and are not
confidence intervals.

The reliability diagram uses top-label confidence rather than classwise
calibration and depends on ten fixed bins. Empty or sparsely populated bins
carry little evidence. Only sigmoid calibration with three inner folds is
evaluated; isotonic calibration, repeated splits, independent datasets, and
deployment shift are outside this version's scope.

The synthetic perturbations do not model a measured sensor, acquisition
pipeline, missingness mechanism, or production data drift. Missing cells are
selected independently across the two features, and Gaussian noise is
independent across cells. Other patterns—including structured missingness,
bias, outliers, correlated noise, label noise, and train-time corruption—can
produce materially different results.

The committed artifacts demonstrate a deterministic implementation of feature
ablation, split-integrity checks, a negative control, cross-fitted probability
diagnostics, and synthetic robustness experiments. They are not a benchmark
claim, causal feature-importance result, deployment guarantee, or ecological
conclusion.

## Reproduction

From the repository root:

```bash
python examples/run_demo.py
python examples/run_demo.py --verify-only
```
