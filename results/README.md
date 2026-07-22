# Version 0.2 Evaluation Result

## Question

How much improvement over a majority-class baseline can a linear classifier
provide when it uses only bill length and bill depth to separate the three
species in the pinned Palmer Penguins dataset, and how much do its scores vary
across five stratified folds?

## Controlled Setup

- Dataset rows: 344
- Selected features: `bill_length_mm`, `bill_depth_mm`
- Missing selected feature cells: 4
- Holdout: stratified 75/25 split
- Cross-validation: five stratified folds with shuffling
- Random state: 42 for both evaluation paths
- Holdout training rows: 258
- Holdout test rows: 86
- Fold training rows: 275 or 276
- Fold validation rows: 68 or 69
- Preprocessing: median imputation and standardization fitted inside each
  training partition
- Baseline: most-frequent `DummyClassifier`
- Prototype: `LogisticRegression` with a maximum of 1,000 iterations

Both models use the same holdout and cross-validation partitions. Every
cross-validation fold refits the full preprocessing and classifier pipeline.

## Holdout Summary

| Model | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.441860 | 0.333333 | 0.204301 |
| Logistic regression | 0.941860 | 0.919671 | 0.923661 |

The dummy model predicts Adelie for every holdout row. Its raw accuracy reflects
the majority-class share, while balanced accuracy and macro F1 expose its lack
of useful recall for Chinstrap and Gentoo.

Logistic regression improves holdout accuracy by 0.500000 and macro F1 by
0.719360. It correctly classifies 81 of 86 holdout rows. Recall is 1.000000 for
Adelie, 0.823529 for Chinstrap, and 0.935484 for Gentoo.

## Cross-Validation Summary

The standard deviations below are population standard deviations across the
five observed fold scores.

| Model | Accuracy, mean ± std | Balanced Accuracy, mean ± std | Macro F1, mean ± std |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.441858 ± 0.006490 | 0.333333 ± 0.000000 | 0.204291 ± 0.002081 |
| Logistic regression | 0.944800 ± 0.033510 | 0.923659 ± 0.043207 | 0.928288 ± 0.041342 |

Logistic-regression macro F1 ranges from 0.856905 to 0.981414. Its mean paired
macro-F1 gain over the dummy model is 0.723996, with a minimum gain of 0.654885
across the five shared folds.

| Fold | Accuracy | Balanced Accuracy | Macro F1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.942029 | 0.897436 | 0.916157 |
| 2 | 0.985507 | 0.986111 | 0.981414 |
| 3 | 0.956522 | 0.941270 | 0.946570 |
| 4 | 0.884058 | 0.858095 | 0.856905 |
| 5 | 0.955882 | 0.935385 | 0.940392 |

Fold 4 produces the lowest logistic-regression scores. Its Chinstrap recall is
0.714286, compared with 0.900000 Adelie recall and 0.960000 Gentoo recall. This
supports an error-review hypothesis that Chinstrap remains the least stable
class under the constrained two-feature design. It does not identify a causal
reason for the errors.

![Cross-validation fold scores](cross_validation_scores.png)

## Holdout Error Pattern

The logistic-regression holdout confusion matrix is:

| Actual / Predicted | Adelie | Chinstrap | Gentoo |
| --- | ---: | ---: | ---: |
| Adelie | 38 | 0 | 0 |
| Chinstrap | 1 | 14 | 2 |
| Gentoo | 0 | 2 | 29 |

All five holdout errors involve Chinstrap: one Chinstrap is predicted as
Adelie, two Chinstraps are predicted as Gentoo, and two Gentoos are predicted
as Chinstrap.

![Logistic regression confusion matrix](confusion_matrix.png)

## Interpretation Boundary

The holdout score is close to the five-fold mean, and logistic regression
exceeds the dummy baseline in every observed fold. This is stronger evidence
than a single split alone, but it remains specific to one dataset revision,
one feature pair, one cross-validation partition, and two baseline models.

The five training partitions overlap, so their scores are not independent.
The reported standard deviations describe observed fold variation; they are
not confidence intervals. The evaluation does not establish transfer to a new
island, year, collection protocol, or field setting.

The committed artifacts are functional evidence for a deterministic evaluation
pipeline, not a benchmark claim or an ecological conclusion.

## Reproduction

From the repository root:

```bash
python examples/run_demo.py
python examples/run_demo.py --verify-only
```
