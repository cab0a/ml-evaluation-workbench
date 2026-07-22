# Version 0.1 Evaluation Result

## Question

How much improvement over a majority-class baseline can a linear classifier
provide when it uses only bill length and bill depth to separate the three
species in the pinned Palmer Penguins dataset?

## Controlled Setup

- Dataset rows: 344
- Selected features: `bill_length_mm`, `bill_depth_mm`
- Missing selected feature cells: 4
- Split: stratified 75/25 holdout
- Random state: 42
- Training rows: 258
- Test rows: 86
- Preprocessing: training-partition median imputation and standardization
- Baseline: most-frequent `DummyClassifier`
- Prototype: `LogisticRegression` with a maximum of 1,000 iterations

## Summary

| Model | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.441860 | 0.333333 | 0.204301 |
| Logistic regression | 0.941860 | 0.919671 | 0.923661 |

The dummy model predicts Adelie for every holdout row. Its raw accuracy reflects
the majority-class share, while balanced accuracy and macro F1 expose its lack
of useful recall for Chinstrap and Gentoo.

Logistic regression improves accuracy by 0.500000 and macro F1 by 0.719360. It
correctly classifies 81 of 86 holdout rows. Recall is 1.000000 for Adelie,
0.823529 for Chinstrap, and 0.935484 for Gentoo.

## Error Pattern

The logistic-regression confusion matrix is:

| Actual / Predicted | Adelie | Chinstrap | Gentoo |
| --- | ---: | ---: | ---: |
| Adelie | 38 | 0 | 0 |
| Chinstrap | 1 | 14 | 2 |
| Gentoo | 0 | 2 | 29 |

All five errors involve Chinstrap: one Chinstrap is predicted as Adelie, two
Chinstraps are predicted as Gentoo, and two Gentoos are predicted as
Chinstrap. This is consistent with overlap in a two-dimensional bill-feature
space; the result does not establish why those individual errors occur.

![Logistic regression confusion matrix](confusion_matrix.png)

## Interpretation Boundary

This result shows that the two bill measurements carry substantial separable
signal under one random holdout. It does not establish expected accuracy on a
new island, a later year, another measurement protocol, or field observations.
Cross-validation, group-aware evaluation, calibration, and robustness tests
remain outside v0.1 and are listed in the roadmap.

The committed artifacts are functional evidence for a deterministic pipeline,
not a benchmark claim or an ecological conclusion.

## Reproduction

From the repository root:

```bash
python examples/run_demo.py
python examples/run_demo.py --verify-only
```
