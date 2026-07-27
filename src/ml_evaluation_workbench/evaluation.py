"""Deterministic holdout and cross-validation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = (
    "bill_length_mm",
    "bill_depth_mm",
)
FEATURES = NUMERIC_FEATURES
TARGET = "species"
SCORE_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")
MODEL_NAMES = ("dummy", "logistic_regression", "knn")
MODEL_ROLES = {
    "dummy": "reference_baseline",
    "logistic_regression": "linear_baseline",
    "knn": "nonlinear_comparator",
}
COMPARISON_PAIRS = (
    ("logistic_regression", "dummy"),
    ("knn", "dummy"),
    ("knn", "logistic_regression"),
)
DIAGNOSTIC_MODEL_NAMES = ("logistic_regression", "knn")
FEATURE_SETS = {
    "bill_length_only": ("bill_length_mm",),
    "bill_depth_only": ("bill_depth_mm",),
    "both_bill_measurements": FEATURES,
}
REFERENCE_FEATURE_SET = "both_bill_measurements"
CALIBRATION_NAMES = ("uncalibrated", "sigmoid")
PROBABILITY_SCORE_NAMES = (
    "accuracy",
    "log_loss",
    "multiclass_brier",
    "top_label_ece",
)
CALIBRATION_BIN_COUNT = 10
MISSING_RATES = (0.0, 0.1, 0.25, 0.5)
NOISE_STD_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)
ROBUSTNESS_PERTURBATIONS = ("missing_values", "gaussian_noise")


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    cross_validation_folds: pd.DataFrame
    model_comparison: pd.DataFrame
    feature_ablation_folds: pd.DataFrame
    feature_ablation_summary: pd.DataFrame
    leakage_diagnostic_folds: pd.DataFrame
    leakage_diagnostics: dict[str, Any]
    probability_calibration_folds: pd.DataFrame
    probability_calibration_summary: pd.DataFrame
    probability_calibration_predictions: pd.DataFrame
    probability_calibration_bins: pd.DataFrame
    robustness_folds: pd.DataFrame
    robustness_summary: pd.DataFrame
    robustness_diagnostics: dict[str, Any]
    confusion: np.ndarray
    labels: tuple[str, ...]


def _pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _models(random_state: int) -> dict[str, Pipeline]:
    return {
        "dummy": _pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "knn": _pipeline(
            KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                metric="minkowski",
                p=2,
            )
        ),
    }


def _model_configurations(random_state: int) -> dict[str, dict[str, Any]]:
    return {
        "dummy": {
            "classifier": "DummyClassifier",
            "strategy": "most_frequent",
        },
        "logistic_regression": {
            "classifier": "LogisticRegression",
            "max_iter": 1000,
            "random_state": random_state,
        },
        "knn": {
            "classifier": "KNeighborsClassifier",
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "metric": "minkowski",
            "p": 2,
        },
    }


def _model_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    recalls = recall_score(
        actual,
        predicted,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(actual, predicted)), 6
        ),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro")), 6
        ),
        "per_class_recall": {
            label: round(float(value), 6)
            for label, value in zip(labels, recalls, strict=True)
        },
    }


def _score_summary(values: pd.Series) -> dict[str, float]:
    scores = values.to_numpy(dtype=float)
    return {
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores, ddof=0)), 6),
        "min": round(float(np.min(scores)), 6),
        "max": round(float(np.max(scores)), 6),
    }


def _stratified_splits(
    frame: pd.DataFrame,
    *,
    random_state: int,
    cv_folds: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    return tuple(splitter.split(features, target))


def _cross_validate(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    random_state: int,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    rows: list[dict[str, Any]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splits,
        start=1,
    ):
        for model_name, model in _models(random_state).items():
            model.fit(features.iloc[train_indices], target.iloc[train_indices])
            predicted = model.predict(features.iloc[validation_indices])
            scores = _model_metrics(
                target.iloc[validation_indices],
                predicted,
                labels,
            )
            row: dict[str, Any] = {
                "fold": fold,
                "model": model_name,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                **{
                    score_name: scores[score_name]
                    for score_name in SCORE_NAMES
                },
            }
            row.update(
                {
                    f"recall_{label.lower()}": value
                    for label, value in scores["per_class_recall"].items()
                }
            )
            rows.append(row)

    fold_scores = pd.DataFrame(rows)
    model_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in MODEL_NAMES:
        model_rows = fold_scores[fold_scores["model"] == model_name]
        model_summaries[model_name] = {
            score_name: _score_summary(model_rows[score_name])
            for score_name in SCORE_NAMES
        }

    paired_differences: dict[str, dict[str, dict[str, float]]] = {}
    for left_model, right_model in COMPARISON_PAIRS:
        comparison_name = f"{left_model}_minus_{right_model}"
        paired_differences[comparison_name] = {}
        for score_name in SCORE_NAMES:
            by_model = fold_scores.pivot(
                index="fold",
                columns="model",
                values=score_name,
            )
            paired_differences[comparison_name][score_name] = _score_summary(
                by_model[left_model] - by_model[right_model]
            )

    summary = {
        "strategy": "stratified_k_fold",
        "folds": len(splits),
        "shuffle": True,
        "random_state": random_state,
        "standard_deviation": "population_across_folds",
        "models": model_summaries,
        "paired_difference": paired_differences,
    }
    return fold_scores, summary


def _metric_differences(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, float]:
    return {
        score_name: round(left[score_name] - right[score_name], 6)
        for score_name in SCORE_NAMES
    }


def _model_comparison_table(
    model_metrics: dict[str, dict[str, Any]],
    cross_validation: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        holdout = model_metrics[model_name]
        cross_validation_scores = cross_validation["models"][model_name]
        row: dict[str, Any] = {
            "model": model_name,
            "role": MODEL_ROLES[model_name],
        }
        row.update(
            {
                f"holdout_{score_name}": holdout[score_name]
                for score_name in SCORE_NAMES
            }
        )
        for score_name in SCORE_NAMES:
            row[f"cv_{score_name}_mean"] = cross_validation_scores[
                score_name
            ]["mean"]
            row[f"cv_{score_name}_std"] = cross_validation_scores[
                score_name
            ]["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_ablation(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    random_state: int,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = frame[TARGET]
    rows: list[dict[str, Any]] = []
    for feature_set, feature_names in FEATURE_SETS.items():
        features = frame[list(feature_names)]
        for fold, (train_indices, validation_indices) in enumerate(
            splits,
            start=1,
        ):
            for model_name in DIAGNOSTIC_MODEL_NAMES:
                model = _models(random_state)[model_name]
                model.fit(
                    features.iloc[train_indices],
                    target.iloc[train_indices],
                )
                predicted = model.predict(features.iloc[validation_indices])
                scores = _model_metrics(
                    target.iloc[validation_indices],
                    predicted,
                    labels,
                )
                row: dict[str, Any] = {
                    "feature_set": feature_set,
                    "features": "|".join(feature_names),
                    "fold": fold,
                    "model": model_name,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    **{
                        score_name: scores[score_name]
                        for score_name in SCORE_NAMES
                    },
                }
                row.update(
                    {
                        f"recall_{label.lower()}": value
                        for label, value in scores[
                            "per_class_recall"
                        ].items()
                    }
                )
                rows.append(row)

    fold_scores = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    feature_set_metrics: dict[str, Any] = {}
    for feature_set, feature_names in FEATURE_SETS.items():
        feature_set_metrics[feature_set] = {
            "features": list(feature_names),
            "models": {},
        }
        for model_name in DIAGNOSTIC_MODEL_NAMES:
            model_rows = fold_scores[
                (fold_scores["feature_set"] == feature_set)
                & (fold_scores["model"] == model_name)
            ].set_index("fold")
            reference_rows = fold_scores[
                (fold_scores["feature_set"] == REFERENCE_FEATURE_SET)
                & (fold_scores["model"] == model_name)
            ].set_index("fold")
            score_metrics: dict[str, Any] = {}
            summary_row: dict[str, Any] = {
                "feature_set": feature_set,
                "features": "|".join(feature_names),
                "model": model_name,
            }
            for score_name in SCORE_NAMES:
                score_summary = _score_summary(model_rows[score_name])
                paired_difference = _score_summary(
                    model_rows[score_name] - reference_rows[score_name]
                )
                score_metrics[score_name] = {
                    **score_summary,
                    "paired_difference_vs_both": paired_difference,
                }
                summary_row[f"{score_name}_mean"] = score_summary["mean"]
                summary_row[f"{score_name}_std"] = score_summary["std"]
                summary_row[
                    f"{score_name}_difference_vs_both_mean"
                ] = paired_difference["mean"]
            feature_set_metrics[feature_set]["models"][
                model_name
            ] = score_metrics
            summary_rows.append(summary_row)

    summary = {
        "strategy": "shared_stratified_k_fold",
        "folds": len(splits),
        "models": list(DIAGNOSTIC_MODEL_NAMES),
        "reference_feature_set": REFERENCE_FEATURE_SET,
        "selection_policy": "diagnostic_only_no_model_selection",
        "feature_sets": feature_set_metrics,
    }
    return fold_scores, pd.DataFrame(summary_rows), summary


def _leakage_diagnostics(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    random_state: int,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
    observed_fold_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    validation_coverage = np.zeros(len(frame), dtype=int)
    rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, int]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splits,
        start=1,
    ):
        overlap_rows = int(
            np.intersect1d(train_indices, validation_indices).size
        )
        validation_coverage[validation_indices] += 1
        partition_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "train_validation_overlap_rows": overlap_rows,
            }
        )
        shuffled_target = target.iloc[train_indices].to_numpy(copy=True)
        rng = np.random.default_rng(random_state + fold)
        rng.shuffle(shuffled_target)

        for model_name in DIAGNOSTIC_MODEL_NAMES:
            model = _models(random_state)[model_name]
            model.fit(features.iloc[train_indices], shuffled_target)
            predicted = model.predict(features.iloc[validation_indices])
            shuffled_scores = _model_metrics(
                target.iloc[validation_indices],
                predicted,
                labels,
            )
            observed_row = observed_fold_scores[
                (observed_fold_scores["fold"] == fold)
                & (observed_fold_scores["model"] == model_name)
            ].iloc[0]
            row: dict[str, Any] = {
                "fold": fold,
                "model": model_name,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "train_validation_overlap_rows": overlap_rows,
            }
            for score_name in SCORE_NAMES:
                shuffled_value = shuffled_scores[score_name]
                observed_value = float(observed_row[score_name])
                row[f"shuffled_{score_name}"] = shuffled_value
                row[f"observed_{score_name}"] = observed_value
                row[
                    f"observed_minus_shuffled_{score_name}"
                ] = round(observed_value - shuffled_value, 6)
            row.update(
                {
                    f"shuffled_recall_{label.lower()}": value
                    for label, value in shuffled_scores[
                        "per_class_recall"
                    ].items()
                }
            )
            rows.append(row)

    diagnostic_folds = pd.DataFrame(rows)
    model_summaries: dict[str, Any] = {}
    for model_name in DIAGNOSTIC_MODEL_NAMES:
        model_rows = diagnostic_folds[
            diagnostic_folds["model"] == model_name
        ]
        model_summaries[model_name] = {
            score_name: {
                "shuffled": _score_summary(
                    model_rows[f"shuffled_{score_name}"]
                ),
                "observed_minus_shuffled": _score_summary(
                    model_rows[
                        f"observed_minus_shuffled_{score_name}"
                    ]
                ),
            }
            for score_name in SCORE_NAMES
        }

    maximum_overlap = max(
        row["train_validation_overlap_rows"] for row in partition_rows
    )
    coverage_minimum = int(validation_coverage.min())
    coverage_maximum = int(validation_coverage.max())
    all_rows_partitioned = all(
        row["train_rows"] + row["validation_rows"] == len(frame)
        for row in partition_rows
    )
    split_integrity_passed = (
        maximum_overlap == 0
        and coverage_minimum == 1
        and coverage_maximum == 1
        and all_rows_partitioned
    )
    diagnostics = {
        "interpretation": "negative_control_not_proof_of_no_leakage",
        "preprocessing_fit_scope": "training_partition_pipeline",
        "split_integrity": {
            "passed": split_integrity_passed,
            "folds": partition_rows,
            "maximum_train_validation_overlap_rows": maximum_overlap,
            "validation_coverage_minimum": coverage_minimum,
            "validation_coverage_maximum": coverage_maximum,
            "all_rows_partitioned_per_fold": all_rows_partitioned,
        },
        "shuffled_training_labels": {
            "scope": "within_each_training_fold",
            "seed_rule": "random_state_plus_fold_number",
            "validation_labels": "unchanged",
            "models": model_summaries,
        },
    }
    return diagnostic_folds, diagnostics


def _probability_metrics(
    actual: pd.Series,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    *,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> dict[str, float]:
    actual_values = actual.astype(str).to_numpy()
    label_to_index = {
        label: index for index, label in enumerate(labels)
    }
    actual_indices = np.array(
        [label_to_index[value] for value in actual_values],
        dtype=int,
    )
    predicted_indices = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted_indices == actual_indices
    one_hot = np.eye(len(labels), dtype=float)[actual_indices]
    bin_indices = np.minimum(
        (confidence * bin_count).astype(int),
        bin_count - 1,
    )
    ece = 0.0
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(correct[mask]))
            - float(np.mean(confidence[mask]))
        )
    return {
        "accuracy": round(float(np.mean(correct)), 6),
        "log_loss": round(
            float(log_loss(actual_values, probabilities, labels=list(labels))),
            6,
        ),
        "multiclass_brier": round(
            float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            6,
        ),
        "top_label_ece": round(ece, 6),
    }


def _aligned_probabilities(
    model: Any,
    features: pd.DataFrame,
    labels: tuple[str, ...],
) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    model_labels = [str(value) for value in model.classes_]
    column_indices = [model_labels.index(label) for label in labels]
    return probabilities[:, column_indices]


def _calibration_bins(
    predictions: pd.DataFrame,
    *,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in DIAGNOSTIC_MODEL_NAMES:
        for calibration_name in CALIBRATION_NAMES:
            model_rows = predictions[
                (predictions["model"] == model_name)
                & (predictions["calibration"] == calibration_name)
            ]
            confidence = model_rows["confidence"].to_numpy(dtype=float)
            correct = model_rows["correct"].to_numpy(dtype=bool)
            bin_indices = np.minimum(
                (confidence * bin_count).astype(int),
                bin_count - 1,
            )
            for bin_index in range(bin_count):
                mask = bin_indices == bin_index
                sample_count = int(np.sum(mask))
                rows.append(
                    {
                        "model": model_name,
                        "calibration": calibration_name,
                        "bin": bin_index + 1,
                        "lower_bound": round(bin_index / bin_count, 6),
                        "upper_bound": round(
                            (bin_index + 1) / bin_count,
                            6,
                        ),
                        "sample_count": sample_count,
                        "mean_confidence": (
                            round(float(np.mean(confidence[mask])), 6)
                            if sample_count
                            else None
                        ),
                        "empirical_accuracy": (
                            round(float(np.mean(correct[mask])), 6)
                            if sample_count
                            else None
                        ),
                        "absolute_gap": (
                            round(
                                abs(
                                    float(np.mean(correct[mask]))
                                    - float(np.mean(confidence[mask]))
                                ),
                                6,
                            )
                            if sample_count
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _probability_calibration(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    random_state: int,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
    calibration_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splits,
        start=1,
    ):
        inner_splitter = StratifiedKFold(
            n_splits=calibration_folds,
            shuffle=True,
            random_state=random_state + fold,
        )
        for model_name in DIAGNOSTIC_MODEL_NAMES:
            base_model = _models(random_state)[model_name]
            models = {
                "uncalibrated": base_model,
                "sigmoid": CalibratedClassifierCV(
                    estimator=_models(random_state)[model_name],
                    method="sigmoid",
                    cv=inner_splitter,
                    ensemble=True,
                ),
            }
            for calibration_name, model in models.items():
                model.fit(
                    features.iloc[train_indices],
                    target.iloc[train_indices],
                )
                probabilities = _aligned_probabilities(
                    model,
                    features.iloc[validation_indices],
                    labels,
                )
                scores = _probability_metrics(
                    target.iloc[validation_indices],
                    probabilities,
                    labels,
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "calibration": calibration_name,
                        "train_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                        **scores,
                    }
                )
                predicted_indices = np.argmax(probabilities, axis=1)
                predicted_labels = np.asarray(labels)[predicted_indices]
                confidence = np.max(probabilities, axis=1)
                for row_offset, frame_index in enumerate(validation_indices):
                    row: dict[str, Any] = {
                        "source_row": int(frame_index) + 2,
                        "fold": fold,
                        "model": model_name,
                        "calibration": calibration_name,
                        "actual": str(target.iloc[frame_index]),
                        "predicted": str(predicted_labels[row_offset]),
                        "confidence": round(
                            float(confidence[row_offset]),
                            6,
                        ),
                        "correct": bool(
                            predicted_labels[row_offset]
                            == str(target.iloc[frame_index])
                        ),
                    }
                    row.update(
                        {
                            f"probability_{label.lower()}": round(
                                float(probabilities[row_offset, label_index]),
                                6,
                            )
                            for label_index, label in enumerate(labels)
                        }
                    )
                    prediction_rows.append(row)

    fold_scores = pd.DataFrame(fold_rows)
    predictions = (
        pd.DataFrame(prediction_rows)
        .sort_values(["model", "calibration", "source_row"])
        .reset_index(drop=True)
    )
    summary_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    paired_differences: dict[str, Any] = {}
    for model_name in DIAGNOSTIC_MODEL_NAMES:
        model_summaries[model_name] = {}
        for calibration_name in CALIBRATION_NAMES:
            method_rows = fold_scores[
                (fold_scores["model"] == model_name)
                & (fold_scores["calibration"] == calibration_name)
            ]
            score_summaries = {
                score_name: _score_summary(method_rows[score_name])
                for score_name in PROBABILITY_SCORE_NAMES
            }
            model_summaries[model_name][
                calibration_name
            ] = score_summaries
            summary_row: dict[str, Any] = {
                "model": model_name,
                "calibration": calibration_name,
            }
            for score_name, score_summary in score_summaries.items():
                summary_row[f"{score_name}_mean"] = score_summary["mean"]
                summary_row[f"{score_name}_std"] = score_summary["std"]
            summary_rows.append(summary_row)

        by_method = fold_scores[
            fold_scores["model"] == model_name
        ].pivot(index="fold", columns="calibration")
        paired_differences[model_name] = {
            "sigmoid_minus_uncalibrated": {
                score_name: _score_summary(
                    by_method[score_name]["sigmoid"]
                    - by_method[score_name]["uncalibrated"]
                )
                for score_name in PROBABILITY_SCORE_NAMES
            }
        }

    bins = _calibration_bins(predictions)
    summary = {
        "strategy": "shared_outer_stratified_k_fold",
        "outer_folds": len(splits),
        "models": list(DIAGNOSTIC_MODEL_NAMES),
        "methods": list(CALIBRATION_NAMES),
        "inner_calibration": {
            "method": "sigmoid",
            "folds": calibration_folds,
            "scope": "outer_training_partition_only",
            "shuffle": True,
            "seed_rule": "random_state_plus_outer_fold",
            "ensemble": True,
        },
        "metrics": {
            "log_loss": "multiclass_cross_entropy_lower_is_better",
            "multiclass_brier": (
                "mean_sum_of_squared_class_probability_errors_lower_is_better"
            ),
            "top_label_ece": (
                "ten_equal_width_confidence_bins_lower_is_better"
            ),
            "accuracy": "argmax_class_accuracy_higher_is_better",
        },
        "models_summary": model_summaries,
        "paired_difference": paired_differences,
        "interpretation": (
            "diagnostic_comparison_not_a_guarantee_that_calibration_improves"
        ),
    }
    return (
        fold_scores,
        pd.DataFrame(summary_rows),
        predictions,
        bins,
        summary,
    )


def _inject_missing_values(
    features: pd.DataFrame,
    *,
    rate: float,
    seed: int,
) -> tuple[pd.DataFrame, int, int]:
    values = features.to_numpy(dtype=float, copy=True)
    observed_locations = np.argwhere(~np.isnan(values))
    eligible_cells = len(observed_locations)
    affected_cells = int(round(rate * eligible_cells))
    if affected_cells:
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            eligible_cells,
            size=affected_cells,
            replace=False,
        )
        selected_locations = observed_locations[selected]
        values[
            selected_locations[:, 0],
            selected_locations[:, 1],
        ] = np.nan
    return (
        pd.DataFrame(
            values,
            index=features.index,
            columns=features.columns,
        ),
        affected_cells,
        eligible_cells,
    )


def _add_gaussian_noise(
    features: pd.DataFrame,
    *,
    training_standard_deviation: pd.Series,
    multiplier: float,
    seed: int,
) -> tuple[pd.DataFrame, int, int, dict[str, float]]:
    values = features.to_numpy(dtype=float, copy=True)
    observed = ~np.isnan(values)
    eligible_cells = int(np.sum(observed))
    feature_noise_standard_deviation = (
        training_standard_deviation.to_numpy(dtype=float) * multiplier
    )
    affected_cells = eligible_cells if multiplier > 0.0 else 0
    if affected_cells:
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            loc=0.0,
            scale=feature_noise_standard_deviation,
            size=values.shape,
        )
        values[observed] += noise[observed]
    noise_scales = {
        feature_name: round(
            float(feature_noise_standard_deviation[index]),
            6,
        )
        for index, feature_name in enumerate(features.columns)
    }
    return (
        pd.DataFrame(
            values,
            index=features.index,
            columns=features.columns,
        ),
        affected_cells,
        eligible_cells,
        noise_scales,
    )


def _robustness_evaluation(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    random_state: int,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    rows: list[dict[str, Any]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splits,
        start=1,
    ):
        training_features = features.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        training_standard_deviation = training_features.std(
            axis=0,
            skipna=True,
            ddof=0,
        ).fillna(0.0)
        conditions: list[dict[str, Any]] = []
        for severity_index, rate in enumerate(MISSING_RATES):
            seed = random_state + 10_000 + fold * 100 + severity_index
            (
                perturbed,
                affected_cells,
                eligible_cells,
            ) = _inject_missing_values(
                validation_features,
                rate=rate,
                seed=seed,
            )
            conditions.append(
                {
                    "perturbation": "missing_values",
                    "severity": rate,
                    "severity_unit": (
                        "fraction_of_observed_validation_cells"
                    ),
                    "seed": seed,
                    "features": perturbed,
                    "affected_cells": affected_cells,
                    "eligible_cells": eligible_cells,
                    "noise_scales": {
                        feature_name: None for feature_name in FEATURES
                    },
                }
            )
        for severity_index, multiplier in enumerate(
            NOISE_STD_MULTIPLIERS
        ):
            seed = random_state + 20_000 + fold * 100 + severity_index
            (
                perturbed,
                affected_cells,
                eligible_cells,
                noise_scales,
            ) = _add_gaussian_noise(
                validation_features,
                training_standard_deviation=training_standard_deviation,
                multiplier=multiplier,
                seed=seed,
            )
            conditions.append(
                {
                    "perturbation": "gaussian_noise",
                    "severity": multiplier,
                    "severity_unit": (
                        "training_feature_standard_deviation_multiplier"
                    ),
                    "seed": seed,
                    "features": perturbed,
                    "affected_cells": affected_cells,
                    "eligible_cells": eligible_cells,
                    "noise_scales": noise_scales,
                }
            )

        for model_name in DIAGNOSTIC_MODEL_NAMES:
            model = _models(random_state)[model_name]
            model.fit(
                training_features,
                target.iloc[train_indices],
            )
            for condition in conditions:
                predicted = model.predict(condition["features"])
                scores = _model_metrics(
                    target.iloc[validation_indices],
                    predicted,
                    labels,
                )
                row: dict[str, Any] = {
                    "perturbation": condition["perturbation"],
                    "severity": condition["severity"],
                    "severity_unit": condition["severity_unit"],
                    "fold": fold,
                    "model": model_name,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "perturbation_seed": condition["seed"],
                    "eligible_cells": condition["eligible_cells"],
                    "affected_cells": condition["affected_cells"],
                    "affected_fraction": round(
                        condition["affected_cells"]
                        / condition["eligible_cells"],
                        6,
                    ),
                    **{
                        f"noise_std_{feature_name}": condition[
                            "noise_scales"
                        ][feature_name]
                        for feature_name in FEATURES
                    },
                    **{
                        score_name: scores[score_name]
                        for score_name in SCORE_NAMES
                    },
                }
                rows.append(row)

    fold_scores = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    perturbation_summaries: dict[str, Any] = {}
    severity_values = {
        "missing_values": MISSING_RATES,
        "gaussian_noise": NOISE_STD_MULTIPLIERS,
    }
    severity_units = {
        "missing_values": "fraction_of_observed_validation_cells",
        "gaussian_noise": (
            "training_feature_standard_deviation_multiplier"
        ),
    }
    for perturbation in ROBUSTNESS_PERTURBATIONS:
        perturbation_summaries[perturbation] = {
            "severity_unit": severity_units[perturbation],
            "conditions": {},
        }
        for severity in severity_values[perturbation]:
            condition_key = f"{severity:g}"
            perturbation_summaries[perturbation]["conditions"][
                condition_key
            ] = {}
            for model_name in DIAGNOSTIC_MODEL_NAMES:
                condition_rows = fold_scores[
                    (fold_scores["perturbation"] == perturbation)
                    & (fold_scores["severity"] == severity)
                    & (fold_scores["model"] == model_name)
                ].set_index("fold")
                baseline_rows = fold_scores[
                    (fold_scores["perturbation"] == perturbation)
                    & (fold_scores["severity"] == 0.0)
                    & (fold_scores["model"] == model_name)
                ].set_index("fold")
                score_metrics: dict[str, Any] = {}
                summary_row: dict[str, Any] = {
                    "perturbation": perturbation,
                    "severity": severity,
                    "severity_unit": severity_units[perturbation],
                    "model": model_name,
                }
                for score_name in SCORE_NAMES:
                    score_summary = _score_summary(
                        condition_rows[score_name]
                    )
                    paired_difference = _score_summary(
                        condition_rows[score_name]
                        - baseline_rows[score_name]
                    )
                    score_metrics[score_name] = {
                        **score_summary,
                        "paired_difference_vs_unperturbed": (
                            paired_difference
                        ),
                    }
                    summary_row[
                        f"{score_name}_mean"
                    ] = score_summary["mean"]
                    summary_row[
                        f"{score_name}_std"
                    ] = score_summary["std"]
                    summary_row[
                        f"{score_name}_difference_vs_unperturbed_mean"
                    ] = paired_difference["mean"]
                perturbation_summaries[perturbation]["conditions"][
                    condition_key
                ][model_name] = score_metrics
                summary_rows.append(summary_row)

    diagnostics = {
        "strategy": "shared_outer_fold_validation_perturbation",
        "outer_folds": len(splits),
        "training_data": "unchanged",
        "validation_labels": "unchanged",
        "models": list(DIAGNOSTIC_MODEL_NAMES),
        "seed_rule": {
            "missing_values": (
                "random_state_plus_10000_plus_fold_times_100"
                "_plus_severity_index"
            ),
            "gaussian_noise": (
                "random_state_plus_20000_plus_fold_times_100"
                "_plus_severity_index"
            ),
            "shared_across_models": True,
        },
        "missing_values": {
            "rates": list(MISSING_RATES),
            "scope": "observed_validation_feature_cells_only",
            "selection": "without_replacement_nearest_integer_cell_count",
            "existing_missing_cells": "preserved_and_not_counted_as_injected",
        },
        "gaussian_noise": {
            "standard_deviation_multipliers": list(
                NOISE_STD_MULTIPLIERS
            ),
            "distribution": "zero_mean_gaussian",
            "scale_source": "outer_training_partition_population_std",
            "scope": "observed_validation_feature_cells_only",
            "existing_missing_cells": "preserved",
        },
        "summary": perturbation_summaries,
        "interpretation": (
            "controlled_sensitivity_evidence_not_deployment_robustness"
        ),
    }
    return fold_scores, pd.DataFrame(summary_rows), diagnostics


def evaluate_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
    cv_folds: int = 5,
    calibration_folds: int = 3,
) -> EvaluationResult:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1")
    missing = sorted(set(FEATURES + (TARGET,)) - set(frame.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )
    if frame[TARGET].isna().any():
        raise ValueError("Dataset contains a missing target value")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    smallest_class = int(frame[TARGET].value_counts().min())
    if cv_folds > smallest_class:
        raise ValueError(
            "cv_folds must not exceed the smallest class count "
            f"({smallest_class})"
        )
    if calibration_folds < 2:
        raise ValueError("calibration_folds must be at least 2")

    labels = tuple(sorted(str(value) for value in frame[TARGET].unique()))
    splits = _stratified_splits(
        frame,
        random_state=random_state,
        cv_folds=cv_folds,
    )
    smallest_outer_training_class = min(
        int(frame.iloc[train_indices][TARGET].value_counts().min())
        for train_indices, _ in splits
    )
    if calibration_folds > smallest_outer_training_class:
        raise ValueError(
            "calibration_folds must not exceed the smallest class count "
            "in an outer training partition "
            f"({smallest_outer_training_class})"
        )
    cross_validation_folds, cross_validation_summary = _cross_validate(
        frame,
        labels=labels,
        random_state=random_state,
        splits=splits,
    )
    (
        feature_ablation_folds,
        feature_ablation_summary,
        feature_ablation_metrics,
    ) = _feature_ablation(
        frame,
        labels=labels,
        random_state=random_state,
        splits=splits,
    )
    leakage_diagnostic_folds, leakage_diagnostics = _leakage_diagnostics(
        frame,
        labels=labels,
        random_state=random_state,
        splits=splits,
        observed_fold_scores=cross_validation_folds,
    )
    (
        probability_calibration_folds,
        probability_calibration_summary,
        probability_calibration_predictions,
        probability_calibration_bins,
        probability_calibration_metrics,
    ) = _probability_calibration(
        frame,
        labels=labels,
        random_state=random_state,
        splits=splits,
        calibration_folds=calibration_folds,
    )
    (
        robustness_folds,
        robustness_summary,
        robustness_diagnostics,
    ) = _robustness_evaluation(
        frame,
        labels=labels,
        random_state=random_state,
        splits=splits,
    )
    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET],
    )
    features_train = frame.iloc[train_indices][list(FEATURES)]
    features_test = frame.iloc[test_indices][list(FEATURES)]
    target_train = frame.iloc[train_indices][TARGET]
    target_test = frame.iloc[test_indices][TARGET]

    predicted: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    model_configurations = _model_configurations(random_state)
    for name, model in _models(random_state).items():
        model.fit(features_train, target_train)
        predicted[name] = model.predict(features_test)
        model_metrics[name] = {
            "configuration": model_configurations[name],
            **_model_metrics(
                target_test,
                predicted[name],
                labels,
            ),
        }

    logistic_confusion = confusion_matrix(
        target_test,
        predicted["logistic_regression"],
        labels=list(labels),
    )
    prediction_rows = pd.DataFrame(
        {
            "source_row": test_indices + 2,
            "actual": target_test.to_numpy(),
            **{
                f"{model_name}_prediction": predicted[model_name]
                for model_name in MODEL_NAMES
            },
        }
    )
    for model_name in MODEL_NAMES:
        prediction_rows[f"{model_name}_correct"] = (
            prediction_rows["actual"]
            == prediction_rows[f"{model_name}_prediction"]
        )
    prediction_rows = prediction_rows.sort_values("source_row").reset_index(
        drop=True
    )

    holdout_gains = {
        f"{left_model}_minus_{right_model}": _metric_differences(
            model_metrics[left_model],
            model_metrics[right_model],
        )
        for left_model, right_model in COMPARISON_PAIRS
    }
    model_comparison = _model_comparison_table(
        model_metrics,
        cross_validation_summary,
    )
    metrics = {
        "report_version": 6,
        "dataset": {
            "rows": len(frame),
            "target": TARGET,
            "classes": list(labels),
            "features": list(FEATURES),
            "missing_feature_cells": int(
                frame[list(FEATURES)].isna().sum().sum()
            ),
        },
        "split": {
            "strategy": "stratified_holdout",
            "random_state": random_state,
            "test_size": test_size,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
        },
        "models": model_metrics,
        "comparison": {
            "positive_difference_favors": "left_model",
            "holdout_gain": holdout_gains,
        },
        "cross_validation": cross_validation_summary,
        "feature_ablation": feature_ablation_metrics,
        "leakage_diagnostics": leakage_diagnostics,
        "probability_calibration": probability_calibration_metrics,
        "robustness": robustness_diagnostics,
    }
    return EvaluationResult(
        metrics=metrics,
        predictions=prediction_rows,
        cross_validation_folds=cross_validation_folds,
        model_comparison=model_comparison,
        feature_ablation_folds=feature_ablation_folds,
        feature_ablation_summary=feature_ablation_summary,
        leakage_diagnostic_folds=leakage_diagnostic_folds,
        leakage_diagnostics=leakage_diagnostics,
        probability_calibration_folds=probability_calibration_folds,
        probability_calibration_summary=probability_calibration_summary,
        probability_calibration_predictions=probability_calibration_predictions,
        probability_calibration_bins=probability_calibration_bins,
        robustness_folds=robustness_folds,
        robustness_summary=robustness_summary,
        robustness_diagnostics=robustness_diagnostics,
        confusion=logistic_confusion,
        labels=labels,
    )
