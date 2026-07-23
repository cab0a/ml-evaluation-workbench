"""Deterministic holdout and cross-validation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
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


def evaluate_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
    cv_folds: int = 5,
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

    labels = tuple(sorted(str(value) for value in frame[TARGET].unique()))
    splits = _stratified_splits(
        frame,
        random_state=random_state,
        cv_folds=cv_folds,
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
        "report_version": 4,
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
        confusion=logistic_confusion,
        labels=labels,
    )
