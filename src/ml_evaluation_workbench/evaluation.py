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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = (
    "bill_length_mm",
    "bill_depth_mm",
)
FEATURES = NUMERIC_FEATURES
TARGET = "species"
SCORE_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")
MODEL_NAMES = ("dummy", "logistic_regression")


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    cross_validation_folds: pd.DataFrame
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


def _cross_validate(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    random_state: int,
    cv_folds: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    features = frame[list(FEATURES)]
    target = frame[TARGET]
    rows: list[dict[str, Any]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, target),
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

    fold_scores = pd.DataFrame(rows).sort_values(
        ["fold", "model"],
        kind="stable",
    )
    fold_scores = fold_scores.reset_index(drop=True)
    model_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in MODEL_NAMES:
        model_rows = fold_scores[fold_scores["model"] == model_name]
        model_summaries[model_name] = {
            score_name: _score_summary(model_rows[score_name])
            for score_name in SCORE_NAMES
        }

    paired_gains: dict[str, dict[str, float]] = {}
    for score_name in SCORE_NAMES:
        by_model = fold_scores.pivot(
            index="fold",
            columns="model",
            values=score_name,
        )
        paired_gains[score_name] = _score_summary(
            by_model["logistic_regression"] - by_model["dummy"]
        )

    summary = {
        "strategy": "stratified_k_fold",
        "folds": cv_folds,
        "shuffle": True,
        "random_state": random_state,
        "standard_deviation": "population_across_folds",
        "models": model_summaries,
        "paired_gain": paired_gains,
    }
    return fold_scores, summary


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
    cross_validation_folds, cross_validation_summary = _cross_validate(
        frame,
        labels=labels,
        random_state=random_state,
        cv_folds=cv_folds,
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
    for name, model in _models(random_state).items():
        model.fit(features_train, target_train)
        predicted[name] = model.predict(features_test)
        model_metrics[name] = _model_metrics(
            target_test,
            predicted[name],
            labels,
        )

    logistic_confusion = confusion_matrix(
        target_test,
        predicted["logistic_regression"],
        labels=list(labels),
    )
    prediction_rows = pd.DataFrame(
        {
            "source_row": test_indices + 2,
            "actual": target_test.to_numpy(),
            "dummy_prediction": predicted["dummy"],
            "logistic_regression_prediction": predicted[
                "logistic_regression"
            ],
        }
    )
    prediction_rows["dummy_correct"] = (
        prediction_rows["actual"] == prediction_rows["dummy_prediction"]
    )
    prediction_rows["logistic_regression_correct"] = (
        prediction_rows["actual"]
        == prediction_rows["logistic_regression_prediction"]
    )
    prediction_rows = prediction_rows.sort_values("source_row").reset_index(
        drop=True
    )

    dummy_metrics = model_metrics["dummy"]
    logistic_metrics = model_metrics["logistic_regression"]
    metrics = {
        "report_version": 2,
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
            "accuracy_gain": round(
                logistic_metrics["accuracy"] - dummy_metrics["accuracy"], 6
            ),
            "balanced_accuracy_gain": round(
                logistic_metrics["balanced_accuracy"]
                - dummy_metrics["balanced_accuracy"],
                6,
            ),
            "macro_f1_gain": round(
                logistic_metrics["macro_f1"] - dummy_metrics["macro_f1"],
                6,
            ),
        },
        "cross_validation": cross_validation_summary,
    }
    return EvaluationResult(
        metrics=metrics,
        predictions=prediction_rows,
        cross_validation_folds=cross_validation_folds,
        confusion=logistic_confusion,
        labels=labels,
    )
