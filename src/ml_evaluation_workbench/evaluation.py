"""Deterministic baseline training and evaluation."""

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = (
    "bill_length_mm",
    "bill_depth_mm",
)
FEATURES = NUMERIC_FEATURES
TARGET = "species"


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
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


def evaluate_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
) -> EvaluationResult:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1")
    missing = sorted(set(FEATURES + (TARGET,)) - set(frame.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if frame[TARGET].isna().any():
        raise ValueError("Dataset contains a missing target value")

    labels = tuple(sorted(str(value) for value in frame[TARGET].unique()))
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

    models = {
        "dummy": _pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
    }
    predicted: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
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
        "report_version": 1,
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
    }
    return EvaluationResult(
        metrics=metrics,
        predictions=prediction_rows,
        confusion=logistic_confusion,
        labels=labels,
    )
