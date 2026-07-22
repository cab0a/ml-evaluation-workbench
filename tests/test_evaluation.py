from __future__ import annotations

import numpy as np
import pytest

from ml_evaluation_workbench import evaluate_dataset


def test_evaluation_is_deterministic(penguins_frame) -> None:
    first = evaluate_dataset(penguins_frame)
    second = evaluate_dataset(penguins_frame)

    assert first.metrics == second.metrics
    assert first.predictions.equals(second.predictions)
    assert np.array_equal(first.confusion, second.confusion)


def test_logistic_regression_exceeds_dummy_baseline(penguins_frame) -> None:
    result = evaluate_dataset(penguins_frame)
    dummy = result.metrics["models"]["dummy"]
    logistic = result.metrics["models"]["logistic_regression"]

    assert logistic["accuracy"] > dummy["accuracy"]
    assert logistic["balanced_accuracy"] > dummy["balanced_accuracy"]
    assert logistic["macro_f1"] > dummy["macro_f1"]
    assert 0.8 < logistic["macro_f1"] < 1.0


def test_stratified_holdout_and_confusion_matrix_are_consistent(
    penguins_frame,
) -> None:
    result = evaluate_dataset(penguins_frame)

    assert result.metrics["split"]["train_rows"] == 258
    assert result.metrics["split"]["test_rows"] == 86
    assert result.labels == ("Adelie", "Chinstrap", "Gentoo")
    assert result.confusion.shape == (3, 3)
    assert int(result.confusion.sum()) == 86
    assert len(result.predictions) == 86
    assert result.predictions["source_row"].is_monotonic_increasing


def test_pipeline_handles_missing_bill_measurements(penguins_frame) -> None:
    assert int(
        penguins_frame[["bill_length_mm", "bill_depth_mm"]].isna().sum().sum()
    ) == 4

    result = evaluate_dataset(penguins_frame)

    assert result.metrics["dataset"]["missing_feature_cells"] == 4


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_invalid_test_size_is_rejected(penguins_frame, test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        evaluate_dataset(penguins_frame, test_size=test_size)
