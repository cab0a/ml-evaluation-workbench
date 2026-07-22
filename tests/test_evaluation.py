from __future__ import annotations

import numpy as np
import pytest

from ml_evaluation_workbench import evaluate_dataset


def test_evaluation_is_deterministic(penguins_frame) -> None:
    first = evaluate_dataset(penguins_frame)
    second = evaluate_dataset(penguins_frame)

    assert first.metrics == second.metrics
    assert first.predictions.equals(second.predictions)
    assert first.cross_validation_folds.equals(
        second.cross_validation_folds
    )
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


def test_cross_validation_records_shared_fold_evidence(penguins_frame) -> None:
    result = evaluate_dataset(penguins_frame)
    fold_scores = result.cross_validation_folds
    summary = result.metrics["cross_validation"]

    assert summary["strategy"] == "stratified_k_fold"
    assert summary["folds"] == 5
    assert summary["standard_deviation"] == "population_across_folds"
    assert len(fold_scores) == 10
    assert set(fold_scores["model"]) == {"dummy", "logistic_regression"}
    assert set(fold_scores["fold"]) == {1, 2, 3, 4, 5}
    for model_name in ("dummy", "logistic_regression"):
        model_rows = fold_scores[fold_scores["model"] == model_name]
        assert int(model_rows["validation_rows"].sum()) == 344


def test_cross_validation_summary_matches_fold_scores(penguins_frame) -> None:
    result = evaluate_dataset(penguins_frame)
    fold_scores = result.cross_validation_folds
    summary = result.metrics["cross_validation"]

    for model_name in ("dummy", "logistic_regression"):
        model_rows = fold_scores[fold_scores["model"] == model_name]
        for score_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            expected_mean = round(float(model_rows[score_name].mean()), 6)
            expected_std = round(
                float(model_rows[score_name].std(ddof=0)),
                6,
            )
            observed = summary["models"][model_name][score_name]
            assert observed["mean"] == expected_mean
            assert observed["std"] == expected_std

    assert (
        summary["models"]["logistic_regression"]["macro_f1"]["mean"]
        > summary["models"]["dummy"]["macro_f1"]["mean"]
    )


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_invalid_test_size_is_rejected(penguins_frame, test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        evaluate_dataset(penguins_frame, test_size=test_size)


@pytest.mark.parametrize("cv_folds", [0, 1, 69])
def test_invalid_cross_validation_fold_count_is_rejected(
    penguins_frame,
    cv_folds: int,
) -> None:
    with pytest.raises(ValueError, match="cv_folds"):
        evaluate_dataset(penguins_frame, cv_folds=cv_folds)
