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
    assert first.model_comparison.equals(second.model_comparison)
    assert first.feature_ablation_folds.equals(
        second.feature_ablation_folds
    )
    assert first.feature_ablation_summary.equals(
        second.feature_ablation_summary
    )
    assert first.leakage_diagnostic_folds.equals(
        second.leakage_diagnostic_folds
    )
    assert first.leakage_diagnostics == second.leakage_diagnostics
    assert np.array_equal(first.confusion, second.confusion)


def test_logistic_regression_exceeds_dummy_baseline(penguins_frame) -> None:
    result = evaluate_dataset(penguins_frame)
    dummy = result.metrics["models"]["dummy"]
    logistic = result.metrics["models"]["logistic_regression"]

    assert logistic["accuracy"] > dummy["accuracy"]
    assert logistic["balanced_accuracy"] > dummy["balanced_accuracy"]
    assert logistic["macro_f1"] > dummy["macro_f1"]
    assert 0.8 < logistic["macro_f1"] < 1.0


def test_knn_is_a_fixed_nonlinear_comparator(penguins_frame) -> None:
    result = evaluate_dataset(penguins_frame)
    dummy = result.metrics["models"]["dummy"]
    knn = result.metrics["models"]["knn"]

    assert knn["configuration"] == {
        "classifier": "KNeighborsClassifier",
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "metric": "minkowski",
        "p": 2,
    }
    assert knn["accuracy"] > dummy["accuracy"]
    assert knn["balanced_accuracy"] > dummy["balanced_accuracy"]
    assert knn["macro_f1"] > dummy["macro_f1"]


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
    assert "knn_prediction" in result.predictions
    assert "knn_correct" in result.predictions


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
    assert len(fold_scores) == 15
    assert set(fold_scores["model"]) == {
        "dummy",
        "logistic_regression",
        "knn",
    }
    assert set(fold_scores["fold"]) == {1, 2, 3, 4, 5}
    for model_name in ("dummy", "logistic_regression", "knn"):
        model_rows = fold_scores[fold_scores["model"] == model_name]
        assert int(model_rows["validation_rows"].sum()) == 344


def test_cross_validation_summary_matches_fold_scores(penguins_frame) -> None:
    result = evaluate_dataset(penguins_frame)
    fold_scores = result.cross_validation_folds
    summary = result.metrics["cross_validation"]

    for model_name in ("dummy", "logistic_regression", "knn"):
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
    assert (
        summary["models"]["knn"]["macro_f1"]["mean"]
        > summary["models"]["dummy"]["macro_f1"]["mean"]
    )


def test_controlled_model_comparison_uses_paired_differences(
    penguins_frame,
) -> None:
    result = evaluate_dataset(penguins_frame)
    holdout = result.metrics["comparison"]["holdout_gain"]
    cross_validation = result.metrics["cross_validation"]["paired_difference"]

    assert set(holdout) == {
        "logistic_regression_minus_dummy",
        "knn_minus_dummy",
        "knn_minus_logistic_regression",
    }
    assert set(cross_validation) == set(holdout)
    expected = round(
        result.metrics["models"]["knn"]["macro_f1"]
        - result.metrics["models"]["logistic_regression"]["macro_f1"],
        6,
    )
    assert holdout["knn_minus_logistic_regression"]["macro_f1"] == expected
    assert list(result.model_comparison["model"]) == [
        "dummy",
        "logistic_regression",
        "knn",
    ]


def test_feature_ablation_uses_shared_folds_and_reference(
    penguins_frame,
) -> None:
    result = evaluate_dataset(penguins_frame)
    folds = result.feature_ablation_folds
    summary = result.metrics["feature_ablation"]

    assert len(folds) == 30
    assert set(folds["feature_set"]) == {
        "bill_length_only",
        "bill_depth_only",
        "both_bill_measurements",
    }
    assert set(folds["model"]) == {"logistic_regression", "knn"}
    assert summary["selection_policy"] == "diagnostic_only_no_model_selection"
    for feature_set in summary["feature_sets"]:
        for model_name in ("logistic_regression", "knn"):
            model_rows = folds[
                (folds["feature_set"] == feature_set)
                & (folds["model"] == model_name)
            ]
            assert int(model_rows["validation_rows"].sum()) == 344

    for model_name in ("logistic_regression", "knn"):
        reference = summary["feature_sets"]["both_bill_measurements"][
            "models"
        ][model_name]
        observed = result.metrics["cross_validation"]["models"][model_name]
        for score_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert reference[score_name]["mean"] == observed[score_name]["mean"]
            assert (
                reference[score_name]["paired_difference_vs_both"]["mean"]
                == 0.0
            )


def test_leakage_diagnostics_check_partitions_and_negative_control(
    penguins_frame,
) -> None:
    result = evaluate_dataset(penguins_frame)
    diagnostics = result.leakage_diagnostics
    fold_rows = result.leakage_diagnostic_folds

    assert diagnostics["interpretation"] == (
        "negative_control_not_proof_of_no_leakage"
    )
    integrity = diagnostics["split_integrity"]
    assert integrity["passed"] is True
    assert integrity["maximum_train_validation_overlap_rows"] == 0
    assert integrity["validation_coverage_minimum"] == 1
    assert integrity["validation_coverage_maximum"] == 1
    assert len(fold_rows) == 10
    assert int(fold_rows["train_validation_overlap_rows"].max()) == 0

    negative_control = diagnostics["shuffled_training_labels"]["models"]
    observed = result.metrics["cross_validation"]["models"]
    for model_name in ("logistic_regression", "knn"):
        shuffled_macro_f1 = negative_control[model_name]["macro_f1"][
            "shuffled"
        ]["mean"]
        observed_macro_f1 = observed[model_name]["macro_f1"]["mean"]
        difference = negative_control[model_name]["macro_f1"][
            "observed_minus_shuffled"
        ]["mean"]
        assert shuffled_macro_f1 < 0.5
        assert observed_macro_f1 - shuffled_macro_f1 == pytest.approx(
            difference,
            abs=1e-6,
        )
        assert difference > 0.4


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
