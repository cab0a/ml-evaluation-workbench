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
    assert first.probability_calibration_folds.equals(
        second.probability_calibration_folds
    )
    assert first.probability_calibration_summary.equals(
        second.probability_calibration_summary
    )
    assert first.probability_calibration_predictions.equals(
        second.probability_calibration_predictions
    )
    assert first.probability_calibration_bins.equals(
        second.probability_calibration_bins
    )
    assert first.robustness_folds.equals(second.robustness_folds)
    assert first.robustness_summary.equals(second.robustness_summary)
    assert first.robustness_diagnostics == second.robustness_diagnostics
    assert first.class_imbalance_folds.equals(
        second.class_imbalance_folds
    )
    assert first.class_imbalance_summary.equals(
        second.class_imbalance_summary
    )
    assert (
        first.class_imbalance_diagnostics
        == second.class_imbalance_diagnostics
    )
    assert np.array_equal(first.confusion, second.confusion)


def test_logistic_regression_exceeds_dummy_baseline(evaluation_result) -> None:
    result = evaluation_result
    dummy = result.metrics["models"]["dummy"]
    logistic = result.metrics["models"]["logistic_regression"]

    assert logistic["accuracy"] > dummy["accuracy"]
    assert logistic["balanced_accuracy"] > dummy["balanced_accuracy"]
    assert logistic["macro_f1"] > dummy["macro_f1"]
    assert 0.8 < logistic["macro_f1"] < 1.0


def test_knn_is_a_fixed_nonlinear_comparator(evaluation_result) -> None:
    result = evaluation_result
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
    evaluation_result,
) -> None:
    result = evaluation_result

    assert result.metrics["split"]["train_rows"] == 258
    assert result.metrics["split"]["test_rows"] == 86
    assert result.labels == ("Adelie", "Chinstrap", "Gentoo")
    assert result.confusion.shape == (3, 3)
    assert int(result.confusion.sum()) == 86
    assert len(result.predictions) == 86
    assert result.predictions["source_row"].is_monotonic_increasing
    assert "knn_prediction" in result.predictions
    assert "knn_correct" in result.predictions


def test_pipeline_handles_missing_bill_measurements(
    penguins_frame,
    evaluation_result,
) -> None:
    assert int(
        penguins_frame[["bill_length_mm", "bill_depth_mm"]].isna().sum().sum()
    ) == 4

    result = evaluation_result

    assert result.metrics["dataset"]["missing_feature_cells"] == 4


def test_cross_validation_records_shared_fold_evidence(
    evaluation_result,
) -> None:
    result = evaluation_result
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


def test_cross_validation_summary_matches_fold_scores(
    evaluation_result,
) -> None:
    result = evaluation_result
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
    evaluation_result,
) -> None:
    result = evaluation_result
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
    evaluation_result,
) -> None:
    result = evaluation_result
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
    evaluation_result,
) -> None:
    result = evaluation_result
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


def test_probability_calibration_uses_cross_fitted_probabilities(
    evaluation_result,
) -> None:
    result = evaluation_result
    folds = result.probability_calibration_folds
    predictions = result.probability_calibration_predictions
    summary = result.metrics["probability_calibration"]

    assert summary["strategy"] == "shared_outer_stratified_k_fold"
    assert summary["outer_folds"] == 5
    assert summary["inner_calibration"] == {
        "method": "sigmoid",
        "folds": 3,
        "scope": "outer_training_partition_only",
        "shuffle": True,
        "seed_rule": "random_state_plus_outer_fold",
        "ensemble": True,
    }
    assert len(folds) == 20
    assert len(predictions) == 344 * 4
    assert set(folds["model"]) == {"logistic_regression", "knn"}
    assert set(folds["calibration"]) == {"uncalibrated", "sigmoid"}
    for model_name in ("logistic_regression", "knn"):
        primary_folds = (
            result.cross_validation_folds[
                result.cross_validation_folds["model"] == model_name
            ]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        uncalibrated_folds = (
            folds[
                (folds["model"] == model_name)
                & (folds["calibration"] == "uncalibrated")
            ]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        assert np.array_equal(
            uncalibrated_folds["validation_rows"],
            primary_folds["validation_rows"],
        )
        assert np.allclose(
            uncalibrated_folds["accuracy"],
            primary_folds["accuracy"],
        )
        for calibration_name in ("uncalibrated", "sigmoid"):
            method_rows = predictions[
                (predictions["model"] == model_name)
                & (predictions["calibration"] == calibration_name)
            ]
            assert len(method_rows) == 344
            assert method_rows["source_row"].nunique() == 344
            probability_columns = [
                "probability_adelie",
                "probability_chinstrap",
                "probability_gentoo",
            ]
            assert np.allclose(
                method_rows[probability_columns].sum(axis=1),
                1.0,
                atol=2e-6,
            )


def test_probability_calibration_metrics_and_bins_are_auditable(
    evaluation_result,
) -> None:
    result = evaluation_result
    folds = result.probability_calibration_folds
    bins = result.probability_calibration_bins
    summary = result.metrics["probability_calibration"]

    assert len(bins) == 40
    assert set(bins["bin"]) == set(range(1, 11))
    assert int(bins["sample_count"].sum()) == 344 * 4
    for model_name in ("logistic_regression", "knn"):
        for calibration_name in ("uncalibrated", "sigmoid"):
            model_rows = folds[
                (folds["model"] == model_name)
                & (folds["calibration"] == calibration_name)
            ]
            recorded = summary["models_summary"][model_name][
                calibration_name
            ]
            for score_name in (
                "accuracy",
                "log_loss",
                "multiclass_brier",
                "top_label_ece",
            ):
                assert recorded[score_name]["mean"] == round(
                    float(model_rows[score_name].mean()),
                    6,
                )
                assert np.isfinite(model_rows[score_name]).all()
        assert (
            "sigmoid_minus_uncalibrated"
            in summary["paired_difference"][model_name]
        )


def test_robustness_uses_shared_folds_and_validation_perturbations(
    evaluation_result,
) -> None:
    result = evaluation_result
    folds = result.robustness_folds
    diagnostics = result.robustness_diagnostics

    assert diagnostics["strategy"] == (
        "shared_outer_fold_validation_perturbation"
    )
    assert diagnostics["training_data"] == "unchanged"
    assert diagnostics["validation_labels"] == "unchanged"
    assert diagnostics["seed_rule"]["shared_across_models"] is True
    assert len(folds) == 80
    assert set(folds["perturbation"]) == {
        "missing_values",
        "gaussian_noise",
    }
    assert set(
        folds[folds["perturbation"] == "missing_values"]["severity"]
    ) == {0.0, 0.1, 0.25, 0.5}
    assert set(
        folds[folds["perturbation"] == "gaussian_noise"]["severity"]
    ) == {0.0, 0.25, 0.5, 1.0}
    seed_counts = folds.groupby(
        ["perturbation", "severity", "fold"]
    )["perturbation_seed"].nunique()
    assert int(seed_counts.max()) == 1

    for model_name in ("logistic_regression", "knn"):
        primary = (
            result.cross_validation_folds[
                result.cross_validation_folds["model"] == model_name
            ]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        for perturbation in ("missing_values", "gaussian_noise"):
            baseline = (
                folds[
                    (folds["model"] == model_name)
                    & (folds["perturbation"] == perturbation)
                    & (folds["severity"] == 0.0)
                ]
                .sort_values("fold")
                .reset_index(drop=True)
            )
            for score_name in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ):
                assert np.allclose(
                    baseline[score_name],
                    primary[score_name],
                )


def test_robustness_records_cell_accounting_and_summaries(
    evaluation_result,
) -> None:
    result = evaluation_result
    folds = result.robustness_folds
    summary = result.robustness_summary

    assert len(summary) == 16
    one_model = folds[folds["model"] == "logistic_regression"]
    for perturbation, severity in (
        ("missing_values", 0.5),
        ("gaussian_noise", 1.0),
    ):
        condition_rows = one_model[
            (one_model["perturbation"] == perturbation)
            & (one_model["severity"] == severity)
        ]
        assert int(condition_rows["eligible_cells"].sum()) == 684
        if perturbation == "missing_values":
            assert condition_rows["affected_fraction"].between(
                0.49,
                0.51,
            ).all()
        else:
            assert np.array_equal(
                condition_rows["affected_cells"],
                condition_rows["eligible_cells"],
            )
            assert (
                condition_rows["noise_std_bill_length_mm"] > 0
            ).all()
            assert (
                condition_rows["noise_std_bill_depth_mm"] > 0
            ).all()

    for row in summary.itertuples(index=False):
        fold_rows = folds[
            (folds["perturbation"] == row.perturbation)
            & (folds["severity"] == row.severity)
            & (folds["model"] == row.model)
        ]
        assert row.macro_f1_mean == round(
            float(fold_rows["macro_f1"].mean()),
            6,
        )


def test_class_imbalance_uses_shared_training_samples_and_validation_folds(
    evaluation_result,
) -> None:
    result = evaluation_result
    folds = result.class_imbalance_folds
    diagnostics = result.class_imbalance_diagnostics

    assert diagnostics["strategy"] == (
        "shared_outer_fold_training_class_downsampling"
    )
    assert diagnostics["target_class"] == "Chinstrap"
    assert diagnostics["global_class_counts"] == {
        "Adelie": 152,
        "Chinstrap": 68,
        "Gentoo": 124,
    }
    assert diagnostics["validation_data"] == "unchanged"
    assert diagnostics["validation_labels"] == "unchanged"
    assert diagnostics["seed_rule"]["shared_across_models"] is True
    assert len(folds) == 40
    assert set(folds["retention_fraction"]) == {1.0, 0.75, 0.5, 0.25}

    shared_columns = (
        "resampling_seed",
        "retained_target_rows",
        "retained_target_source_rows_sha256",
    )
    for column in shared_columns:
        unique_counts = folds.groupby(
            ["retention_fraction", "fold"]
        )[column].nunique()
        assert int(unique_counts.max()) == 1

    for model_name in ("logistic_regression", "knn"):
        primary = (
            result.cross_validation_folds[
                result.cross_validation_folds["model"] == model_name
            ]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        full_retention = (
            folds[
                (folds["model"] == model_name)
                & (folds["retention_fraction"] == 1.0)
            ]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        for score_name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ):
            assert np.allclose(
                full_retention[score_name],
                primary[score_name],
            )


def test_class_imbalance_records_counts_recall_and_summaries(
    evaluation_result,
) -> None:
    result = evaluation_result
    folds = result.class_imbalance_folds
    summary = result.class_imbalance_summary

    assert len(summary) == 8
    one_model = folds[folds["model"] == "logistic_regression"]
    full_retention = one_model[
        one_model["retention_fraction"] == 1.0
    ].set_index("fold")
    quarter_retention = one_model[
        one_model["retention_fraction"] == 0.25
    ].set_index("fold")
    assert (
        quarter_retention["retained_target_rows"]
        < full_retention["retained_target_rows"]
    ).all()
    assert (
        quarter_retention["target_share_after"]
        < full_retention["target_share_after"]
    ).all()
    assert (
        quarter_retention["train_rows_adelie"]
        == full_retention["train_rows_adelie"]
    ).all()
    assert (
        quarter_retention["train_rows_gentoo"]
        == full_retention["train_rows_gentoo"]
    ).all()

    for row in summary.itertuples(index=False):
        fold_rows = folds[
            (folds["retention_fraction"] == row.retention_fraction)
            & (folds["model"] == row.model)
        ]
        assert row.macro_f1_mean == round(
            float(fold_rows["macro_f1"].mean()),
            6,
        )
        assert row.target_class_recall_mean == round(
            float(fold_rows["recall_chinstrap"].mean()),
            6,
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


@pytest.mark.parametrize("calibration_folds", [0, 1, 101])
def test_invalid_calibration_fold_count_is_rejected(
    penguins_frame,
    calibration_folds: int,
) -> None:
    with pytest.raises(ValueError, match="calibration_folds"):
        evaluate_dataset(
            penguins_frame,
            calibration_folds=calibration_folds,
        )
