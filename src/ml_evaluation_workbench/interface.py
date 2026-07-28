"""Machine-readable contracts for the documented stable interfaces."""

from __future__ import annotations

from typing import Any, Sequence


GENERATED_ARTIFACTS = (
    {
        "path": "class_imbalance.png",
        "media_type": "image/png",
        "role": "class_imbalance_figure",
    },
    {
        "path": "class_imbalance_diagnostics.json",
        "media_type": "application/json",
        "role": "class_imbalance_protocol",
    },
    {
        "path": "class_imbalance_folds.csv",
        "media_type": "text/csv",
        "role": "class_imbalance_fold_evidence",
    },
    {
        "path": "class_imbalance_summary.csv",
        "media_type": "text/csv",
        "role": "class_imbalance_summary",
    },
    {
        "path": "confusion_matrix.png",
        "media_type": "image/png",
        "role": "holdout_confusion_matrix",
    },
    {
        "path": "cross_experiment_summary.csv",
        "media_type": "text/csv",
        "role": "representative_contrast_summary",
    },
    {
        "path": "cross_experiment_summary.png",
        "media_type": "image/png",
        "role": "macro_f1_contrast_figure",
    },
    {
        "path": "cross_validation_folds.csv",
        "media_type": "text/csv",
        "role": "cross_validation_fold_evidence",
    },
    {
        "path": "cross_validation_scores.png",
        "media_type": "image/png",
        "role": "cross_validation_figure",
    },
    {
        "path": "feature_ablation_folds.csv",
        "media_type": "text/csv",
        "role": "feature_ablation_fold_evidence",
    },
    {
        "path": "feature_ablation_scores.png",
        "media_type": "image/png",
        "role": "feature_ablation_figure",
    },
    {
        "path": "feature_ablation_summary.csv",
        "media_type": "text/csv",
        "role": "feature_ablation_summary",
    },
    {
        "path": "interface_contract.json",
        "media_type": "application/json",
        "role": "documented_interface_contract",
    },
    {
        "path": "leakage_diagnostic_folds.csv",
        "media_type": "text/csv",
        "role": "negative_control_fold_evidence",
    },
    {
        "path": "leakage_diagnostics.json",
        "media_type": "application/json",
        "role": "leakage_diagnostic_protocol",
    },
    {
        "path": "metrics.json",
        "media_type": "application/json",
        "role": "complete_evaluation_report",
    },
    {
        "path": "model_comparison.csv",
        "media_type": "text/csv",
        "role": "compact_model_comparison",
    },
    {
        "path": "predictions.csv",
        "media_type": "text/csv",
        "role": "holdout_row_predictions",
    },
    {
        "path": "probability_calibration.png",
        "media_type": "image/png",
        "role": "calibration_figure",
    },
    {
        "path": "probability_calibration_bins.csv",
        "media_type": "text/csv",
        "role": "calibration_bin_evidence",
    },
    {
        "path": "probability_calibration_folds.csv",
        "media_type": "text/csv",
        "role": "calibration_fold_evidence",
    },
    {
        "path": "probability_calibration_predictions.csv",
        "media_type": "text/csv",
        "role": "cross_fitted_probabilities",
    },
    {
        "path": "probability_calibration_summary.csv",
        "media_type": "text/csv",
        "role": "calibration_summary",
    },
    {
        "path": "robustness.png",
        "media_type": "image/png",
        "role": "validation_perturbation_figure",
    },
    {
        "path": "robustness_diagnostics.json",
        "media_type": "application/json",
        "role": "validation_perturbation_protocol",
    },
    {
        "path": "robustness_folds.csv",
        "media_type": "text/csv",
        "role": "validation_perturbation_fold_evidence",
    },
    {
        "path": "robustness_summary.csv",
        "media_type": "text/csv",
        "role": "validation_perturbation_summary",
    },
)
GENERATED_ARTIFACT_NAMES = tuple(
    artifact["path"] for artifact in GENERATED_ARTIFACTS
)

CSV_ARTIFACT_SCHEMAS = {
    "class_imbalance_folds.csv": (
        "target_class",
        "retention_fraction",
        "fold",
        "model",
        "resampling_seed",
        "original_train_rows",
        "resampled_train_rows",
        "validation_rows",
        "original_target_rows",
        "retained_target_rows",
        "dropped_target_rows",
        "target_share_before",
        "target_share_after",
        "retained_target_source_rows_sha256",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "train_rows_adelie",
        "train_rows_chinstrap",
        "train_rows_gentoo",
        "recall_adelie",
        "recall_chinstrap",
        "recall_gentoo",
    ),
    "class_imbalance_summary.csv": (
        "target_class",
        "retention_fraction",
        "model",
        "retained_target_rows_mean",
        "target_share_after_mean",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_difference_vs_full_retention_mean",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_difference_vs_full_retention_mean",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_difference_vs_full_retention_mean",
        "target_class_recall_mean",
        "target_class_recall_std",
        "target_class_recall_difference_vs_full_retention_mean",
    ),
    "cross_experiment_summary.csv": (
        "experiment",
        "comparison",
        "display_label",
        "model",
        "metric",
        "preferred_direction",
        "reference",
        "condition",
        "reference_mean",
        "reference_std",
        "condition_mean",
        "condition_std",
        "condition_minus_reference_mean",
        "condition_minus_reference_std",
        "preferred_effect_mean",
        "folds",
        "source_artifact",
        "interpretation",
    ),
    "cross_validation_folds.csv": (
        "fold",
        "model",
        "train_rows",
        "validation_rows",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_adelie",
        "recall_chinstrap",
        "recall_gentoo",
    ),
    "feature_ablation_folds.csv": (
        "feature_set",
        "features",
        "fold",
        "model",
        "train_rows",
        "validation_rows",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_adelie",
        "recall_chinstrap",
        "recall_gentoo",
    ),
    "feature_ablation_summary.csv": (
        "feature_set",
        "features",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_difference_vs_both_mean",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_difference_vs_both_mean",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_difference_vs_both_mean",
    ),
    "leakage_diagnostic_folds.csv": (
        "fold",
        "model",
        "train_rows",
        "validation_rows",
        "train_validation_overlap_rows",
        "shuffled_accuracy",
        "observed_accuracy",
        "observed_minus_shuffled_accuracy",
        "shuffled_balanced_accuracy",
        "observed_balanced_accuracy",
        "observed_minus_shuffled_balanced_accuracy",
        "shuffled_macro_f1",
        "observed_macro_f1",
        "observed_minus_shuffled_macro_f1",
        "shuffled_recall_adelie",
        "shuffled_recall_chinstrap",
        "shuffled_recall_gentoo",
    ),
    "model_comparison.csv": (
        "model",
        "role",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_macro_f1",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ),
    "predictions.csv": (
        "source_row",
        "actual",
        "dummy_prediction",
        "logistic_regression_prediction",
        "knn_prediction",
        "dummy_correct",
        "logistic_regression_correct",
        "knn_correct",
    ),
    "probability_calibration_bins.csv": (
        "model",
        "calibration",
        "bin",
        "lower_bound",
        "upper_bound",
        "sample_count",
        "mean_confidence",
        "empirical_accuracy",
        "absolute_gap",
    ),
    "probability_calibration_folds.csv": (
        "fold",
        "model",
        "calibration",
        "train_rows",
        "validation_rows",
        "accuracy",
        "log_loss",
        "multiclass_brier",
        "top_label_ece",
    ),
    "probability_calibration_predictions.csv": (
        "source_row",
        "fold",
        "model",
        "calibration",
        "actual",
        "predicted",
        "confidence",
        "correct",
        "probability_adelie",
        "probability_chinstrap",
        "probability_gentoo",
    ),
    "probability_calibration_summary.csv": (
        "model",
        "calibration",
        "accuracy_mean",
        "accuracy_std",
        "log_loss_mean",
        "log_loss_std",
        "multiclass_brier_mean",
        "multiclass_brier_std",
        "top_label_ece_mean",
        "top_label_ece_std",
    ),
    "robustness_folds.csv": (
        "perturbation",
        "severity",
        "severity_unit",
        "fold",
        "model",
        "train_rows",
        "validation_rows",
        "perturbation_seed",
        "eligible_cells",
        "affected_cells",
        "affected_fraction",
        "noise_std_bill_length_mm",
        "noise_std_bill_depth_mm",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ),
    "robustness_summary.csv": (
        "perturbation",
        "severity",
        "severity_unit",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_difference_vs_unperturbed_mean",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_difference_vs_unperturbed_mean",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_difference_vs_unperturbed_mean",
    ),
}

JSON_ARTIFACT_TOP_LEVEL_KEYS = {
    "class_imbalance_diagnostics.json": (
        "strategy",
        "outer_folds",
        "target_class",
        "target_class_selection",
        "global_class_counts",
        "retention_fractions",
        "sampling",
        "seed_rule",
        "training_data",
        "validation_data",
        "validation_labels",
        "summary",
        "interpretation",
    ),
    "interface_contract.json": (
        "contract_version",
        "project_version",
        "stability",
        "compatibility",
        "cli",
        "python_api",
        "reports",
        "reproducibility",
    ),
    "leakage_diagnostics.json": (
        "interpretation",
        "preprocessing_fit_scope",
        "split_integrity",
        "shuffled_training_labels",
    ),
    "metrics.json": (
        "report_version",
        "project_version",
        "dataset",
        "split",
        "models",
        "comparison",
        "cross_validation",
        "feature_ablation",
        "leakage_diagnostics",
        "probability_calibration",
        "robustness",
        "class_imbalance",
        "cross_experiment_summary",
    ),
    "robustness_diagnostics.json": (
        "strategy",
        "outer_folds",
        "training_data",
        "validation_labels",
        "models",
        "seed_rule",
        "missing_values",
        "gaussian_noise",
        "summary",
        "interpretation",
    ),
}


def build_interface_contract(
    *,
    project_version: str,
    report_version: int,
    package_exports: Sequence[str],
    metrics_top_level_keys: Sequence[str],
    evaluation_result_fields: Sequence[str],
) -> dict[str, Any]:
    """Build the documented CLI, Python, and artifact interface contract."""
    return {
        "contract_version": 3,
        "project_version": project_version,
        "stability": {
            "status": "stable_1_x",
            "policy": (
                "documented_cli_python_and_artifact_interfaces_remain"
                "_backward_compatible_within_1_x"
            ),
            "breaking_changes": "require_a_new_major_version",
            "scope": (
                "software_interfaces_and_reproduction_workflow"
                "_not_model_deployment_performance"
            ),
        },
        "compatibility": {
            "python": ["3.10", "3.11", "3.12", "3.13", "3.14"],
        },
        "cli": {
            "program": "ml-evaluation-workbench",
            "commands": {
                "evaluate": {
                    "positional_arguments": [
                        {
                            "name": "dataset",
                            "type": "path",
                            "description": "Palmer Penguins CSV path",
                        }
                    ],
                    "options": [
                        {
                            "name": "--output-dir",
                            "type": "path",
                            "default": "results",
                        },
                        {
                            "name": "--random-state",
                            "type": "integer",
                            "default": 42,
                        },
                        {
                            "name": "--test-size",
                            "type": "float",
                            "default": 0.25,
                        },
                        {
                            "name": "--cv-folds",
                            "type": "integer",
                            "default": 5,
                        },
                        {
                            "name": "--calibration-folds",
                            "type": "integer",
                            "default": 3,
                        },
                    ],
                    "writes_checksum_manifest": True,
                },
                "verify": {
                    "positional_arguments": [
                        {
                            "name": "artifact_dir",
                            "type": "path",
                            "description": (
                                "Directory containing generated artifacts "
                                "and checksums.sha256"
                            ),
                        }
                    ],
                    "options": [],
                },
            },
            "exit_codes": {
                "0": "success",
                "2": "argument_input_or_io_error",
            },
        },
        "python_api": {
            "package_exports": list(package_exports),
            "evaluate_dataset": {
                "parameters": [
                    {
                        "name": "frame",
                        "type": "pandas.DataFrame",
                        "required": True,
                    },
                    {
                        "name": "random_state",
                        "type": "integer",
                        "default": 42,
                    },
                    {
                        "name": "test_size",
                        "type": "float",
                        "default": 0.25,
                    },
                    {
                        "name": "cv_folds",
                        "type": "integer",
                        "default": 5,
                    },
                    {
                        "name": "calibration_folds",
                        "type": "integer",
                        "default": 3,
                    },
                ],
                "returns": "EvaluationResult",
            },
            "verify_artifact_manifest": {
                "parameters": [
                    {
                        "name": "output_dir",
                        "type": "str | pathlib.Path",
                        "required": True,
                    }
                ],
                "returns": "integer artifact count",
            },
            "evaluation_result_fields": list(evaluation_result_fields),
        },
        "reports": {
            "metrics_json": {
                "report_version": report_version,
                "top_level_keys": list(metrics_top_level_keys),
            },
            "generated_artifacts": [
                dict(artifact) for artifact in GENERATED_ARTIFACTS
            ],
            "csv_columns": {
                path: list(columns)
                for path, columns in CSV_ARTIFACT_SCHEMAS.items()
            },
            "json_top_level_keys": {
                path: list(keys)
                for path, keys in JSON_ARTIFACT_TOP_LEVEL_KEYS.items()
            },
            "schema_policy": (
                "documented_artifact_names_column_order_and_top_level"
                "_keys_are_stable_within_1_x"
            ),
            "reference_manifest": {
                "path": "checksums.sha256",
                "producer": "ml-evaluation-workbench evaluate",
                "covers_generated_artifacts": True,
            },
        },
        "reproducibility": {
            "dataset_verification_command": (
                "python examples/download_penguins.py --check"
            ),
            "artifact_generation_command": "python examples/run_demo.py",
            "artifact_verification_command": (
                "ml-evaluation-workbench verify results"
            ),
            "manifest_algorithm": "sha256",
            "reference_environment": {
                "operating_system": "ubuntu-latest",
                "python": "3.12",
                "constraints_file": "requirements-reproducibility.txt",
                "ci_expectation": (
                    "byte_exact_match_to_committed_results"
                ),
            },
            "compatibility_matrix_expectation": (
                "successful_execution_and_self_consistent_generated_manifest"
            ),
        },
    }
