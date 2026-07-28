"""Machine-readable contracts for the documented pre-1.0 interfaces."""

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
        "contract_version": 1,
        "project_version": project_version,
        "stability": {
            "status": "pre_1_0",
            "policy": (
                "documented_interfaces_are_versioned_additive_changes"
                "_remain_possible_before_1_0"
            ),
        },
        "compatibility": {
            "python": ["3.10", "3.11", "3.12", "3.13", "3.14"],
        },
        "cli": {
            "program": "ml-evaluation-workbench",
            "command": "evaluate",
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
            "reference_manifest": {
                "path": "checksums.sha256",
                "producer": "examples/run_demo.py",
                "covers_generated_artifacts": True,
            },
        },
    }
