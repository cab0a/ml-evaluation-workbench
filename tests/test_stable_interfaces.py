from __future__ import annotations

import csv
import inspect
import json
from dataclasses import fields
from pathlib import Path

from ml_evaluation_workbench import EvaluationResult
from ml_evaluation_workbench import __all__ as package_exports
from ml_evaluation_workbench import evaluate_dataset
from ml_evaluation_workbench import verify_artifact_manifest
from ml_evaluation_workbench.cli import build_parser
from ml_evaluation_workbench.interface import (
    CSV_ARTIFACT_SCHEMAS,
    GENERATED_ARTIFACT_NAMES,
    JSON_ARTIFACT_TOP_LEVEL_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

STABLE_PACKAGE_EXPORTS = (
    "DATASET_SHA256",
    "DATASET_URL",
    "EvaluationResult",
    "__version__",
    "download_dataset",
    "evaluate_dataset",
    "load_dataset",
    "sha256_file",
    "verify_artifact_manifest",
    "verify_dataset",
)
STABLE_EVALUATION_RESULT_FIELDS = (
    "metrics",
    "predictions",
    "cross_validation_folds",
    "model_comparison",
    "feature_ablation_folds",
    "feature_ablation_summary",
    "leakage_diagnostic_folds",
    "leakage_diagnostics",
    "probability_calibration_folds",
    "probability_calibration_summary",
    "probability_calibration_predictions",
    "probability_calibration_bins",
    "robustness_folds",
    "robustness_summary",
    "robustness_diagnostics",
    "class_imbalance_folds",
    "class_imbalance_summary",
    "class_imbalance_diagnostics",
    "cross_experiment_summary",
    "confusion",
    "labels",
)
STABLE_ARTIFACT_NAMES = (
    "class_imbalance.png",
    "class_imbalance_diagnostics.json",
    "class_imbalance_folds.csv",
    "class_imbalance_summary.csv",
    "confusion_matrix.png",
    "cross_experiment_summary.csv",
    "cross_experiment_summary.png",
    "cross_validation_folds.csv",
    "cross_validation_scores.png",
    "feature_ablation_folds.csv",
    "feature_ablation_scores.png",
    "feature_ablation_summary.csv",
    "interface_contract.json",
    "leakage_diagnostic_folds.csv",
    "leakage_diagnostics.json",
    "metrics.json",
    "model_comparison.csv",
    "predictions.csv",
    "probability_calibration.png",
    "probability_calibration_bins.csv",
    "probability_calibration_folds.csv",
    "probability_calibration_predictions.csv",
    "probability_calibration_summary.csv",
    "robustness.png",
    "robustness_diagnostics.json",
    "robustness_folds.csv",
    "robustness_summary.csv",
)


def test_stable_python_exports_and_result_fields() -> None:
    assert tuple(package_exports) == STABLE_PACKAGE_EXPORTS
    assert tuple(field.name for field in fields(EvaluationResult)) == (
        STABLE_EVALUATION_RESULT_FIELDS
    )


def test_stable_public_function_signatures() -> None:
    evaluate_parameters = inspect.signature(evaluate_dataset).parameters
    assert tuple(evaluate_parameters) == (
        "frame",
        "random_state",
        "test_size",
        "cv_folds",
        "calibration_folds",
    )
    assert evaluate_parameters["frame"].kind is (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    for name, default in (
        ("random_state", 42),
        ("test_size", 0.25),
        ("cv_folds", 5),
        ("calibration_folds", 3),
    ):
        assert evaluate_parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert evaluate_parameters[name].default == default

    verify_parameters = inspect.signature(
        verify_artifact_manifest
    ).parameters
    assert tuple(verify_parameters) == ("output_dir",)
    assert verify_parameters["output_dir"].default is inspect.Parameter.empty


def test_stable_cli_commands_and_defaults() -> None:
    parser = build_parser()
    evaluate_args = parser.parse_args(["evaluate", "input.csv"])
    assert evaluate_args.command == "evaluate"
    assert evaluate_args.dataset == "input.csv"
    assert evaluate_args.output_dir == "results"
    assert evaluate_args.random_state == 42
    assert evaluate_args.test_size == 0.25
    assert evaluate_args.cv_folds == 5
    assert evaluate_args.calibration_folds == 3

    verify_args = parser.parse_args(["verify", "results"])
    assert verify_args.command == "verify"
    assert verify_args.artifact_dir == "results"


def test_stable_artifact_inventory_and_schemas() -> None:
    assert GENERATED_ARTIFACT_NAMES == STABLE_ARTIFACT_NAMES
    assert set(CSV_ARTIFACT_SCHEMAS) == {
        name for name in STABLE_ARTIFACT_NAMES if name.endswith(".csv")
    }
    assert set(JSON_ARTIFACT_TOP_LEVEL_KEYS) == {
        name for name in STABLE_ARTIFACT_NAMES if name.endswith(".json")
    }

    for name, expected_columns in CSV_ARTIFACT_SCHEMAS.items():
        with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
            observed_columns = tuple(next(csv.reader(handle)))
        assert observed_columns == expected_columns

    for name, expected_keys in JSON_ARTIFACT_TOP_LEVEL_KEYS.items():
        value = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        assert tuple(value) == expected_keys


def test_stable_contract_and_report_versions() -> None:
    contract = json.loads(
        (RESULTS / "interface_contract.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (RESULTS / "metrics.json").read_text(encoding="utf-8")
    )

    assert contract["contract_version"] == 3
    assert contract["project_version"] == "1.0.0"
    assert contract["stability"]["status"] == "stable_1_x"
    assert contract["reports"]["metrics_json"]["report_version"] == 8
    assert metrics["report_version"] == 8
    assert metrics["project_version"] == "1.0.0"
