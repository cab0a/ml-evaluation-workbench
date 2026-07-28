from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from ml_evaluation_workbench import EvaluationResult
from ml_evaluation_workbench import __all__ as package_exports
from ml_evaluation_workbench.cli import main

from conftest import DATASET


def test_cli_writes_documented_artifacts(tmp_path: Path, capsys) -> None:
    status = main(
        [
            "evaluate",
            str(DATASET),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert metrics["project_version"] == "0.8.0"
    assert metrics["report_version"] == 8
    assert metrics["dataset"]["rows"] == 344
    assert metrics["cross_validation"]["folds"] == 5
    assert len(predictions) == 86
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 0
    assert (tmp_path / "cross_validation_folds.csv").stat().st_size > 0
    assert (tmp_path / "cross_validation_scores.png").stat().st_size > 0
    comparison = pd.read_csv(tmp_path / "model_comparison.csv")
    assert list(comparison["model"]) == [
        "dummy",
        "logistic_regression",
        "knn",
    ]
    assert (tmp_path / "feature_ablation_folds.csv").stat().st_size > 0
    assert (tmp_path / "feature_ablation_summary.csv").stat().st_size > 0
    assert (tmp_path / "feature_ablation_scores.png").stat().st_size > 0
    assert (tmp_path / "leakage_diagnostic_folds.csv").stat().st_size > 0
    assert (tmp_path / "leakage_diagnostics.json").stat().st_size > 0
    assert (tmp_path / "probability_calibration_folds.csv").stat().st_size > 0
    assert (
        tmp_path / "probability_calibration_summary.csv"
    ).stat().st_size > 0
    assert (
        tmp_path / "probability_calibration_predictions.csv"
    ).stat().st_size > 0
    assert (tmp_path / "probability_calibration_bins.csv").stat().st_size > 0
    assert (tmp_path / "probability_calibration.png").stat().st_size > 0
    assert (tmp_path / "robustness_folds.csv").stat().st_size > 0
    assert (tmp_path / "robustness_summary.csv").stat().st_size > 0
    assert (tmp_path / "robustness_diagnostics.json").stat().st_size > 0
    assert (tmp_path / "robustness.png").stat().st_size > 0
    assert (tmp_path / "class_imbalance_folds.csv").stat().st_size > 0
    assert (tmp_path / "class_imbalance_summary.csv").stat().st_size > 0
    assert (
        tmp_path / "class_imbalance_diagnostics.json"
    ).stat().st_size > 0
    assert (tmp_path / "class_imbalance.png").stat().st_size > 0
    cross_experiment = pd.read_csv(
        tmp_path / "cross_experiment_summary.csv"
    )
    assert len(cross_experiment) == 25
    assert (tmp_path / "cross_experiment_summary.png").stat().st_size > 0
    contract = json.loads(
        (tmp_path / "interface_contract.json").read_text(encoding="utf-8")
    )
    assert contract["project_version"] == "0.8.0"
    assert contract["cli"]["command"] == "evaluate"
    assert contract["reports"]["metrics_json"]["report_version"] == 8
    assert contract["python_api"]["package_exports"] == package_exports
    assert contract["python_api"]["evaluation_result_fields"] == [
        field.name for field in fields(EvaluationResult)
    ]
    output = capsys.readouterr().out
    assert "Dummy accuracy:" in output
    assert "Logistic regression macro F1:" in output
    assert "Logistic regression CV macro F1 mean:" in output
    assert "KNN macro F1:" in output
    assert "KNN CV macro F1 mean:" in output
    assert "KNN shuffled-label macro F1 mean:" in output
    assert "Split integrity check: passed" in output
    assert "logistic_regression CV log loss, uncalibrated:" in output
    assert "knn CV log loss, sigmoid calibrated:" in output
    assert "macro F1 at 50% injected missingness:" in output
    assert "macro F1 at 1.0x feature noise:" in output
    assert "macro F1 at 25% Chinstrap retention:" in output
    assert "Chinstrap recall at 25% retention:" in output
    assert "Cross-experiment contrasts: 25" in output


def test_cli_reports_invalid_test_size(tmp_path: Path, capsys) -> None:
    status = main(
        [
            "evaluate",
            str(DATASET),
            "--output-dir",
            str(tmp_path),
            "--test-size",
            "1.0",
        ]
    )

    assert status == 2
    assert "test_size must be" in capsys.readouterr().err


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == "0.8.0\n"
