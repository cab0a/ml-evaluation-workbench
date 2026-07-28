"""Regenerate or verify the committed v1.0 evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from ml_evaluation_workbench.cli import main as workbench_main
from ml_evaluation_workbench.interface import GENERATED_ARTIFACT_NAMES
from ml_evaluation_workbench.reproducibility import (
    verify_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = GENERATED_ARTIFACT_NAMES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.verify_only:
        try:
            count = verify_artifact_manifest(args.output_dir)
        except ValueError as exc:
            print(f"Verification failed: {exc}", file=sys.stderr)
            return 1
        print(f"Verified: {count} reference artifacts")
        return 0

    status = workbench_main(
        [
            "evaluate",
            "data/penguins.csv",
            "--output-dir",
            str(args.output_dir),
            "--random-state",
            "42",
            "--test-size",
            "0.25",
            "--cv-folds",
            "5",
            "--calibration-folds",
            "3",
        ]
    )
    if status != 0:
        raise SystemExit(f"Expected evaluation status 0, got {status}")
    metrics = json.loads(
        (args.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    expected_split = {"train_rows": 258, "test_rows": 86}
    observed_split = {
        key: metrics["split"][key] for key in expected_split
    }
    if observed_split != expected_split:
        raise SystemExit(
            f"Split mismatch: expected {expected_split}, got {observed_split}"
        )
    if (
        metrics["models"]["logistic_regression"]["macro_f1"]
        <= metrics["models"]["dummy"]["macro_f1"]
    ):
        raise SystemExit("Logistic regression did not exceed the dummy baseline")
    cross_validation = metrics["cross_validation"]
    if cross_validation["folds"] != 5:
        raise SystemExit(
            "Cross-validation fold mismatch: "
            f"expected 5, got {cross_validation['folds']}"
        )
    if (
        cross_validation["models"]["logistic_regression"]["macro_f1"]["mean"]
        <= cross_validation["models"]["dummy"]["macro_f1"]["mean"]
    ):
        raise SystemExit(
            "Logistic regression did not exceed the cross-validation baseline"
        )
    if (
        cross_validation["models"]["knn"]["macro_f1"]["mean"]
        <= cross_validation["models"]["dummy"]["macro_f1"]["mean"]
    ):
        raise SystemExit("KNN did not exceed the cross-validation baseline")
    feature_ablation = metrics["feature_ablation"]
    for model_name in ("logistic_regression", "knn"):
        observed = cross_validation["models"][model_name]["macro_f1"]["mean"]
        ablation_reference = feature_ablation["feature_sets"][
            "both_bill_measurements"
        ]["models"][model_name]["macro_f1"]["mean"]
        if ablation_reference != observed:
            raise SystemExit(
                "Feature-ablation reference mismatch for "
                f"{model_name}: expected {observed}, got {ablation_reference}"
            )
    leakage_diagnostics = metrics["leakage_diagnostics"]
    if not leakage_diagnostics["split_integrity"]["passed"]:
        raise SystemExit("Split-integrity diagnostics did not pass")
    shuffled_models = leakage_diagnostics["shuffled_training_labels"][
        "models"
    ]
    for model_name in ("logistic_regression", "knn"):
        shuffled = shuffled_models[model_name]["macro_f1"]["shuffled"]["mean"]
        observed = cross_validation["models"][model_name]["macro_f1"]["mean"]
        if shuffled >= observed:
            raise SystemExit(
                f"Shuffled-label control did not reduce {model_name} macro F1"
            )
    calibration = metrics["probability_calibration"]
    if calibration["outer_folds"] != 5:
        raise SystemExit(
            "Calibration outer-fold mismatch: "
            f"expected 5, got {calibration['outer_folds']}"
        )
    if calibration["inner_calibration"]["folds"] != 3:
        raise SystemExit(
            "Calibration inner-fold mismatch: expected 3, got "
            f"{calibration['inner_calibration']['folds']}"
        )
    for model_name in ("logistic_regression", "knn"):
        for method in ("uncalibrated", "sigmoid"):
            method_metrics = calibration["models_summary"][model_name][
                method
            ]
            for score_name in (
                "accuracy",
                "log_loss",
                "multiclass_brier",
                "top_label_ece",
            ):
                value = method_metrics[score_name]["mean"]
                if not isinstance(value, (int, float)) or value < 0:
                    raise SystemExit(
                        "Invalid calibration metric for "
                        f"{model_name}/{method}/{score_name}: {value}"
                    )
    robustness = metrics["robustness"]
    if robustness["strategy"] != (
        "shared_outer_fold_validation_perturbation"
    ):
        raise SystemExit("Unexpected robustness evaluation strategy")
    if robustness["training_data"] != "unchanged":
        raise SystemExit("Robustness evaluation changed training data")
    if robustness["validation_labels"] != "unchanged":
        raise SystemExit("Robustness evaluation changed validation labels")
    for model_name in ("logistic_regression", "knn"):
        expected = cross_validation["models"][model_name]["macro_f1"]["mean"]
        for perturbation in ("missing_values", "gaussian_noise"):
            baseline = robustness["summary"][perturbation]["conditions"][
                "0"
            ][model_name]["macro_f1"]["mean"]
            if baseline != expected:
                raise SystemExit(
                    "Robustness baseline mismatch for "
                    f"{model_name}/{perturbation}: "
                    f"expected {expected}, got {baseline}"
                )
            highest_condition = (
                "0.5" if perturbation == "missing_values" else "1"
            )
            perturbed = robustness["summary"][perturbation][
                "conditions"
            ][highest_condition][model_name]["macro_f1"]["mean"]
            if perturbed >= baseline:
                raise SystemExit(
                    "Highest robustness severity did not reduce macro F1 for "
                    f"{model_name}/{perturbation}"
                )
    class_imbalance = metrics["class_imbalance"]
    if class_imbalance["strategy"] != (
        "shared_outer_fold_training_class_downsampling"
    ):
        raise SystemExit("Unexpected class-imbalance evaluation strategy")
    if class_imbalance["target_class"] != "Chinstrap":
        raise SystemExit(
            "Unexpected class-imbalance target: "
            f"{class_imbalance['target_class']}"
        )
    if class_imbalance["validation_data"] != "unchanged":
        raise SystemExit("Class-imbalance evaluation changed validation data")
    if class_imbalance["validation_labels"] != "unchanged":
        raise SystemExit(
            "Class-imbalance evaluation changed validation labels"
        )
    for model_name in ("logistic_regression", "knn"):
        expected = cross_validation["models"][model_name]["macro_f1"]["mean"]
        full_retention = class_imbalance["summary"]["conditions"]["1"][
            model_name
        ]["macro_f1"]["mean"]
        if full_retention != expected:
            raise SystemExit(
                "Class-imbalance baseline mismatch for "
                f"{model_name}: expected {expected}, got {full_retention}"
            )
        quarter_retention = class_imbalance["summary"]["conditions"]["0.25"][
            model_name
        ]
        if quarter_retention["macro_f1"]["mean"] >= full_retention:
            raise SystemExit(
                "Quarter target-class retention did not reduce macro F1 for "
                f"{model_name}"
            )
        if (
            quarter_retention["target_class_recall"]["mean"]
            >= class_imbalance["summary"]["conditions"]["1"][model_name][
                "target_class_recall"
            ]["mean"]
        ):
            raise SystemExit(
                "Quarter target-class retention did not reduce target recall "
                f"for {model_name}"
            )
    cross_experiment = metrics["cross_experiment_summary"]
    if cross_experiment["schema_version"] != 1:
        raise SystemExit("Unexpected cross-experiment schema version")
    if cross_experiment["row_count"] != 25:
        raise SystemExit(
            "Cross-experiment row-count mismatch: expected 25, got "
            f"{cross_experiment['row_count']}"
        )
    with (args.output_dir / "cross_experiment_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != cross_experiment["row_count"]:
        raise SystemExit(
            "Cross-experiment CSV does not match its metadata row count"
        )
    if not all(row["folds"] == "5" for row in summary_rows):
        raise SystemExit("Cross-experiment contrast has an unexpected fold count")
    interface_contract = json.loads(
        (args.output_dir / "interface_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if interface_contract["project_version"] != "1.0.0":
        raise SystemExit("Unexpected interface-contract project version")
    if interface_contract["contract_version"] != 3:
        raise SystemExit("Unexpected interface-contract schema version")
    if interface_contract["stability"]["status"] != "stable_1_x":
        raise SystemExit("Unexpected interface-contract stability status")
    if interface_contract["reports"]["metrics_json"]["report_version"] != 8:
        raise SystemExit("Unexpected interface-contract report version")
    if len(interface_contract["reports"]["csv_columns"]) != 15:
        raise SystemExit("Unexpected interface-contract CSV schema count")
    if len(interface_contract["reports"]["json_top_level_keys"]) != 5:
        raise SystemExit("Unexpected interface-contract JSON schema count")
    contracted_artifacts = {
        artifact["path"]
        for artifact in interface_contract["reports"][
            "generated_artifacts"
        ]
    }
    if contracted_artifacts != set(ARTIFACT_NAMES):
        raise SystemExit(
            "Interface-contract artifact list does not match the manifest"
        )
    count = verify_artifact_manifest(args.output_dir)
    print(f"Verified generated artifacts: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
