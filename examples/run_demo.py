"""Regenerate or verify the committed v0.4 evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from ml_evaluation_workbench.cli import main as workbench_main


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = (
    "confusion_matrix.png",
    "cross_validation_folds.csv",
    "cross_validation_scores.png",
    "feature_ablation_folds.csv",
    "feature_ablation_scores.png",
    "feature_ablation_summary.csv",
    "leakage_diagnostic_folds.csv",
    "leakage_diagnostics.json",
    "metrics.json",
    "model_comparison.csv",
    "predictions.csv",
)
MANIFEST_NAME = "checksums.sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path) -> Path:
    manifest = output_dir / MANIFEST_NAME
    lines = [
        f"{_sha256(output_dir / name)}  {name}\n" for name in ARTIFACT_NAMES
    ]
    manifest.write_text("".join(lines), encoding="utf-8", newline="\n")
    return manifest


def _verify_manifest(output_dir: Path) -> int:
    manifest = output_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise ValueError(f"Checksum manifest not found: {manifest}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid checksum line {line_number}")
        digest, name = parts
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Invalid SHA-256 on line {line_number}")
        if name not in ARTIFACT_NAMES:
            raise ValueError(f"Unexpected artifact in manifest: {name}")
        if name in expected:
            raise ValueError(f"Duplicate artifact in manifest: {name}")
        expected[name] = digest
    if set(expected) != set(ARTIFACT_NAMES):
        missing = sorted(set(ARTIFACT_NAMES) - set(expected))
        raise ValueError("Missing manifest artifacts: " + ", ".join(missing))
    for name, digest in expected.items():
        artifact = output_dir / name
        if not artifact.is_file():
            raise ValueError(f"Artifact not found: {name}")
        if _sha256(artifact) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.verify_only:
        try:
            count = _verify_manifest(args.output_dir)
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
    manifest = _write_manifest(args.output_dir)
    print(f"Checksums: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
