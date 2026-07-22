"""Command-line interface for reproducible baseline evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .dataset import load_dataset, sha256_file
from .evaluation import evaluate_dataset
from .reporting import (
    write_confusion_matrix,
    write_cross_validation_scores,
    write_csv,
    write_json,
    write_predictions,
)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    source = Path(args.dataset)
    output_dir = Path(args.output_dir)
    frame = load_dataset(source)
    result = evaluate_dataset(
        frame,
        random_state=args.random_state,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )
    dataset_summary = result.metrics["dataset"]
    result.metrics = {
        "report_version": result.metrics["report_version"],
        "project_version": __version__,
        "dataset": {
            "source": source.as_posix(),
            "sha256": sha256_file(source),
            **dataset_summary,
        },
        "split": result.metrics["split"],
        "models": result.metrics["models"],
        "comparison": result.metrics["comparison"],
        "cross_validation": result.metrics["cross_validation"],
    }

    metrics_path = write_json(output_dir / "metrics.json", result.metrics)
    predictions_path = write_predictions(
        output_dir / "predictions.csv",
        result.predictions,
    )
    confusion_path = write_confusion_matrix(
        output_dir / "confusion_matrix.png",
        result.confusion,
        result.labels,
    )
    cross_validation_path = write_csv(
        output_dir / "cross_validation_folds.csv",
        result.cross_validation_folds,
    )
    cross_validation_plot_path = write_cross_validation_scores(
        output_dir / "cross_validation_scores.png",
        result.cross_validation_folds,
    )

    dummy = result.metrics["models"]["dummy"]
    logistic = result.metrics["models"]["logistic_regression"]
    cv_logistic = result.metrics["cross_validation"]["models"][
        "logistic_regression"
    ]
    print(f"Dataset rows: {result.metrics['dataset']['rows']}")
    print(f"Training rows: {result.metrics['split']['train_rows']}")
    print(f"Test rows: {result.metrics['split']['test_rows']}")
    print(f"Dummy accuracy: {dummy['accuracy']:.3f}")
    print(f"Logistic regression accuracy: {logistic['accuracy']:.3f}")
    print(f"Logistic regression macro F1: {logistic['macro_f1']:.3f}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print(
        "Logistic regression CV macro F1 mean: "
        f"{cv_logistic['macro_f1']['mean']:.3f}"
    )
    print(
        "Logistic regression CV macro F1 std: "
        f"{cv_logistic['macro_f1']['std']:.3f}"
    )
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Cross-validation fold scores: {cross_validation_path}")
    print(f"Cross-validation scores: {cross_validation_plot_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-evaluation-workbench",
        description="Run deterministic machine-learning baseline evaluations.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Compare dummy and logistic-regression classifiers",
    )
    evaluate_parser.add_argument("dataset", help="Palmer Penguins CSV path")
    evaluate_parser.add_argument(
        "--output-dir",
        default="results",
        help="Artifact directory (default: results)",
    )
    evaluate_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Holdout and cross-validation split seed (default: 42)",
    )
    evaluate_parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout fraction (default: 0.25)",
    )
    evaluate_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5)",
    )
    evaluate_parser.set_defaults(handler=_cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
