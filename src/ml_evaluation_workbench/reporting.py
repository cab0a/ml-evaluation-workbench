"""Atomic evaluation artifact writers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def write_predictions(path: str | Path, frame: pd.DataFrame) -> Path:
    return write_csv(path, frame)


def write_confusion_matrix(
    path: str | Path,
    matrix: np.ndarray,
    labels: tuple[str, ...],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    figure, axis = plt.subplots(figsize=(6.0, 5.0), dpi=120)
    try:
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_title("Logistic Regression Confusion Matrix")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks(range(len(labels)), labels=labels, rotation=20)
        axis.set_yticks(range(len(labels)), labels=labels)
        threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = int(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def write_cross_validation_scores(
    path: str | Path,
    frame: pd.DataFrame,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    score_names = ("accuracy", "balanced_accuracy", "macro_f1")
    titles = ("Accuracy", "Balanced Accuracy", "Macro F1")
    model_styles = {
        "dummy": ("Majority-class dummy", "#6b7280"),
        "logistic_regression": ("Logistic regression", "#2563eb"),
        "knn": ("5-nearest neighbors", "#059669"),
    }
    figure, axes = plt.subplots(
        1,
        len(score_names),
        figsize=(12.0, 4.0),
        dpi=120,
        sharey=True,
    )
    try:
        for axis, score_name, title in zip(
            axes,
            score_names,
            titles,
            strict=True,
        ):
            for model_name, (label, color) in model_styles.items():
                model_rows = frame[frame["model"] == model_name]
                axis.plot(
                    model_rows["fold"],
                    model_rows[score_name],
                    marker="o",
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
            axis.set_title(title)
            axis.set_xlabel("Fold")
            axis.set_xticks(sorted(frame["fold"].unique()))
            axis.set_ylim(0.0, 1.02)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Score")
        axes[0].legend(loc="lower right", fontsize=8)
        figure.suptitle("Stratified Cross-Validation by Fold")
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def write_feature_ablation_scores(
    path: str | Path,
    frame: pd.DataFrame,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    feature_sets = (
        "bill_length_only",
        "bill_depth_only",
        "both_bill_measurements",
    )
    feature_labels = ("Bill length", "Bill depth", "Both")
    model_styles = {
        "logistic_regression": ("Logistic regression", "#2563eb"),
        "knn": ("5-nearest neighbors", "#059669"),
    }
    positions = np.arange(len(feature_sets), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=120)
    try:
        for offset, (model_name, (label, color)) in zip(
            (-width / 2.0, width / 2.0),
            model_styles.items(),
            strict=True,
        ):
            model_rows = frame.set_index(
                ["feature_set", "model"]
            ).loc[
                [(feature_set, model_name) for feature_set in feature_sets]
            ]
            axis.bar(
                positions + offset,
                model_rows["macro_f1_mean"],
                width=width,
                yerr=model_rows["macro_f1_std"],
                capsize=4,
                color=color,
                label=label,
            )
        axis.set_title("Feature Ablation under Shared Cross-Validation Folds")
        axis.set_xlabel("Feature set")
        axis.set_ylabel("Macro F1, mean ± fold standard deviation")
        axis.set_xticks(positions, labels=feature_labels)
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right")
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def write_probability_calibration(
    path: str | Path,
    bins: pd.DataFrame,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    model_titles = {
        "logistic_regression": "Logistic regression",
        "knn": "5-nearest neighbors",
    }
    method_styles = {
        "uncalibrated": ("Uncalibrated", "#6b7280", "o"),
        "sigmoid": ("Sigmoid calibrated", "#d97706", "s"),
    }
    figure, axes = plt.subplots(
        1,
        len(model_titles),
        figsize=(10.0, 4.5),
        dpi=120,
        sharex=True,
        sharey=True,
    )
    try:
        for axis, (model_name, title) in zip(
            axes,
            model_titles.items(),
            strict=True,
        ):
            axis.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color="#9ca3af",
                linestyle="--",
                linewidth=1.0,
                label="Perfect calibration",
            )
            for calibration_name, (
                label,
                color,
                marker,
            ) in method_styles.items():
                method_rows = bins[
                    (bins["model"] == model_name)
                    & (bins["calibration"] == calibration_name)
                    & (bins["sample_count"] > 0)
                ]
                axis.plot(
                    method_rows["mean_confidence"],
                    method_rows["empirical_accuracy"],
                    marker=marker,
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
            axis.set_title(title)
            axis.set_xlabel("Mean predicted confidence")
            axis.set_xlim(0.0, 1.02)
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Empirical accuracy")
        axes[1].legend(loc="lower right", fontsize=8)
        figure.suptitle(
            "Top-Label Reliability under Shared Outer Folds"
        )
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def write_robustness_scores(
    path: str | Path,
    summary: pd.DataFrame,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    experiment_settings = {
        "missing_values": (
            "Injected Missing Values",
            "Injected fraction of observed cells",
        ),
        "gaussian_noise": (
            "Gaussian Feature Noise",
            "Noise standard deviation multiplier",
        ),
    }
    model_styles = {
        "logistic_regression": ("Logistic regression", "#2563eb"),
        "knn": ("5-nearest neighbors", "#059669"),
    }
    figure, axes = plt.subplots(
        1,
        len(experiment_settings),
        figsize=(10.0, 4.5),
        dpi=120,
        sharey=True,
    )
    try:
        for axis, (
            perturbation,
            (title, x_label),
        ) in zip(
            axes,
            experiment_settings.items(),
            strict=True,
        ):
            perturbation_rows = summary[
                summary["perturbation"] == perturbation
            ]
            for model_name, (label, color) in model_styles.items():
                model_rows = (
                    perturbation_rows[
                        perturbation_rows["model"] == model_name
                    ]
                    .sort_values("severity")
                    .reset_index(drop=True)
                )
                axis.errorbar(
                    model_rows["severity"],
                    model_rows["macro_f1_mean"],
                    yerr=model_rows["macro_f1_std"],
                    marker="o",
                    capsize=4,
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
            severities = sorted(
                perturbation_rows["severity"].unique()
            )
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.set_xticks(severities)
            if perturbation == "missing_values":
                axis.set_xticklabels(
                    [f"{severity:.0%}" for severity in severities]
                )
            axis.set_ylim(0.0, 1.02)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Macro F1, mean ± fold standard deviation")
        axes[1].legend(loc="lower left", fontsize=8)
        figure.suptitle(
            "Validation-Feature Robustness under Shared Outer Folds"
        )
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def write_class_imbalance_scores(
    path: str | Path,
    summary: pd.DataFrame,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    model_styles = {
        "logistic_regression": ("Logistic regression", "#2563eb"),
        "knn": ("5-nearest neighbors", "#059669"),
    }
    target_class = str(summary["target_class"].iloc[0])
    panels = (
        (
            "Macro F1",
            "macro_f1_mean",
            "macro_f1_std",
            "Macro F1, mean ± fold standard deviation",
        ),
        (
            f"{target_class} Recall",
            "target_class_recall_mean",
            "target_class_recall_std",
            f"{target_class} recall, mean ± fold standard deviation",
        ),
    )
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(10.0, 4.5),
        dpi=120,
        sharey=True,
    )
    try:
        for axis, (
            title,
            mean_column,
            standard_deviation_column,
            y_label,
        ) in zip(axes, panels, strict=True):
            for model_name, (label, color) in model_styles.items():
                model_rows = (
                    summary[summary["model"] == model_name]
                    .sort_values("retention_fraction")
                    .reset_index(drop=True)
                )
                retention_percent = (
                    model_rows["retention_fraction"].to_numpy() * 100
                )
                axis.errorbar(
                    retention_percent,
                    model_rows[mean_column],
                    yerr=model_rows[standard_deviation_column],
                    marker="o",
                    capsize=4,
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
            axis.set_title(title)
            axis.set_xlabel(f"Retained {target_class} training rows")
            axis.set_xticks([25, 50, 75, 100])
            axis.set_xticklabels(["25%", "50%", "75%", "100%"])
            axis.set_ylabel(y_label)
            axis.set_ylim(0.0, 1.02)
            axis.grid(axis="y", alpha=0.25)
        axes[1].legend(loc="lower right", fontsize=8)
        figure.suptitle(
            "Training-Class Retention Sensitivity under Shared Outer Folds"
        )
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def write_cross_experiment_summary(
    path: str | Path,
    summary: pd.DataFrame,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    macro_rows = (
        summary[summary["metric"] == "macro_f1"]
        .iloc[::-1]
        .reset_index(drop=True)
    )
    experiment_colors = {
        "model_comparison": "#2563eb",
        "feature_ablation": "#7c3aed",
        "shuffled_label_control": "#dc2626",
        "validation_robustness": "#d97706",
        "class_imbalance": "#059669",
    }
    colors = [
        experiment_colors[experiment]
        for experiment in macro_rows["experiment"]
    ]
    positions = np.arange(len(macro_rows))
    values = macro_rows["preferred_effect_mean"].to_numpy(dtype=float)
    errors = macro_rows[
        "condition_minus_reference_std"
    ].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 8.0), dpi=120)
    try:
        axis.barh(
            positions,
            values,
            xerr=errors,
            color=colors,
            alpha=0.88,
            capsize=3,
        )
        axis.axvline(0.0, color="#111827", linewidth=1.0)
        axis.set_yticks(positions)
        axis.set_yticklabels(macro_rows["display_label"], fontsize=8)
        axis.set_xlabel(
            "Paired mean macro-F1 effect "
            "(positive favors the condition)"
        )
        axis.set_title(
            "Representative Cross-Experiment Macro-F1 Contrasts"
        )
        axis.grid(axis="x", alpha=0.25)
        for position, value, error in zip(
            positions,
            values,
            errors,
            strict=True,
        ):
            offset = error + 0.015
            text_position = (
                value + offset if value >= 0 else value - offset
            )
            alignment = "left" if value >= 0 else "right"
            axis.text(
                text_position,
                position,
                f"{value:+.3f}",
                va="center",
                ha=alignment,
                fontsize=7,
            )
        lower = float(np.min(values - errors))
        upper = float(np.max(values + errors))
        span = max(upper - lower, 0.2)
        axis.set_xlim(
            lower - span * 0.12,
            upper + span * 0.12,
        )
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "ml-evaluation-workbench"},
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination
