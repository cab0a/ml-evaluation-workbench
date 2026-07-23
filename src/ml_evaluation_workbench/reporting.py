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
