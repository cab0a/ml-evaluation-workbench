"""Reproducible machine-learning baseline and variability evaluation."""

from .dataset import (
    DATASET_SHA256,
    DATASET_URL,
    download_dataset,
    load_dataset,
    sha256_file,
    verify_dataset,
)
from .evaluation import EvaluationResult, evaluate_dataset

__all__ = [
    "DATASET_SHA256",
    "DATASET_URL",
    "EvaluationResult",
    "__version__",
    "download_dataset",
    "evaluate_dataset",
    "load_dataset",
    "sha256_file",
    "verify_dataset",
]

__version__ = "0.2.0"
