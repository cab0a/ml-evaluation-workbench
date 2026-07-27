from __future__ import annotations

from pathlib import Path

import pytest

from ml_evaluation_workbench import (
    EvaluationResult,
    evaluate_dataset,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "penguins.csv"


@pytest.fixture(scope="session")
def penguins_frame():
    return load_dataset(DATASET)


@pytest.fixture(scope="session")
def evaluation_result(penguins_frame) -> EvaluationResult:
    return evaluate_dataset(penguins_frame)
