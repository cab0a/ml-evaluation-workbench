from __future__ import annotations

from pathlib import Path

import pytest

from ml_evaluation_workbench import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "penguins.csv"


@pytest.fixture
def penguins_frame():
    return load_dataset(DATASET)
