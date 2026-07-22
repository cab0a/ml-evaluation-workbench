from __future__ import annotations

from pathlib import Path

import pytest

from ml_evaluation_workbench import DATASET_SHA256, load_dataset, verify_dataset

from conftest import DATASET


def test_committed_dataset_has_expected_provenance() -> None:
    assert verify_dataset(DATASET) == DATASET_SHA256


def test_dataset_has_expected_rows_columns_and_classes() -> None:
    frame = load_dataset(DATASET)

    assert len(frame) == 344
    assert list(frame.columns) == [
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
        "year",
    ]
    assert sorted(frame["species"].unique()) == [
        "Adelie",
        "Chinstrap",
        "Gentoo",
    ]


def test_dataset_checksum_rejects_modified_content(tmp_path: Path) -> None:
    modified = tmp_path / "penguins.csv"
    modified.write_bytes(DATASET.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Dataset checksum mismatch"):
        verify_dataset(modified)
