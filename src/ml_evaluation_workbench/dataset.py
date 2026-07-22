"""Palmer Penguins dataset loading and provenance checks."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


DATASET_COMMIT = "8957207b78d6ccd1b4654a9dd9c9041b657478ab"
DATASET_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    f"{DATASET_COMMIT}/inst/extdata/penguins.csv"
)
DATASET_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"
EXPECTED_COLUMNS = (
    "species",
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
    "year",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Dataset not found: {source}")
    observed = sha256_file(source)
    if observed != DATASET_SHA256:
        raise ValueError(
            "Dataset checksum mismatch: "
            f"expected {DATASET_SHA256}, observed {observed}"
        )
    return observed


def download_dataset(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "ml-evaluation-workbench/0.1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    observed = hashlib.sha256(content).hexdigest()
    if observed != DATASET_SHA256:
        raise ValueError(
            "Downloaded dataset checksum mismatch: "
            f"expected {DATASET_SHA256}, observed {observed}"
        )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def load_dataset(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source, na_values=["NA"])
    observed_columns = tuple(frame.columns)
    if observed_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset columns: " + ", ".join(observed_columns)
        )
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame["species"].isna().any():
        raise ValueError("Dataset contains a missing target value")
    return frame.reset_index(drop=True)
