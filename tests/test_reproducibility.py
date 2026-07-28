from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ml_evaluation_workbench.interface import GENERATED_ARTIFACT_NAMES
from ml_evaluation_workbench.reproducibility import (
    _write_artifact_manifest,
    verify_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_committed_reference_artifacts_match_manifest() -> None:
    assert verify_artifact_manifest(ROOT / "results") == 27


def test_manifest_detects_changed_artifact(tmp_path: Path) -> None:
    for name in GENERATED_ARTIFACT_NAMES:
        (tmp_path / name).write_bytes((ROOT / "results" / name).read_bytes())
    (tmp_path / "checksums.sha256").write_bytes(
        (ROOT / "results" / "checksums.sha256").read_bytes()
    )
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch: metrics.json"):
        verify_artifact_manifest(tmp_path)


def test_manifest_writer_covers_the_documented_artifact_set(
    tmp_path: Path,
) -> None:
    for name in GENERATED_ARTIFACT_NAMES:
        (tmp_path / name).write_bytes((ROOT / "results" / name).read_bytes())

    manifest = _write_artifact_manifest(tmp_path)

    assert manifest == tmp_path / "checksums.sha256"
    assert verify_artifact_manifest(tmp_path) == len(
        GENERATED_ARTIFACT_NAMES
    )


def test_reference_constraints_pin_direct_runtime_dependencies() -> None:
    constraints = [
        Requirement(line)
        for line in (
            ROOT / "requirements-reproducibility.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    pinned = {
        canonicalize_name(requirement.name): requirement
        for requirement in constraints
    }

    for name in ("matplotlib", "numpy", "pandas", "scikit-learn"):
        constrained = pinned[canonicalize_name(name)]
        assert len(list(constrained.specifier)) == 1
        assert next(iter(constrained.specifier)).operator == "=="
