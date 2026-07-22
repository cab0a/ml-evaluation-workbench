from __future__ import annotations

from pathlib import Path

import pytest

from examples.run_demo import _verify_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_committed_reference_artifacts_match_manifest() -> None:
    assert _verify_manifest(ROOT / "results") == 3


def test_manifest_detects_changed_artifact(tmp_path: Path) -> None:
    for name in ("confusion_matrix.png", "metrics.json", "predictions.csv"):
        (tmp_path / name).write_bytes((ROOT / "results" / name).read_bytes())
    (tmp_path / "checksums.sha256").write_bytes(
        (ROOT / "results" / "checksums.sha256").read_bytes()
    )
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch: metrics.json"):
        _verify_manifest(tmp_path)
