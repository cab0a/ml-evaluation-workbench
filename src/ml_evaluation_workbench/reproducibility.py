"""Checksum manifests for generated evaluation artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .dataset import sha256_file
from .interface import GENERATED_ARTIFACT_NAMES


MANIFEST_NAME = "checksums.sha256"


def _write_artifact_manifest(output_dir: str | Path) -> Path:
    """Atomically write checksums for the documented generated artifacts."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ValueError(f"Artifact directory not found: {directory}")
    lines = []
    for name in GENERATED_ARTIFACT_NAMES:
        artifact = directory / name
        if not artifact.is_file():
            raise ValueError(f"Artifact not found: {name}")
        lines.append(f"{sha256_file(artifact)}  {name}\n")

    destination = directory / MANIFEST_NAME
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=directory,
            prefix=f".{MANIFEST_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write("".join(lines))
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def verify_artifact_manifest(output_dir: str | Path) -> int:
    """Verify the complete documented artifact set against its manifest."""
    directory = Path(output_dir)
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        raise ValueError(f"Checksum manifest not found: {manifest}")

    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid checksum line {line_number}")
        digest, name = parts
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Invalid SHA-256 on line {line_number}")
        if name not in GENERATED_ARTIFACT_NAMES:
            raise ValueError(f"Unexpected artifact in manifest: {name}")
        if name in expected:
            raise ValueError(f"Duplicate artifact in manifest: {name}")
        expected[name] = digest

    expected_names = set(GENERATED_ARTIFACT_NAMES)
    if set(expected) != expected_names:
        missing = sorted(expected_names - set(expected))
        raise ValueError("Missing manifest artifacts: " + ", ".join(missing))

    for name, digest in expected.items():
        artifact = directory / name
        if not artifact.is_file():
            raise ValueError(f"Artifact not found: {name}")
        if sha256_file(artifact) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(expected)
