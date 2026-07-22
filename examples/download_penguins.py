"""Download or verify the pinned Palmer Penguins dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_evaluation_workbench import download_dataset, verify_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/penguins.csv"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        digest = verify_dataset(args.output)
        print(f"Verified dataset: {args.output}")
        print(f"SHA-256: {digest}")
        return 0
    destination = download_dataset(args.output)
    print(f"Downloaded dataset: {destination}")
    print(f"SHA-256: {verify_dataset(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
