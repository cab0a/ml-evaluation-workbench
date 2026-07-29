# Compatibility

## 日本語概要

本書は、1.xで維持するコマンドライン、Python API、評価レポート、成果物名、CSV列順、JSON項目、対応Python版の境界を定義します。互換な追加変更と、メジャー版の更新を必要とする変更を区別しています。

安定性の対象と対象外は以下の英語本文を参照してください。

---

Version 1.0.0 defines the documented software and artifact surfaces that remain
backward compatible throughout the 1.x release line.

## Stable 1.x Surfaces

| Boundary | Stable surface |
| --- | --- |
| CLI | `evaluate` and `verify`, documented required arguments, option names and defaults, and exit-code meanings |
| Python API | Package-root exports, documented function parameters and defaults, and `EvaluationResult` fields |
| Complete report | `metrics.json` report version 8 and its documented top-level sections |
| Interface contract | Contract schema version 3 |
| Artifact inventory | All 27 documented artifact names, media types, and roles |
| CSV schemas | Ordered columns for all 15 CSV artifacts |
| JSON schemas | Ordered top-level keys for all five JSON artifacts |
| Runtime | Python 3.10 through 3.14 with declared dependency ranges |
| Reference reproduction | Python 3.12 on Ubuntu with `requirements-reproducibility.txt` |

The generated
[`interface_contract.json`](../results/interface_contract.json) is the
machine-readable inventory of these surfaces.

## Compatible 1.x Changes

A compatible release may add:

- optional CLI commands or arguments;
- package-root exports;
- optional Python parameters or result fields;
- additive report fields;
- new artifacts that do not remove or redefine existing ones; or
- documentation corrections that do not change interface meaning.

Consumers should use structured fields rather than console wording and ignore
unknown additive fields where the documented version policy permits them.

## Breaking Changes

The following require a new major package version or an explicit report or
contract schema transition:

- removing or renaming a documented command, option, or package-root export;
- changing an existing required argument or default covered by the contract;
- changing the meaning of exit code `0` or `2`;
- removing or renaming an `EvaluationResult` field;
- removing or renaming a documented artifact;
- changing a stable CSV column sequence;
- removing, renaming, or redefining a stable JSON top-level key; or
- redefining an existing report-version field incompatibly.

## Compatibility and Reference Environments

The Python 3.10 through 3.14 CI matrix installs the declared dependency ranges,
runs tests, evaluates the dataset, and verifies each newly generated manifest.
This establishes successful, self-consistent execution for the tested
environment.

Byte-exact reference reproduction is narrower. A separate Python 3.12 Ubuntu
job installs [`requirements-reproducibility.txt`](../requirements-reproducibility.txt),
regenerates `results/`, verifies the manifest, and requires an empty Git diff.

PNG bytes and some dependency-sensitive outputs are not promised to match the
reference environment on every supported Python and dependency combination.

## Outside the Stability Boundary

The compatibility promise does not cover:

- imports from package submodules;
- internal helper functions or implementation ordering;
- human-readable console wording;
- numerical results for arbitrary datasets, settings, or dependencies;
- byte-identical output outside the reference environment;
- model performance on external or production data; or
- suitability for deployment, ecological inference, or wildlife decisions.

Release changes are recorded in [`CHANGELOG.md`](../CHANGELOG.md).
