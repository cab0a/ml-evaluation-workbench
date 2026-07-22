# Dataset Provenance

`penguins.csv` is the simplified Palmer Penguins dataset maintained by Allison
Horst, Alison Hill, and Kristen Gorman. It contains measurements for 344
penguins from three species observed in the Palmer Archipelago, Antarctica.

- Upstream repository: <https://github.com/allisonhorst/palmerpenguins>
- Pinned commit: `8957207b78d6ccd1b4654a9dd9c9041b657478ab`
- Upstream path: `inst/extdata/penguins.csv`
- Retrieved: 2026-07-22
- SHA-256: `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`
- Data license: [CC0 1.0 Universal](https://allisonhorst.github.io/palmerpenguins/LICENSE.html)

The dataset is committed so the v0.1 evaluation can run without a network
request. Regenerate it from the pinned source with:

```bash
python examples/download_penguins.py
```

Verify the committed bytes without downloading:

```bash
python examples/download_penguins.py --check
```

The project code is licensed separately under the MIT License.
