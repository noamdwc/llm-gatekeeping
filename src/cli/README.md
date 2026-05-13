## CLI tools (`src/cli/`)

This folder contains **human-facing command line entrypoints** intended for running the pipeline locally and for quick manual investigation.

Canonical invocation pattern:

```bash
python -m src.cli.<tool> [args...]
```

### `infer_split.py`

- **Use case**: quick **ML-only** evaluation on an existing split parquet.
- **Inputs**: `data/processed/splits/{test,val,unseen_val,unseen_test}.parquet` and `data/processed/models/ml_baseline.pkl`
- **Outputs**: `reports/research/inference_ml_<split>.md`
- **When to use**: you want a fast report without running DVC.

Example:

```bash
python -m src.cli.infer_split --mode ml --split test
```

### `predict.py`

- **Use case**: classify **arbitrary prompt text** (stdin or file) and print **JSON per input**.
- **Inputs**: raw text lines (not dataset splits); requires `models/ml_baseline.pkl` for ML/hybrid.
- **Outputs**: JSON to stdout (one JSON object per input line).
- **When to use**: spot-checking, demos, or integration into other scripts via piping.

Examples:

```bash
echo "some prompt" | python -m src.cli.predict --mode ml --pretty
echo "some prompt" | python -m src.cli.predict --mode hybrid --pretty
```

## Differences at a glance

- **`infer_split` vs `predict`**:
  - `infer_split` is *dataset/split-based* and produces a report.
  - `predict` is *text-based* and produces JSON.
- **Canonical DVC pipeline vs CLI tools**:
  - the DVC + Colab handoff path in `README.md` produces the final verdict report.
  - these CLIs are retained for targeted local checks and manual inspection.
