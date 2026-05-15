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

## Differences at a glance

- **Canonical DVC pipeline vs CLI tools**:
  - the DVC + Colab handoff path in `README.md` produces the final verdict report.
  - retained CLIs are for targeted local checks and manual inspection.
