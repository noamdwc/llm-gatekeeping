## CLI tools (`src/cli/`)

This folder contains **human-facing command line entrypoints** intended for running the pipeline locally and for quick manual investigation.

Canonical invocation pattern:

```bash
python -m src.cli.<tool> [args...]
```

### `infer_split.py`

- **Use case**: quick **ML-only** evaluation on an existing split parquet.
- **Inputs**: `data/processed/splits/{test,val,test_unseen}.parquet` and `data/processed/models/ml_baseline.pkl`
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

### `research_external.py`

- **Use case**: produce **research artifacts for one external dataset** (wide parquet + markdown report).
- **Inputs**: dataset config from `configs/default.yaml:external_datasets.<key>`, plus `models/ml_baseline.pkl`
- **Outputs**:
  - `data/processed/research_external/research_external_<key>.parquet`
  - `reports/research_external/research_external_<key>.md`
- **When to use**:
  - running ad-hoc external research for a single dataset
  - invoked by DVC `foreach` stages (`research_external@<key>`)

LLM control is via the `SKIP_LLM` environment variable (defaults to `"1"` = skip).
The CLI flags `--skip-llm` and `--no-skip-llm` override the environment variable
in either direction. When neither flag is provided, `SKIP_LLM` is consulted.

For `--mode llm`, throughput/reliability controls are available:
- `--llm-max-concurrency`: parallel workers for LLM requests (defaults to `llm.max_concurrency` in config)
- `--llm-checkpoint-every`: checkpoint interval in samples (defaults to `llm.checkpoint_every`)
- `--no-llm-resume`: disable resume from an existing predictions parquet

Examples:

```bash
python -m src.cli.research_external --dataset deepset                # ML-only (SKIP_LLM defaults to "1")
SKIP_LLM=0 python -m src.cli.research_external --dataset deepset    # Include LLM predictions via env var
python -m src.cli.research_external --dataset deepset --skip-llm     # Force skip LLM regardless of env
python -m src.cli.research_external --dataset deepset --no-skip-llm  # Force run LLM regardless of env
```

## Differences at a glance

- **`infer_split` vs `predict`**:
  - `infer_split` is *dataset/split-based* and produces a report.
  - `predict` is *text-based* and produces JSON.
- **`research_external` vs `eval_external`**:
  - `research_external` produces *wide research parquet + detailed report*.
  - `eval_external` is a lighter binary-only evaluation report.
