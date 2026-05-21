## CLI tools (`src/cli/`)

This folder contains **human-facing command line entrypoints** intended for running the pipeline locally and for quick manual investigation.

Canonical invocation pattern:

```bash
uv run --active python -m src.cli.<tool> [args...]
```

Activate the Conda environment first (`conda activate llm_gate`). uv should run
with `--active` so CLI tools use the Conda-managed ML dependencies.

## Differences at a glance

- **Canonical DVC pipeline vs CLI tools**:
  - the DVC + Colab handoff path in `README.md` produces the final verdict report.
  - retained CLIs are for targeted local checks and manual inspection.
