# Environment Management

This project uses **Conda** and **uv** together.

Conda is the source of truth for the Python runtime and environment isolation.
Use Python 3.11, which satisfies the project metadata in `pyproject.toml`.

uv is used as the package installer and fast command runner inside the active
Conda environment. Do not let uv create a separate project virtualenv for this
repo; run commands with `uv run --active ...`.

## Why both Conda and uv?

Conda provides stable environment isolation for ML dependencies. Keeping uv
inside the active Conda environment avoids accidental mixing with a separate
project virtualenv.

uv provides a fast and consistent way to install Python dependencies and run
project commands without replacing Conda as the environment manager.

## Setup

```bash
conda create -n llm_gate python=3.11
conda activate llm_gate
conda install -c conda-forge uv
uv pip install -r requirements.txt
```

If the environment is already active, all direct commands should go through uv:

```bash
uv run --active python -m src.cli.final_verdict_report
uv run --active dvc repro final_verdict_report
uv run --active pytest
```

## Makefile shortcuts

The Makefile wraps the most common uv invocations:

```bash
make lint
make format
make test
make test-v
make repro
```

Use direct `uv run --active ...` commands when you need custom flags or a
specific DVC stage.
