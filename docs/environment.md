# Environment Management

This project uses **Conda** and **uv** together.

Conda is the source of truth for the Python runtime and environment isolation.
Use Python 3.11, which satisfies the project metadata in `pyproject.toml`.
`environment.yml` is the source of truth for Python dependencies.

uv is used as the fast command runner inside the active Conda environment.
Python dependencies are installed by Conda from `environment.yml`. Do not let
uv create a separate project virtualenv for this repo; run commands with
`uv run --active --no-project ...`. Do not use `uv sync` for this repository.

## Why both Conda and uv?

Conda provides stable environment isolation for ML dependencies. Keeping uv
inside the active Conda environment avoids accidental mixing with a separate
project virtualenv.

uv provides a fast and consistent way to run project commands without replacing
Conda as the environment and dependency manager.

`pyproject.toml` exists for project metadata and tool configuration. `uv.lock`
does not define the runtime environment; the Conda environment file does.

## Setup

```bash
conda env create -f environment.yml
conda activate llm-gate
```

If the environment is already active, all direct commands should go through uv:

```bash
uv run --active --no-project python -m src.cli.final_verdict_report
uv run --active --no-project dvc repro final_verdict_report
uv run --active --no-project pytest
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

Use direct `uv run --active --no-project ...` commands when you need custom flags or a
specific DVC stage.
