# Setup And Dev Workflow

## Dependencies
Dependencies are managed through `requirements.txt`. There is no `pyproject.toml`, `package.json`, `Dockerfile`, or CI workflow visible in the repo.

Main packages:
- `datasets`
- `openai`
- `pandas`
- `scikit-learn`
- `numpy`
- `pyyaml`
- `pyarrow`
- `wandb`
- `dvc`
- `pytest`, `pytest-cov`, `pytest-mock`

## Local Setup
The README suggests:

```bash
conda create -n llm_gate python=3.14
conda activate llm_gate
pip install -r requirements.txt
echo "NVIDIA_API_KEY=nvapi-..." > .env
huggingface-cli login
```

If using OpenAI instead of NVIDIA NIM:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
```

## Common Local Workflows

### Full reproducible pipeline
```bash
dvc repro
```

### Full pipeline including frozen LLM stage
```bash
./run_llm.sh
```

### Lightweight inference
```bash
./run_inference.sh --mode ml --split test
./run_inference.sh --mode hybrid --split test --limit 100
```

### Single-text prediction
```bash
echo "some text" | python -m src.cli.predict --mode hybrid --pretty
```

### Synthetic benign generation
```bash
./run_synth.sh
python -m src.cli.generate_synthetic_benign --category C --limit 50
```

## Testing
Run:
```bash
pytest
```

Observed pytest config in `pytest.ini`:
- `testpaths = tests`
- `addopts = -v --tb=short`

At inspection time, `pytest --collect-only -q` collected 339 tests.

## Formatting / Linting / Type Checks
I could not confirm dedicated commands or config for:
- Black
- Ruff
- Flake8
- mypy
- pyright

That likely means style/tooling is lightweight or handled manually.

## Recommended Dev Workflow For A New Engineer
1. Install deps and set env vars
2. Read `README.md`, `configs/default.yaml`, and `dvc.yaml`
3. Run `pytest`
4. Run `./run_inference.sh --mode ml --split test`
5. If working on pipeline logic, run targeted DVC stages rather than full repros
6. Only run LLM paths when needed because they use API tokens

## Safe First Changes
Good low-risk changes:
- add or improve tests
- improve report text in `src/cli/eval_new.py`
- adjust documentation
- modify pure metric formatting in `src/evaluate.py`

Higher-risk changes:
- modifying split logic in `src/build_splits.py`
- changing ML training scope in `src/ml_classifier/ml_baseline.py`
- changing LLM prompt contracts in `src/llm_classifier/prompts.py`
- changing hybrid routing semantics in `src/hybrid_router.py` or `src/research.py`

## What Is Unclear
- No explicit local dev bootstrap beyond README and shell scripts
- No visible CI pipeline to show required checks before merge
- No explicit style or static-analysis gate was discoverable
