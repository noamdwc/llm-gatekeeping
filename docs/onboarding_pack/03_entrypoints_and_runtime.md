# Entrypoints And Runtime

## Main Ways The Repo Runs

### 1. DVC Research Pipeline
Defined in `dvc.yaml`.

Main stages:
- `preprocess` -> `python -m src.preprocess`
- `build_splits` -> `python -m src.build_splits`
- `ml_model` -> `python -m src.ml_classifier.ml_baseline --research --no-wandb`
- `llm_classifier` -> `python -m src.llm_classifier.llm_classifier --split test --research --no-wandb`
- `research` -> `python -m src.research --split test`
- `eval_new` -> `python -m src.cli.eval_new --split test --config configs/default.yaml --only-main`

There are additional `foreach` stages for external datasets:
- `research_external_llm`
- `research_external`
- `eval_new_external`

### 2. Shell Wrappers
- `run_llm.sh`
  - Unfreezes the DVC `llm_classifier` stage, runs `dvc repro`, then re-freezes it
- `run_inference.sh`
  - Lightweight inference wrapper for `ml`, `hybrid`, or `llm`
- `run_synth.sh`
  - Runs synthetic benign generation outside DVC

### 3. Direct Python Module Entrypoints
- `python -m src.preprocess`
- `python -m src.build_splits`
- `python -m src.ml_classifier.ml_baseline`
- `python -m src.llm_classifier.llm_classifier`
- `python -m src.research`
- `python -m src.cli.eval_new`
- `python -m src.cli.predict`
- `python -m src.cli.infer_split`
- `python -m src.cli.research_external`

## Config Loading
Configuration is loaded almost everywhere through `src.utils.load_config()` or a local equivalent.

Default config path:
- `configs/default.yaml`

Typical pattern:
- CLI parses `--config`
- if omitted, code falls back to `configs/default.yaml`

`src/preprocess.py` has its own local `load_config`, but it points to the same default path.

## Important Environment Variables
- `NVIDIA_API_KEY`
  - Required when `LLM_PROVIDER=nim` or omitted, since NIM is the default
- `OPENAI_API_KEY`
  - Required when `LLM_PROVIDER=openai`
- `LLM_PROVIDER`
  - `nim` or `openai`
- `SKIP_LLM`
  - Referenced in the repo’s LLM/external workflows; relevant for skipping expensive steps

`.env` is loaded in several LLM-related modules via `python-dotenv`.

## Runtime Paths To Understand

### Offline research path
1. Download/load source dataset
2. Build combined labeled parquet
3. Create grouped splits
4. Train ML model and persist it
5. Optionally generate LLM prediction parquet
6. Merge outputs into research parquet
7. Generate Markdown reports

### Lightweight inference path
1. Load saved ML model from `data/processed/models/ml_baseline.pkl`
2. Read input texts from stdin or file
3. Run `ml`, `llm`, or `hybrid` prediction path
4. Print JSON or write Markdown report depending on entrypoint

## If I Want To Run This Locally, What Do I Read First?
Read in this order:
1. `README.md`
2. `configs/default.yaml`
3. `dvc.yaml`
4. `run_inference.sh`
5. `src/cli/predict.py`
6. `src/preprocess.py` and `src/build_splits.py`

Then choose one path:
- for offline reproducible work: `dvc repro`
- for cheap local inference: `./run_inference.sh --mode ml --split test`
- for interactive prediction: `echo "text" | python -m src.cli.predict --mode hybrid --pretty`

## Local Commands Discoverable In Repo
```bash
pip install -r requirements.txt
pytest
dvc repro
./run_llm.sh
./run_inference.sh --mode ml --split test
python -m src.cli.predict --mode ml --pretty
python -m src.cli.generate_synthetic_benign --category all
```

## What Does Not Exist
I could not confirm any of the following:
- a web server entrypoint
- background worker framework
- CI workflows under `.github/workflows/`
- Docker-based local runtime

That reinforces the interpretation that this is a local/research pipeline repo rather than a deployed service.
