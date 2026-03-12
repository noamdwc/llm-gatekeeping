# Directory Map

## Guided Tree
```text
.
├── configs/
│   └── default.yaml
├── src/
│   ├── cli/
│   ├── llm_classifier/
│   ├── ml_classifier/
│   ├── preprocess.py
│   ├── build_splits.py
│   ├── research.py
│   ├── hybrid_router.py
│   ├── evaluate.py
│   ├── eval_external.py
│   ├── embeddings.py
│   ├── llm_cache.py
│   ├── llm_provider.py
│   ├── synthetic_benign.py
│   └── validators.py
├── tests/
├── reports/
├── research_docs/
├── data/
├── dvc.yaml
├── README.md
├── run_inference.sh
├── run_llm.sh
└── run_synth.sh
```

## Top-Level Areas

### `src/`
Core code. This is the most important directory for onboarding.

- `src/preprocess.py`
  - Core
  - Builds the labeled combined dataset
- `src/build_splits.py`
  - Core
  - Creates grouped train/val/test/test_unseen parquets
- `src/ml_classifier/`
  - Core
  - Classical ML baseline and related feature utilities
- `src/llm_classifier/`
  - Core
  - Prompting, constants, and LLM classifier/judge logic
- `src/research.py`
  - Core
  - Batch merge and hybrid research logic
- `src/hybrid_router.py`
  - Core
  - Runtime hybrid routing path
- `src/evaluate.py`
  - Core/supporting
  - Shared metrics and report rendering
- `src/cli/`
  - Supporting
  - Thin command wrappers around core modules
- `src/eval_external.py`
  - Core/supporting
  - Binary evaluation for external datasets
- `src/embeddings.py`
  - Supporting but important
  - Embedding retrieval and exemplar-bank logic
- `src/llm_cache.py`, `src/llm_provider.py`
  - Infra/supporting
  - API provider switching and local response caching
- `src/synthetic_benign.py`, `src/validators.py`
  - Supporting but important to data quality
  - Synthetic benign generation and validation pipeline

### `configs/`
Configuration area.

- `configs/default.yaml`
  - Core config
  - Defines dataset, label taxonomy, model hyperparameters, thresholds, and external datasets

### `tests/`
Test area.

- Broad unit-test coverage across pipeline components
- Good place to learn expected behavior quickly
- Especially useful:
  - `tests/test_hybrid_router.py`
  - `tests/test_llm_classifier.py`
  - `tests/test_research.py`
  - `tests/test_research_external.py`

### `reports/`
Generated outputs and prior evaluation artifacts.

- Supporting/output
- Useful for understanding what the pipeline produces
- Should not be treated as the main source of implementation truth

### `research_docs/`
Human-written research explanations and templates.

- Supporting/contextual
- Helpful for background
- Lower authority than code and config for current behavior

### `data/`
Artifact area.

- Mostly pipeline outputs under `data/processed/`
- Core to runtime, but not source-controlled logic

## Important Single Files
- `README.md`
  - Best high-level intro
- `dvc.yaml`
  - Best execution graph
- `requirements.txt`
  - Dependency list
- `pytest.ini`
  - Minimal pytest config
- `run_inference.sh`, `run_llm.sh`, `run_synth.sh`
  - Shell entrypoints for common workflows

## Areas That Look More Contextual Than Core
- `interview_presentation/`
- `notebooks/`
- `eda.ipynb`
- `PRD.md`, `STATUS.md`, `project_plan.md`

These are useful context, but a new engineer should read code and config first.
