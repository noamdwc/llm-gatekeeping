# Sources To Upload First

## Best 5 Generated Docs

### `00_repo_summary.md`
Best short orientation and index for the whole pack.

### `01_architecture.md`
Best high-level system model and component map.

### `05_core_flows.md`
Best document for tracing behavior end-to-end.

### `03_entrypoints_and_runtime.md`
Best document for practical execution and startup questions.

### `04_key_files.md`
Best document for jumping from summary to actual source files.

## Best 10 Raw Code / Config Files

### `README.md`
The most compact repo-authored overview.

### `dvc.yaml`
The canonical execution graph.

### `configs/default.yaml`
The central behavior/config surface.

### `src/preprocess.py`
Shows how raw data becomes the repo’s canonical labeled dataset.

### `src/build_splits.py`
Shows split semantics and unseen-attack handling.

### `src/ml_classifier/ml_baseline.py`
Shows the ML baseline design, scope, and persistence behavior.

### `src/llm_classifier/llm_classifier.py`
Shows LLM prompting, judge flow, concurrency, and output schema.

### `src/hybrid_router.py`
Shows the live hybrid routing policy.

### `src/research.py`
Shows artifact merge logic and research-mode hybrid behavior.

### `src/evaluate.py`
Shows how success is measured and reported.

## Reasoning
This set is intentionally small. It gives NotebookLM:
- one human summary layer
- one architecture layer
- one flow layer
- the config and orchestration truth
- the five most important source modules

That is usually enough to answer onboarding questions without drowning the context window in lower-signal files.

For convenience, this onboarding pack also includes local copies of these raw files in `docs/onboarding_pack/` with `raw_` prefixes. Python files were copied with `.txt` extensions for easier plain-text upload.
