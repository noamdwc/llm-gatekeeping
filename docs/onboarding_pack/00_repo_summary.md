# Repository Summary

## Index
- `00_repo_summary.md`
- `01_architecture.md`
- `02_directory_map.md`
- `03_entrypoints_and_runtime.md`
- `04_key_files.md`
- `05_core_flows.md`
- `06_setup_and_dev_workflow.md`
- `07_glossary.md`
- `08_first_week_onboarding.md`
- `09_notebooklm_questions.md`
- `10_known_gotchas.md`
- `11_testing_strategy.md`
- `12_config_and_env.md`
- `99_upload_order.md`
- `sources_to_upload_first.md`
- `raw_README.md`
- `raw_dvc.yaml`
- `raw_configs_default.yaml`
- `raw_preprocess.txt`
- `raw_build_splits.txt`
- `raw_ml_baseline.txt`
- `raw_llm_classifier.txt`
- `raw_hybrid_router.txt`
- `raw_research.txt`
- `raw_evaluate.txt`
- `raw_cli_predict.txt`

## What This Repo Appears To Do
This repository builds and evaluates a hierarchical classifier for prompt injection and jailbreak detection. The system combines:
- a character/Unicode-focused ML baseline in `src/ml_classifier/ml_baseline.py`
- an LLM classifier with judge/refinement logic in `src/llm_classifier/llm_classifier.py`
- a hybrid router in `src/hybrid_router.py`

The main dataset appears to be the Mindgard evaded prompt injection dataset, augmented with benign prompts and optional synthetic benign data. This is defined in `configs/default.yaml` and implemented in `src/preprocess.py`.

## Who Or What It Serves
This appears to serve research and evaluation work more than production serving. The strongest evidence:
- the canonical workflow is a DVC pipeline in `dvc.yaml`
- outputs are parquet artifacts and Markdown reports under `data/processed/` and `reports/`
- shell wrappers like `run_llm.sh` and `run_inference.sh` are aimed at experiment runs, not long-running services

There is also a lightweight CLI prediction surface in `src/cli/predict.py` for ad hoc inference.

## Main Technologies
- Python
- pandas, NumPy, scikit-learn
- Hugging Face `datasets`
- OpenAI-compatible APIs via NVIDIA NIM or OpenAI, selected in `src/llm_provider.py`
- DVC for reproducible pipeline orchestration
- pytest for tests
- Weights & Biases hooks in training/inference code

## Likely Runtime Model
- Offline pipeline mode: `dvc repro` executes preprocessing, splits, ML training, research aggregation, and report generation
- Optional LLM pipeline mode: `run_llm.sh` temporarily unfreezes the LLM stage in DVC and runs the full research path
- Lightweight local inference mode: `run_inference.sh` or `python -m src.cli.predict`

## First Concepts To Learn
1. The label hierarchy: `label_binary`, `label_category`, `label_type`
2. The repo is research-first, not service-first
3. The ML model is a Unicode/character-attack specialist and intentionally excludes NLP attacks from training
4. The LLM classifier is expensive and therefore cached and often frozen in DVC
5. The hybrid router trusts ML only for high-confidence Unicode-lane adversarial predictions
6. Most important artifacts are parquet files under `data/processed/` and reports under `reports/`
7. `configs/default.yaml` is the central configuration file
8. DVC stage boundaries in `dvc.yaml` are the best map of the system
9. External datasets are evaluated through separate DVC `foreach` stages
10. The test suite is broad and heavily unit-oriented, with 339 collected tests at inspection time
