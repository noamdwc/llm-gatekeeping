# Upload Order

## Recommended Upload Order For NotebookLM
1. `00_repo_summary.md`
2. `01_architecture.md`
3. `05_core_flows.md`
4. `03_entrypoints_and_runtime.md`
5. `04_key_files.md`
6. `12_config_and_env.md`
7. `11_testing_strategy.md`
8. `02_directory_map.md`
9. `10_known_gotchas.md`
10. `07_glossary.md`
11. `08_first_week_onboarding.md`
12. `09_notebooklm_questions.md`
13. `06_setup_and_dev_workflow.md`
14. `sources_to_upload_first.md`

## Why This Order
- Start with summary and architecture so NotebookLM builds the right mental model first.
- Upload flows and runtime next so questions about execution and debugging stay grounded.
- Upload key files and config before deeper Q&A, because they anchor the model to concrete source paths.
- Leave onboarding plans and prompt lists later; they are useful, but they are not the primary factual substrate.

## Best Raw Repo Files To Upload Alongside The Pack
If you only add a few raw source files, prioritize:
- `README.md`
- `dvc.yaml`
- `configs/default.yaml`
- `src/preprocess.py`
- `src/ml_classifier/ml_baseline.py`
- `src/llm_classifier/llm_classifier.py`
- `src/hybrid_router.py`
- `src/research.py`
- `src/evaluate.py`
- `src/cli/predict.py`
