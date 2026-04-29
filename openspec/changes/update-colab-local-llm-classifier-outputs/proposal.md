## Why

`notebooks/colab_local_llm_classifier.ipynb` currently writes predictions for one configured split at a time, which makes full local LLM coverage across main, unseen, safeguard, and external datasets manual and error-prone. The notebook should produce the complete set of classifier-only prediction artifacts needed by downstream research and evaluation in one run.

## What Changes

- Replace the single `SPLIT` output mode with a configured run list covering `train`, `val`, `test`, `unseen_val`, `unseen_test`, and `safeguard_test`.
- Add external dataset output support for `deepset` and `jackhhao`.
- Write checkpoint and final parquet outputs separately for each split or external dataset, using the repo's established main prediction and external prediction locations.
- Preserve the existing classifier-only prediction schema, Transformers local inference path, checkpoint resume behavior, and output validation for every generated artifact.
- Treat the requested `safegurd_test` target as the existing repo split name `safeguard_test`.

## Capabilities

### New Capabilities
- `colab-local-llm-classifier-batch-outputs`: Covers the Colab local LLM classifier notebook's ability to generate classifier-only prediction parquet outputs for all configured main splits and selected external datasets.

### Modified Capabilities

## Impact

- Affected notebook: `notebooks/colab_local_llm_classifier.ipynb`
- Main split inputs under `data/processed/splits` on Drive are read for `train`, `val`, `test`, `unseen_val`, `unseen_test`, and `safeguard_test`.
- External datasets are loaded through existing repo configuration and helpers for `deepset` and `jackhhao`.
- Prediction outputs expand from one selected split to multiple per-dataset parquet files and checkpoints.
