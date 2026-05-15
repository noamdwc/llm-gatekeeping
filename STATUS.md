# Project Status - `llm-gatekeeping`

Last updated: 2026-05-15

## Current State

The repo's canonical run path is the DVC + Colab handoff + final verdict
pipeline documented in `README.md`.

Final documented artifact:

```bash
reports/pipeline_final_verdict_report.md
```

## Canonical DVC Graph

| Capability | How to run | Outputs |
| --- | --- | --- |
| Synthetic benign inputs | `dvc repro generate_synthetic_benign@B` through `@F` | `data/processed/synthetic_benign/synthetic_benign_<category>.parquet` |
| Preprocess + splits | `dvc repro build_splits` | `data/processed/full_dataset.parquet`, `data/processed/splits/{train,val,test,unseen_val,unseen_test,safeguard_test}.parquet` |
| ML baseline predictions | `dvc repro ml_model` | `data/processed/models/ml_baseline.pkl`, `data/processed/predictions/ml_predictions_*.parquet` |
| DeBERTa predictions | `dvc repro deberta_model` | `artifacts/deberta_classifier/`, `data/processed/predictions/deberta_predictions_*.parquet` |
| DeBERTa external predictions | `dvc repro deberta_external@deepset deberta_external@jackhhao` | `data/processed/predictions_external/deberta_predictions_external_<dataset>.parquet` |
| Manual Colab handoff validation | `dvc repro -s validate_colab_handoff` | `reports/colab_handoff_validation.json` |
| Escalation model | `dvc repro train_escalating_model` | `data/processed/models/escalating_model.pkl`, `data/processed/research/escalating_model_eval_*.parquet` |
| Selective judge outputs | `dvc repro judge_colab_local_predictions@test` and related foreach stages | `*_colab_local_judged.parquet` |
| Final verdict report | `dvc repro final_verdict_report` | `reports/pipeline_final_verdict_report.md` |

## Current Handoff Status

The canonical pipeline is being rerun from the start. `build_splits` is up to
date after the rerun, and the old downloaded Deepset Colab handoff failure
should be treated as stale until fresh Colab classifier artifacts are produced
and `validate_colab_handoff` is run again.

Do not weaken validation or fall back to legacy hosted LLM outputs. If the
fresh handoff artifacts still fail validation, treat that as a new artifact
quality issue to fix at the handoff boundary.

## Non-Canonical Paths

Legacy hosted-LLM research stages, component markdown reports, baseline
comparison scripts, and post-hoc risk-model reports have been removed or are
explicitly non-canonical. Use `README.md` for the current start-to-finish run
path.
