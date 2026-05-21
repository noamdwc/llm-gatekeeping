# Project Status - `llm-gatekeeping`

Last updated: 2026-05-21

## Current State

The repo's canonical run path is the DVC + Colab handoff + final verdict
pipeline documented in `README.md`. The Colab handoff works end-to-end and
feeds the escalation model and final verdict report without falling back to
legacy hosted LLM outputs.

The classifier model runs in Google Colab for cheap GPU access: hosted
NIM/OpenAI endpoints no longer expose `logprobs`, and a Colab GPU is the most
cost-effective way to recover token-level confidence.

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

The Colab handoff is working: fresh classifier artifacts pass
`dvc repro -s validate_colab_handoff` and feed cleanly into
`train_escalating_model` and `final_verdict_report`. The canonical
`reports/pipeline_final_verdict_report.md` covers all configured splits
(internal `test`, `unseen_test`, `safeguard_test`; external `deepset`,
`jackhhao`) with a 4.72% judge-escalation rate at threshold 0.5.

If a future handoff fails validation, fix the artifact at the handoff boundary
rather than weakening validation or falling back to legacy hosted LLM outputs.

## Non-Canonical Paths

Legacy hosted-LLM research stages, component markdown reports, baseline
comparison scripts, and post-hoc risk-model reports have been removed or are
explicitly non-canonical. Use `README.md` for the current start-to-finish run
path.
