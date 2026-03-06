# Project Status — `llm-gatekeeping`

Last updated: 2026-02-28

## Current state (what is functional today)
The repo is functioning as a research/evaluation pipeline for hierarchical prompt-attack detection, with two supported run modes:

- **Research (DVC)**: reproducible pipeline with tracked research artifacts (`dvc repro`)
- **Inference (Bash)**: targeted fast runs (`./run_inference.sh`)

## Implemented pipeline (current DVC graph)
| Capability | How to run | Outputs |
|---|---|---|
| **Preprocess Mindgard dataset + labels** | `python -m src.preprocess` | `data/processed/full_dataset.parquet` |
| **Grouped splits + held-out generalization split** | `python -m src.build_splits` | `data/processed/splits/{train,val,test,test_unseen}.parquet` |
| **ML baseline train + research predictions** | `python -m src.ml_classifier.ml_baseline --research --no-wandb` | `data/processed/models/ml_baseline.pkl`, `data/processed/predictions/ml_predictions_{val,test,test_unseen}.parquet` |
| **LLM classifier (classifier+judge, frozen by default)** | `python -m src.llm_classifier.llm_classifier --split test --research --no-wandb` | `data/processed/predictions/llm_predictions_test.parquet` |
| **Research merge + strict hybrid routing** | `python -m src.research --split test` | `data/processed/research/research_test.parquet` |
| **Main eval markdowns** | `python -m src.cli.eval_new --split test --only-main` | `reports/research/eval_report_{ml,llm,hybrid}.md`, `reports/research/summary_report.md` |
| **External per-dataset research parquet** | `python -m src.cli.research_external --dataset <key>` | `data/processed/research_external/research_external_<key>.parquet` |
| **External per-dataset markdowns (from research parquets)** | `python -m src.cli.eval_new --only-external --dataset <key>` | `reports/research_external/research_external_<key>.md` |

## External datasets and caching behavior
- DVC stages are dynamic via `foreach: ${external_datasets}` for both `research_external` and `eval_new_external`.
- Adding a dataset under `external_datasets` in `configs/default.yaml` allows running only new dataset stages (existing stage outputs remain cached unless dependencies/params changed).

## Current metrics (from tracked report artifacts)
Source: `reports/research/summary_report.md` and `reports/research/eval_report_*.md`.
The checked-in hybrid markdown was previously a partial-coverage artifact; canonical hybrid reports are now strict and require full LLM coverage for all escalations.

| Mode | Rows | Accuracy | Adv F1 | Benign F1 | FNR | Notes |
|---|---:|---:|---:|---:|---:|---|
| **ML (unicode scope)** | 996 | 0.9829 | 0.9905 | 0.9128 | 0.0133 | Evaluated on benign + unicode scope (NLP excluded) |
| **Hybrid** | 1690 | 0.6213 | 0.7506 | 0.2138 | 0.3966 | Routing: `ml=1643`, `llm=47` |
| **LLM** | 100 | 0.7500 | 0.8408 | 0.4186 | 0.2584 | Subsampled LLM coverage in this artifact |

External combined (ML on current research parquets): **accuracy 0.1556**, **FNR 0.9482**, **benign F1 0.2182**.

## Completed vs pending (against `project_plan.md`)
| Roadmap item | Status | Evidence |
|---|---:|---|
| **Step 1 — Benign set + proper splits** | ✅ Completed | `src/preprocess.py`, `src/build_splits.py`, DVC stages + split outputs |
| **Step 2 — Hierarchical LLM classifier** | ✅ Completed | `src/llm_classifier/llm_classifier.py`, `reports/research/eval_report_llm.md` |
| **Step 3 — ML baseline + hybrid router** | ✅ Completed | `src/ml_classifier/ml_baseline.py`, `src/hybrid_router.py`, `reports/research/eval_report_hybrid.md` (strict) |
| **Step 4 — Dynamic few-shot + error analysis** | 🟡 Partially complete | `--dynamic` + exemplar bank implemented (`src/embeddings.py`), but no consolidated `reports/error_analysis.md` |
| **Step 5 — Reporting + polish** | 🟡 Partially complete | Eval report flow is standardized via `src.cli.eval_new`; polish gaps remain |

## Known blockers / gaps
### Productization gaps
- No production HTTP/API integration, policy engine (block/warn/allow), audit trail, or human-review workflow beyond `abstain`.
- Data governance constraints are not encoded (for example retention/logging policy for prompts sent to third-party providers such as NVIDIA NIM).

### Technical gaps affecting generalization
- Cross-dataset performance remains poor on current external artifacts:
  - `deepset`: accuracy **0.4483**, FNR **0.2500**
  - `jackhhao`: accuracy **0.2634**, FNR **0.9496**
  - `safeguard`: accuracy **0.3553**, FNR **0.9676**
  - `spml`: accuracy **0.1260**, FNR **0.9506**
- Calibration is unstable on external data (high-confidence bins with low realized accuracy in multiple reports).
- Benign distribution mismatch is still a likely driver of false positives/false negatives out of distribution.

### Operability notes
- Canonical reproducible path: DVC stages + report generation via `src.cli.eval_new`.
- Legacy reports exist at repo root from older runs; prefer `reports/research/` and `reports/research_external/` as current outputs.

## Verification snapshot
- `pytest -q`: **300 passed** (latest local run).

## Next steps
| Priority | Task | Outcome |
|---:|---|---|
| P0 | Build a consolidated `reports/error_analysis.md` from current artifacts | Sharper failure-mode targeting |
| P1 | Improve benign realism/diversity (including non-LLM benign sources) | Better external benign recall and calibration |
| P1 | Re-tune routing and confidence calibration using current specialist policy | Recover hybrid FNR/accuracy trade-off |
| P1 | Expand CLI smoke tests for end-to-end command contracts | Prevent pipeline/report regressions |
| P2 | Explore domain adaptation / multi-source training for external datasets | Improve cross-dataset robustness |
