# Project Status — `llm-gatekeeping`

Last updated: 2026-02-12

## Current state (what is functional today)
Based on `dvc.yaml`, `src/`, and the existing `reports/` artifacts, the repo functions primarily as a **research + evaluation pipeline** for a hierarchical prompt-attack “gatekeeper”, with two supported run modes:

- **Research (DVC)**: reproducible heavy run (`dvc repro`) producing research parquets + reports
- **Inference (Bash)**: lightweight fast run (`./run_inference.sh`) producing only what’s needed + reports

### Recent progress (this update)
- **Pipeline organization**: outputs are now organized under `data/processed/{splits,models,predictions,research,research_external}/` and `reports/{research,research_external}/`.
- **Research pipeline correctness**: `src.research` now **consumes precomputed** ML/LLM prediction parquets (no redundant recomputation).
- **External datasets**: wired into DVC as `foreach` stages (`research_external@{dataset}`) so new datasets are **additive** and cached.
- **Inference**: added `./run_inference.sh` for fast runs with arguments.
- **Docs/tests**: README + STATUS updated; test suite passes end-to-end.

### Implemented pipeline (DVC-backed)
| Capability | How to run | Outputs |
|---|---|---|
| **Preprocess Mindgard dataset + labels** | `python -m src.preprocess` | `data/processed/full_dataset.parquet` |
| **Grouped splits + held-out generalization set** | `python -m src.build_splits` | `data/processed/splits/{train,val,test,test_unseen}.parquet` |
| **ML baseline training (+ research predictions)** | `python -m src.ml_classifier.ml_baseline --research` | `data/processed/models/ml_baseline.pkl`, `data/processed/predictions/ml_predictions_*.parquet` |
| **LLM classifier (3-stage, frozen by default)** | `python -m src.llm_classifier.llm_classifier --split test --limit 100 --research` | `data/processed/predictions/llm_predictions_test.parquet` |
| **Research merge + hybrid routing + reports** | `python -m src.research --split test` | `data/processed/research/research_test.parquet`, `reports/research/eval_report_*.md` |
| **External dataset eval (binary-only)** | `python -m src.eval_external --dataset <key> --mode ml|hybrid` | `reports/eval_external_<key>.md` |
| **External dataset research (wide parquet + analysis)** | `python -m src.cli.research_external --dataset <key>` | `data/processed/research_external/research_external_<key>.parquet`, `reports/research_external/research_external_<key>.md` |

### External datasets are additive (DVC foreach + caching)
- External dataset research runs as independent DVC stages (e.g. `research_external@deepset`) via `foreach` in `dvc.yaml`.
- Adding a new dataset (new key in `configs/default.yaml` + add to `dvc.yaml` foreach list) means `dvc repro` computes **only the new dataset stage**; existing datasets remain cached and are not recomputed.

### What the repo demonstrably produces (via `reports/`)
| Mode | In-repo test metrics (sampled) | Notes |
|---|---|---|
| **LLM-only** | Binary accuracy **0.69**, false-negative rate **0.2235** | LLM binary gate is weak (misses unicode/invisible-char attacks) |
| **Hybrid (default thresholds)** | Binary accuracy **0.79**, false-negative rate **0.0941**; category accuracy **0.9610**; unicode type accuracy **1.00** | Also reduces calls: **75 LLM calls / 100 samples** (in report) |

## Completed vs pending (per `project_plan.md` vs repo state)
| Roadmap item (`project_plan.md`) | Status | Evidence |
|---|---:|---|
| **Step 1 — Benign set + proper splits** | ✅ Completed | `src/preprocess.py`, `src/build_splits.py`, `dvc.yaml` stages + generated parquet outputs |
| **Step 2 — Hierarchical LLM classifier** | ✅ Completed | `src/llm_classifier/llm_classifier.py`, `reports/research/eval_report_llm.md` (when LLM stage is run) |
| **Step 3 — ML baseline + hybrid router** | ✅ Completed | `src/ml_classifier/ml_baseline.py`, `src/hybrid_router.py`, `reports/research/eval_report_hybrid.md` |
| **Step 4 — Dynamic few-shot + error analysis** | 🟡 Partially complete | Dynamic few-shot is implemented (`--dynamic`, exemplar bank in `src/embeddings.py`), but a dedicated consolidated `reports/error_analysis.md` deliverable is not present |
| **Step 5 — Reporting + polish** | 🟡 Partially complete | Reporting and run commands are now more consistent (DVC + inference script + organized outputs), but additional polish (smoke tests / notebooks alignment) remains |

## Known blockers / gaps (repo-derived)
### Productization gaps (not implemented)
- **No production service integration**: There is no HTTP API/middleware, policy engine (block/warn/allow), audit logging, or human review workflow beyond the `abstain` label.
- **Data governance constraints unspecified**: Not defined whether prompts may be sent to third-party LLMs (OpenAI) or stored.

### Technical gaps impacting correctness/generalization
- **Cross-dataset generalization is currently poor** (binary-only external evals):
  - `reports/eval_external_deepset.md`: benign F1 **0.0** (massive false positives).
  - `reports/eval_external_jackhhao.md`: false-negative rate **~0.50** (many attacks missed).
  - `reports/eval_external_spml.md` / `reports/eval_external_safeguard.md`: low overall accuracy with severe calibration issues.
- **Confidence calibration instability** on external datasets (high-confidence bins with low accuracy).
- **Benign distribution mismatch**: benign generation is topic-taxonomy based; external benign prompts don’t match training distribution, leading to false positives.

### Repo consistency / operability issues
- **Pipeline “how-to” still requires discipline**: There are now canonical commands (`dvc repro`, `./run_llm.sh`, `./run_inference.sh`), but notebooks and ad-hoc scripts may still diverge. Prefer the DVC pipeline for reproducible research outputs.

## Next steps (next sprint: concrete technical tasks)
| Priority | Task | Outcome |
|---:|---|---|
| P0 | Add a lightweight **smoke test suite** for key CLIs (no network) | Prevent regressions (imports, file outputs, schema) |
| P1 | Improve **benign realism / diversity** (generation prompts, held-out topics, add non-LLM benign sources) | Reduce false positives; raise benign F1 |
| P1 | Add **calibration + threshold tuning** for routing and decisioning | More reliable `confidence_*` and better routing behavior |
| P1 | Build an **error analysis report** artifact (per plan Step 4) | Consolidated failure modes + targeted fixes |
| P2 | Investigate **domain adaptation** (train/fine-tune on external datasets, or multi-source training) | Better cross-dataset generalization |

