# Project Status — `llm-gatekeeping`

Last updated: 2026-02-12

## Current state (what is functional today)
Based on `dvc.yaml`, `src/`, and the existing `reports/` artifacts, the repo functions primarily as a **research + evaluation pipeline** for a hierarchical prompt-attack “gatekeeper”.

### Implemented pipeline (DVC-backed)
| Capability | How to run | Outputs |
|---|---|---|
| **Generate benign prompts (LLM)** | `python -m src.generate_benign` | `data/processed/generated_benign.parquet` |
| **Preprocess Mindgard dataset + labels** | `python -m src.preprocess` | `data/processed/full_dataset.parquet` |
| **Grouped splits + held-out generalization set** | `python -m src.build_splits` | `data/processed/{train,val,test,test_unseen}.parquet` |
| **ML baseline training** | `python -m src.ml_classifier.ml_baseline` | `data/processed/ml_baseline.pkl` |
| **LLM classifier eval (3-stage)** | `python -m src.llm_classifier.llm_classifier --split test --limit 100` | `data/processed/predictions_test.csv` |
| **Hybrid router eval (ML→LLM)** | `python -m src.hybrid_router --limit 100` | `reports/eval_report_hybrid.md` |
| **Evaluate predictions CSV** | `python -m src.evaluate --predictions ...` | `reports/eval_report_llm.md` |
| **External dataset eval (binary-only)** | `python -m src.eval_external --dataset <key> --mode ml|hybrid` | `reports/eval_external_<key>.md` |
| **External dataset research (wide parquet + analysis)** | `python -m src.research_external --dataset <key> --skip-llm` | `data/processed/research_external_<key>.parquet`, `reports/research_external_<key>.md` |

### What the repo demonstrably produces (via `reports/`)
| Mode | In-repo test metrics (sampled) | Notes |
|---|---|---|
| **LLM-only** | Binary accuracy **0.69**, false-negative rate **0.2235** | LLM binary gate is weak (misses unicode/invisible-char attacks) |
| **Hybrid (default thresholds)** | Binary accuracy **0.79**, false-negative rate **0.0941**; category accuracy **0.9610**; unicode type accuracy **1.00** | Also reduces calls: **75 LLM calls / 100 samples** (in report) |

## Completed vs pending (per `project_plan.md` vs repo state)
| Roadmap item (`project_plan.md`) | Status | Evidence |
|---|---:|---|
| **Step 1 — Benign set + proper splits** | ✅ Completed | `src/preprocess.py`, `src/build_splits.py`, `dvc.yaml` stages + generated parquet outputs |
| **Step 2 — Hierarchical LLM classifier** | ✅ Completed | `src/llm_classifier/llm_classifier.py`, `reports/eval_report_llm.md` |
| **Step 3 — ML baseline + hybrid router** | ✅ Completed | `src/ml_classifier/ml_baseline.py`, `src/hybrid_router.py`, `reports/eval_report_hybrid.md` |
| **Step 4 — Dynamic few-shot + error analysis** | 🟡 Partially complete | Dynamic few-shot is implemented (`--dynamic`, exemplar bank in `src/embeddings.py`), but a dedicated consolidated `reports/error_analysis.md` deliverable is not present |
| **Step 5 — Reporting + polish** | 🟡 Partially complete | Reports exist; however docs/CLI entrypoints are inconsistent (see “Known blockers”) |

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
- **CLI/docs drift**: `dvc.yaml` uses `python -m src.ml_classifier.ml_baseline` and `python -m src.llm_classifier.llm_classifier`, but some code/docs still reference older module paths (e.g., `src/predict.py` imports `src.ml_baseline` which does not exist in the current tree). This likely breaks `python -m src.predict` until imports are updated.
- **Pipeline “how-to” is messy / inconsistently formatted**: Usage instructions are spread across multiple places (README, DVC stages, notebooks, and CLIs) with inconsistent naming/entrypoints. We need a single, well-formatted runbook with canonical commands and expected artifacts.

## Next steps (next sprint: concrete technical tasks)
| Priority | Task | Outcome |
|---:|---|---|
| P0 | **Fix module-path drift + update CLI/docs** (`src/predict.py`, README commands) | All documented commands run against current package layout |
| P0 | **Organize + format pipeline usage docs** (single runbook; align README ↔ DVC ↔ CLI help; add “quickstart” + “full pipeline” sections) | Clear, consistent “how to run” documentation and fewer operator errors |
| P0 | Add a lightweight **smoke test suite** for key CLIs (no network) | Prevent regressions (imports, file outputs, schema) |
| P1 | Improve **benign realism / diversity** (generation prompts, held-out topics, add non-LLM benign sources) | Reduce false positives; raise benign F1 |
| P1 | Add **calibration + threshold tuning** for routing and decisioning | More reliable `confidence_*` and better routing behavior |
| P1 | Build an **error analysis report** artifact (per plan Step 4) | Consolidated failure modes + targeted fixes |
| P2 | Investigate **domain adaptation** (train/fine-tune on external datasets, or multi-source training) | Better cross-dataset generalization |

