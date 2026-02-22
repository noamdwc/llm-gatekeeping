# PRD — LLM Security Gatekeeper (`llm-gatekeeping`)

## 1) Problem statement
LLM applications are vulnerable to **prompt injection / jailbreak attempts**, including *evasion* variants that modify text to bypass simple filters while preserving intent. This project provides a **pre-processing “gatekeeper”** that classifies incoming prompts as:

- **Level 0 (binary)**: `adversarial` vs `benign`
- **Level 1 (category)**: `unicode_attack` vs `nlp_attack` (for adversarial)
- **Level 2 (type)**: **12 Unicode sub-types** (NLP sub-types are collapsed to `nlp_attack` based on research evidence that they are not separable)

The core product need is **reducing security failures (false negatives)** while balancing **latency** and **cost** via **hybrid routing** (ML-first, escalate uncertain cases to an LLM).

## 2) Target users (personas)
Derived from the repository’s intended use as a “gatekeeper” in front of an LLM.

| Persona | Primary goal | What they need from this project |
|---|---|---|
| **LLM Application Engineer** | Protect downstream LLM from adversarial prompts | Simple inference interface (batch/CLI), fast defaults, clear outputs (`routed_to`, confidences) |
| **Security / Trust & Safety Engineer** | Minimize “attack passes” | Low false-negative rate, review queue (`abstain`), reporting + error analysis |
| **ML / Applied Scientist** | Improve detection + generalization | Reproducible pipeline, configurable labels/splits, evaluation & calibration tooling |

## 3) Product scope
- **In scope (current repo)**: offline training/evaluation pipeline + inference components (ML, LLM, hybrid) for *prompt text classification*.
- **Explicitly not implemented yet**: production API/service integration (middleware, SDK, gateway), policy enforcement (block/warn/allow), storage/retention controls, human review workflows beyond `abstain`.

## 4) Core workflow (as implemented)

| Stage | What it does | Implementation |
|---|---|---|
| **Preprocess** | Load HF dataset; add hierarchical labels; merge benign set; compute `prompt_hash` | `python -m src.preprocess` → `data/processed/full_dataset.parquet` |
| **Splits** | Grouped split by `prompt_hash` + held-out attacks/topics → `test_unseen` | `python -m src.build_splits` |
| **ML baseline** | Char n-gram TF-IDF + Unicode features → LogisticRegression per hierarchy level | `python -m src.ml_classifier.ml_baseline --research` → `data/processed/models/ml_baseline.pkl` |
| **LLM classifier** | Classifier + judge OpenAI pattern with JSON output; optional dynamic few-shot | `python -m src.llm_classifier.llm_classifier [--dynamic]` |
| **Hybrid router** | ML-first; if ML conf < threshold → LLM cascade; if LLM conf < threshold → `abstain` | `python -m src.hybrid_router` |
| **Evaluation** | Hierarchy-level metrics + calibration + usage stats | `python -m src.evaluate` + `reports/research/eval_report_*.md` |

## 5) Functional requirements

### 5.1 Inference (gatekeeping) requirements
- **FR-1**: Accept one or more **prompt texts** and return a structured result with:
  - `label_binary`, `label_category`, `label_type`
  - confidences (`confidence_*`)
  - routing metadata (`routed_to` = `ml` / `llm` / `abstain`)
- **FR-2**: Support **hybrid routing** with configurable thresholds:
  - `ml_confidence_threshold` (default `0.85`)
  - `llm_confidence_threshold` (default `0.7`)
- **FR-3**: Provide **LLM-only** inference (classifier + conditional judge) returning binary + category outputs.
- **FR-4**: Provide **ML-only** inference (no network calls, local model artifact).

### 5.2 Pipeline + reproducibility requirements
- **FR-5**: Config-driven label scheme, splits, thresholds, model params (`configs/default.yaml`).
- **FR-6**: Reproducible pipeline orchestration (DVC stages in `dvc.yaml`).
- **FR-7**: Generate evaluation reports as Markdown under `reports/`.
- **FR-8**: Support external dataset evaluation (binary-only) via:
  - `python -m src.eval_external --dataset <key> --mode ml|hybrid`
  - `python -m src.cli.research_external --dataset <key> ...` (wide parquet + report)

### 5.3 Dynamic few-shot (optional, implemented)
- **FR-9**: Build/load an exemplar bank and retrieve similar examples for dynamic few-shot prompting (`--dynamic`, `src/embeddings.py`).

## 6) Non-functional requirements (NFRs)
Grounded in current code and measured reports.

| NFR | Target / constraint | Evidence / notes |
|---|---|---|
| **Security (secrets)** | OpenAI key via environment (`.env`), not committed | `.env` is referenced in docs and code (`dotenv.load_dotenv()`) |
| **Latency** | ML path should be “near-instant”; LLM calls add ~0.5s each | Hybrid report shows ~0.511s avg per LLM call (sampled eval) |
| **Cost control** | Prefer ML handling; escalate only uncertain prompts | Hybrid report shows 75 LLM calls / 100 samples at default threshold |
| **Determinism** | Stable classification for eval/research runs | LLM temperature set to `0` in config |
| **Scalability** | Batch ML first; LLM only on subset | Hybrid router runs ML vectorized on all rows then escalates |
| **Calibration** | Confidence should be usable for routing/UX | Calibration is computed; reports show high-confidence bins can be inaccurate on some external sets |

## 7) Success metrics
Primary metric emphasis is consistent with the repo’s evaluation framework.

| Metric | Why it matters | Where measured |
|---|---|---|
| **False-negative rate (binary)** | “Attack passes” = security failure | `src/evaluate.py`, `reports/research/eval_report_*.md` |
| **Adversarial recall (binary)** | Catch attacks | same |
| **Benign recall / false-positive rate** | Usability friction | same |
| **Category accuracy (unicode vs NLP)** | Guides mitigations + analysis | same |
| **Unicode type accuracy** | Diagnoses attack technique | same |
| **LLM calls per 100 prompts** | Cost proxy | `reports/research/eval_report_hybrid.md` |
| **LLM avg latency** | UX/SLO proxy | `reports/research/eval_report_hybrid.md` |
| **Cross-dataset performance** | Generalization | `reports/research_external/research_external_*.md` |

## 8) Known risks (from current repo evidence)
- **Domain shift / generalization**: External evaluations show large drops (e.g., very low benign F1 in some datasets).
- **Confidence calibration issues**: High-confidence bins can have poor accuracy on external sets (see calibration tables).
- **Benign dataset realism**: Benign generation relies on an LLM topic taxonomy; quality and representativeness drive false positives.

## 9) Open questions (need your input to finalize product “who/why”)
The repo documents the research + prototype pipeline, but **deployment/product context** is not specified. To make the PRD “production-ready,” I need:

1. **Where will this run?** (library in an app backend, API gateway, sidecar, browser, etc.)
2. **What action does the gatekeeper take?** (block, warn, route to human review, redact, log-only)
3. **Your tolerance targets**: acceptable false-positive rate vs false-negative rate (and which is prioritized for v1).
4. **Traffic + latency budget**: expected QPS / concurrency and target p95/p99.
5. **Data handling constraints**: are prompts allowed to be sent to a 3rd-party LLM (OpenAI) and/or stored for analysis?
