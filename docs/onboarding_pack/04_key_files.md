# Key Files

This list favors files that define behavior, boundaries, or operational workflow.

## Core Docs And Orchestration

### `README.md`
- Why it matters: best short repo overview
- Responsibility: explains problem, pipeline modes, commands, and project structure
- What to look for: the DVC runbook, hierarchy description, and CLI examples

### `dvc.yaml`
- Why it matters: execution graph for the canonical workflow
- Responsibility: defines staged dependencies, params, and outputs
- What to look for: frozen/expensive LLM stage, external `foreach` stages, and artifact outputs

### `configs/default.yaml`
- Why it matters: central behavior/config file
- Responsibility: dataset source, labels, benign generation, ML/LLM/hybrid settings, external datasets
- What to look for: held-out attacks, thresholds, model names, and external dataset definitions

### `requirements.txt`
- Why it matters: concrete dependency footprint
- Responsibility: Python package list
- What to look for: DVC, scikit-learn, OpenAI client, Hugging Face datasets, pytest

## Data Preparation

### `src/preprocess.py`
- Why it matters: first real data-shaping step
- Responsibility: load raw dataset, add hierarchical labels, build benign set, deduplicate, write full parquet
- What to look for: benign construction, synthetic benign integration, prompt hashing

### `src/build_splits.py`
- Why it matters: defines train/val/test/test_unseen semantics
- Responsibility: grouped splitting by `prompt_hash`, held-out attack routing
- What to look for: how unseen attacks are isolated and why groups are preserved

### `src/utils.py`
- Why it matters: shared paths and config loader
- Responsibility: root-relative directory constants, `build_sample_id`, config loading
- What to look for: canonical path layout under `data/processed/` and `reports/`

## ML Path

### `src/ml_classifier/ml_baseline.py`
- Why it matters: main classical ML implementation
- Responsibility: feature building, training, calibration, prediction, persistence
- What to look for: NLP filtering, char-level TF-IDF, handcrafted features, binary calibration

### `src/ml_classifier/utils.py`
- Why it matters: feature-engineering support
- Responsibility: text/Unicode feature extraction used by the ML model
- What to look for: what signals the model sees beyond TF-IDF

### `src/ml_classifier/constants.py`
- Why it matters: local classifier constants
- Responsibility: shared ML-related constants
- What to look for: any assumptions reused across ML modules

## LLM Path

### `src/llm_classifier/llm_classifier.py`
- Why it matters: most complex runtime file
- Responsibility: classifier/judge logic, prompting, concurrency, retries, usage stats, research output
- What to look for: `predict`, `predict_batch`, judge flow, few-shot construction, artifact schema

### `src/llm_classifier/prompts.py`
- Why it matters: prompt contract
- Responsibility: build classifier and judge messages
- What to look for: expected JSON output shape and prompt framing

### `src/llm_classifier/constants.py`
- Why it matters: taxonomy and heuristic support for LLM behavior
- Responsibility: NLP types and pattern constants
- What to look for: how categories are derived and special-cased

### `src/embeddings.py`
- Why it matters: dynamic few-shot retrieval support
- Responsibility: embedding API calls, exemplar bank, similarity selection
- What to look for: how benign/attack exemplar pairs are chosen

### `src/llm_cache.py`
- Why it matters: cost and rerun control
- Responsibility: local deterministic cache for chat completions
- What to look for: request-key normalization and atomic file writes

### `src/llm_provider.py`
- Why it matters: external API switching point
- Responsibility: choose NIM vs OpenAI, resolve model names, create clients
- What to look for: default provider behavior and model translation

## Hybrid / Evaluation / Reporting

### `src/hybrid_router.py`
- Why it matters: direct hybrid routing logic
- Responsibility: ML-first routing and escalation to LLM
- What to look for: fast-path conditions, abstain behavior, route reasons

### `src/research.py`
- Why it matters: artifact-based integration layer
- Responsibility: merge ML and LLM predictions, compute research hybrid outputs, diagnostics, reports
- What to look for: `compute_hybrid_routing`, strict-mode assumptions, routing diagnostics

### `src/evaluate.py`
- Why it matters: shared scoring contract
- Responsibility: binary/category/type/calibration metrics and report rendering
- What to look for: how uncertain/abstain are treated and how FPR views are computed

### `src/cli/eval_new.py`
- Why it matters: final report generator for tracked outputs
- Responsibility: render main and external reports from research artifacts
- What to look for: report scope differences between ML, LLM, and hybrid

### `src/eval_external.py`
- Why it matters: external dataset evaluation path
- Responsibility: load external HF datasets, map labels, run ML/hybrid evaluation
- What to look for: binary-only assumptions and label mapping logic

### `src/cli/research_external.py`
- Why it matters: external-dataset research pipeline glue
- Responsibility: build wide external research parquets and reports
- What to look for: strict requirement for precomputed LLM external artifacts in hybrid mode

## Inference / Developer UX

### `src/cli/predict.py`
- Why it matters: easiest interactive prediction entrypoint
- Responsibility: stdin/file input, dispatch to `ml`, `llm`, or `hybrid`
- What to look for: which saved artifacts must already exist

### `src/cli/infer_split.py`
- Why it matters: cheap ML report path
- Responsibility: run ML on a saved split and generate a Markdown report
- What to look for: scope breakdown between full, ML-scope, and NLP-only subsets

### `run_inference.sh`
- Why it matters: likely first command a new engineer will run
- Responsibility: shell wrapper for ML/LLM/hybrid inference modes
- What to look for: required preconditions and split/model checks

### `run_llm.sh`
- Why it matters: controls expensive DVC LLM stage execution
- Responsibility: unfreeze -> repro -> re-freeze
- What to look for: why the stage is normally frozen

### `run_synth.sh`
- Why it matters: entrypoint for synthetic benign generation
- Responsibility: invoke the synthetic generation CLI outside DVC
- What to look for: why it is intentionally not part of the normal pipeline

## Data Quality Support

### `src/synthetic_benign.py`
- Why it matters: expands benign coverage with instruction-like but safe prompts
- Responsibility: generate synthetic benign prompts across six categories
- What to look for: category definitions and caching behavior

### `src/validators.py`
- Why it matters: protects synthetic benign quality
- Responsibility: heuristic validation, judge validation, near-duplicate filtering
- What to look for: reject patterns and the three-layer validation design

## Tests Worth Reading Early

### `tests/test_hybrid_router.py`
- Why it matters: clarifies intended routing semantics
- Responsibility: tests fast-path, escalation, abstain, and confidence behavior
- What to look for: design assumptions that are easier to learn here than from prose

### `tests/test_llm_classifier.py`
- Why it matters: gives concrete expectations for classifier outputs
- Responsibility: tests usage stats, category derivation, few-shot construction, classify/judge logic
- What to look for: output schema and failure handling

### `tests/test_research.py`
- Why it matters: validates the artifact-merge path
- Responsibility: tests strict mode, hybrid routing, and research DataFrame shape
- What to look for: what fields downstream reporting expects

### `tests/test_research_external.py`
- Why it matters: clarifies external-dataset assumptions
- Responsibility: tests external artifact schema, report generation, and resume behavior
- What to look for: how external flows differ from the main dataset flow
