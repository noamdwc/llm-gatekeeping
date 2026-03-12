# Architecture

## High-Level View
The repo is organized around a staged research pipeline:
- ingest and label data
- build grouped splits
- train an ML baseline
- optionally run an LLM classifier
- merge outputs into hybrid research artifacts
- generate Markdown evaluation reports

The main architectural components are:
- data preparation: `src/preprocess.py`, `src/build_splits.py`
- ML classification: `src/ml_classifier/`
- LLM classification and prompting: `src/llm_classifier/`, `src/embeddings.py`, `src/llm_cache.py`, `src/llm_provider.py`
- hybrid decision logic: `src/hybrid_router.py`, `src/research.py`
- evaluation and reporting: `src/evaluate.py`, `src/cli/eval_new.py`, `src/cli/research_external.py`, `src/eval_external.py`
- orchestration: `dvc.yaml`, `run_*.sh`

## Text Diagram
```text
Hugging Face datasets
        |
        v
src/preprocess.py
  - add hierarchy labels
  - build benign set
  - optionally merge synthetic benign
        |
        v
data/processed/full_dataset.parquet
        |
        v
src/build_splits.py
  - grouped train/val/test
  - held-out attacks -> test_unseen
        |
        +-----------------------------+
        |                             |
        v                             v
src/ml_classifier/ml_baseline.py   src/llm_classifier/llm_classifier.py
  - char TF-IDF + features           - classifier + judge LLM flow
  - train/save model                 - few-shot prompting
  - predict research parquets        - optional dynamic exemplars
        |                             - cache API responses
        +--------------+--------------+
                       |
                       v
                 src/research.py
           - merge prediction artifacts
           - compute hybrid outputs
           - compute routing diagnostics
                       |
                       v
         data/processed/research/*.parquet
                       |
                       v
              src/cli/eval_new.py
           - write Markdown reports
```

## Data Flow
- Raw dataset loading happens in `src/preprocess.py` through `datasets.load_dataset(...)`.
- The combined labeled dataset is saved to `data/processed/full_dataset.parquet`.
- Split creation in `src/build_splits.py` uses `prompt_hash` to keep prompt families together.
- ML and LLM stages each write prediction artifacts under `data/processed/predictions/`.
- `src/research.py` joins those artifacts by `sample_id` and computes hybrid outputs.
- Report generation turns research parquets into Markdown under `reports/research/` and `reports/research_external/`.

## Storage Layers
- Config: `configs/default.yaml`
- Source data: remote Hugging Face datasets
- Local working artifacts: `data/processed/`
- Generated reports: `reports/`
- Cached LLM completions: `.cache/llm/`
- DVC metadata and cache control: `.dvc/`, `dvc.yaml`, `dvc.lock`

## External Integrations
- Hugging Face datasets for training and external evaluation
- NVIDIA NIM or OpenAI chat/embedding APIs, selected by `LLM_PROVIDER`
- Weights & Biases for run logging when enabled

## Boundaries Between Concerns
- `src/preprocess.py` and `src/build_splits.py` own dataset shaping
- `src/ml_classifier/` owns local classical ML behavior
- `src/llm_classifier/` owns prompting, judge logic, and provider-facing classification
- `src/hybrid_router.py` owns live routing decisions
- `src/research.py` owns batch merge logic and research-mode hybrid evaluation
- `src/evaluate.py` owns metric computation, not model execution

## What Seems To Be The Architectural Center Of Gravity
The center of gravity is the DVC research pipeline defined in `dvc.yaml`, with `configs/default.yaml` and `src/research.py` as the main integration points. The repo’s modules exist largely to produce reproducible artifacts and reports rather than to serve a network API.

## What Is Likely To Be Confusing For A New Engineer
- There are two hybrid implementations with overlapping intent:
  - `src/hybrid_router.py` for direct runtime routing
  - `src/research.py` for artifact-based hybrid recomputation
- “Inference” and “research” are related but not identical paths.
- The ML model is intentionally out of scope for NLP attacks, which can look like a bug if you miss the design intent in `src/ml_classifier/ml_baseline.py`.
- LLM behavior depends on both config and environment variables, plus local cache state in `.cache/llm/`.
- DVC freezes the LLM stage by default, so a normal `dvc repro` is not a fully fresh end-to-end LLM run.
