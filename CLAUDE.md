# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- **Python environment**: `/Users/noamc/miniconda3/envs/llm_gate/bin/python` (Python 3.14)
- **Activate**: `conda activate llm_gate`
- **Dependencies**: `pip install -r requirements.txt`
- **NVIDIA API key**: Required for LLM and hybrid modes; stored in `.env` as `NVIDIA_API_KEY` (gitignored). Get one at https://integrate.api.nvidia.com
- **HuggingFace auth**: `huggingface-cli login` (dataset requires access approval)
- **Experiment tracking**: wandb (optional, disable with `--no-wandb`)

## Two Pipeline Modes

### 1. Research Pipeline (DVC)

Heavy, reproducible run via `dvc repro`. Each stage produces output files consumed by downstream stages (no redundant computation). Produces research parquets with all intermediate probabilities + evaluation reports.

```bash
dvc repro                    # Run full pipeline (ML-only; llm_classifier frozen, SKIP_LLM=1 by default)
./run_llm.sh                 # Unfreeze LLM stages + set SKIP_LLM=0, run full pipeline, re-freeze
```

**LLM control**: The `llm_classifier` DVC stage is frozen (unfreeze via `run_llm.sh`). For `research_external` stages, the `SKIP_LLM` env var controls LLM (defaults to `"1"` = skip); `run_llm.sh` sets `SKIP_LLM=0`.

DVC stages: `preprocess → build_splits → ml_model → llm_classifier (frozen) → research → research_external@{dataset}`

### 2. Inference Pipeline (Bash)

Lightweight, fast run for quick evaluation. No DVC overhead. Assumes model + splits already exist from a prior `dvc repro`.

```bash
./run_inference.sh --mode ml --split test                  # ML-only, instant
./run_inference.sh --mode hybrid --split test --limit 100  # Hybrid (API tokens)
./run_inference.sh --mode llm --split test --limit 50      # LLM-only (API tokens)
./run_inference.sh --mode ml --split test_unseen           # Generalization test
```

## Pipeline Module Commands

All modules run as `python -m src.<module>` from the project root:

```bash
python -m src.preprocess                                    # Load dataset → data/processed/full_dataset.parquet
python -m src.build_splits                                  # Grouped splits → data/processed/splits/*.parquet
python -m src.ml_classifier.ml_baseline --research          # Train ML + save research predictions
python -m src.llm_classifier.llm_classifier --split test --research  # LLM classifier (API tokens)
python -m src.llm_classifier.llm_classifier --split test --research --dynamic  # LLM with dynamic few-shot
python -m src.research --split test                         # Merge predictions + hybrid routing + reports
python -m src.cli.research_external --dataset deepset             # External dataset research (SKIP_LLM=1 by default)
SKIP_LLM=0 python -m src.cli.research_external --dataset deepset  # Include LLM predictions
python -m src.cli.generate_synthetic_benign --category all --limit 100  # Generate synthetic benign prompts
```

Prediction CLI:
```bash
echo "text" | python -m src.cli.predict --mode ml --pretty      # ML-only (instant, no API)
echo "text" | python -m src.cli.predict --mode llm --pretty     # LLM-only
echo "text" | python -m src.cli.predict --mode hybrid --pretty  # Hybrid (recommended)
```

## Output Directory Structure

```
data/processed/
  full_dataset.parquet           # preprocess
  splits/                        # build_splits
    train.parquet, val.parquet, test.parquet, test_unseen.parquet
  models/                        # ml_model
    ml_baseline.pkl
  predictions/                   # ml_model + llm_classifier
    ml_predictions_{split}.parquet
    llm_predictions_{split}.parquet
  research/                      # research stage
    research_{split}.parquet
  research_external/             # research_external@{dataset}
    research_external_{dataset}.parquet

reports/
  research/                      # research stage
    eval_report_ml.md, eval_report_hybrid.md, eval_report_llm.md
  research_external/             # research_external@{dataset}
    research_external_{dataset}.md
```

## Adding a New External Dataset

1. Add config to `configs/default.yaml` under `external_datasets`
2. Add the key to the `foreach` list in `dvc.yaml`
3. Run `dvc repro` — only the new dataset stage runs; existing datasets are cached

## Architecture

Three-level hierarchical classifier for adversarial prompt detection:

```
Binary (adversarial/benign) → Category (unicode_attack/nlp_attack) → Type (12 unicode sub-types)
```

NLP attack sub-types are collapsed to a single "nlp_attack" label (sub-types are indistinguishable at 17.9% accuracy). Unicode sub-types classify at 93-100%.

Three classifier backends share this hierarchy:
- **ML** (`ml_classifier/ml_baseline.py`): Char n-gram TF-IDF + handcrafted Unicode features → LogisticRegression per level. `MLBaseline` class handles fit/predict/save/load.
- **LLM** (`llm_classifier/llm_classifier.py`): Classifier + conditional judge NVIDIA NIM chat calls with JSON mode. `HierarchicalLLMClassifier` predicts binary + derived category and can invoke judge on low-confidence cases. Supports static and dynamic few-shot (via `embeddings.py` ExemplarBank). Hard benign examples controlled by `llm.few_shot.include_hard_benign` config flag (default: false).
- **Hybrid** (`hybrid_router.py`): ML runs first on all samples; low-confidence ones (below `ml_confidence_threshold` in config) escalate to LLM. `HybridRouter` wraps both.
- **Synthetic benign** (`synthetic_benign.py` + `validators.py`): `SyntheticBenignGenerator` produces diverse benign prompts across 6 categories (A–F) via LLM. Three-layer validation: `HeuristicBenignValidator` (regex), `JudgeBenignValidator` (LLM), `DeduplicateFilter` (embedding cosine similarity > 0.95). Enable in config with `benign.synthetic.enabled: true`; generate first with `python -m src.cli.generate_synthetic_benign`.

## Key Design Decisions

- **Data splits are grouped by `prompt_hash`** (MD5 of lowered/stripped original prompt) — all variants of the same original prompt stay in the same split. No overlap between train/val/test.
- **Held-out attacks** (Emoji Smuggling + Pruthi) are entirely excluded from train/val/test and placed in `test_unseen.parquet` for generalization testing.
- **Config-driven**: All labels, thresholds, model params, and split ratios live in `configs/default.yaml`.
- **Path constants**: All output paths are defined in `src/utils.py` (SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR, etc.).
- scikit-learn 1.8: `LogisticRegression` no longer accepts `multi_class` parameter.

## Data Flow

```
HuggingFace dataset → preprocess → full_dataset.parquet → build_splits → splits/*.parquet
                                                                              ↓
                                                        ml_model → models/ml_baseline.pkl
                                                                 → predictions/ml_predictions_*.parquet
                                                        llm_classifier → predictions/llm_predictions_*.parquet
                                                                              ↓
                                                        research → research/research_*.parquet
                                                                 → reports/research/*.md
                                                        research_external@{ds} → research_external/*.parquet
                                                                               → reports/research_external/*.md
```

Evaluation (`evaluate.py`) computes binary/category/type metrics plus calibration. Used both as CLI (`--predictions`) and programmatically via `evaluate_dataframe()`.
