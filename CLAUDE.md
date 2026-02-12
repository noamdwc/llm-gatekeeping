# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- **Python environment**: `/Users/noamc/miniconda3/envs/llm_gate/bin/python` (Python 3.14)
- **Activate**: `conda activate llm_gate`
- **Dependencies**: `pip install -r requirements.txt`
- **OpenAI API key**: Required for LLM and hybrid modes; stored in `.env` (gitignored)
- **HuggingFace auth**: `huggingface-cli login` (dataset requires access approval)
- **Experiment tracking**: wandb (optional, disable with `--no-wandb`)

## Pipeline Commands

All modules run as `python -m src.<module>` from the project root:

```bash
python -m src.preprocess          # Load dataset, build benign set → data/processed/full_dataset.parquet
python -m src.build_splits        # Grouped splits → train/val/test/test_unseen.parquet
python -m src.ml_baseline         # Train ML model → data/processed/ml_baseline.pkl
python -m src.llm_classifier --split test --limit 100    # Run LLM classifier (costs API tokens)
python -m src.hybrid_router --limit 100                  # Run hybrid router (costs API tokens)
python -m src.evaluate --predictions data/processed/predictions_test.csv
```

Prediction CLI:
```bash
echo "text" | python -m src.predict --mode ml --pretty      # ML-only (instant, no API)
echo "text" | python -m src.predict --mode llm --pretty     # LLM-only
echo "text" | python -m src.predict --mode hybrid --pretty  # Hybrid (recommended)
```

Dynamic few-shot: add `--dynamic` flag to `src.llm_classifier`. Builds/loads an exemplar bank at `data/processed/exemplar_bank.pkl`.

## Architecture

Three-level hierarchical classifier for adversarial prompt detection:

```
Binary (adversarial/benign) → Category (unicode_attack/nlp_attack) → Type (12 unicode sub-types)
```

NLP attack sub-types are collapsed to a single "nlp_attack" label (sub-types are indistinguishable at 17.9% accuracy). Unicode sub-types classify at 93-100%.

Three classifier backends share this hierarchy:
- **ML** (`ml_baseline.py`): Char n-gram TF-IDF + handcrafted Unicode features → LogisticRegression per level. `MLBaseline` class handles fit/predict/save/load.
- **LLM** (`llm_classifier.py`): Three-stage OpenAI chat calls with JSON mode. `HierarchicalLLMClassifier` runs stages sequentially — benign samples skip stages 1-2. Supports static and dynamic few-shot (via `embeddings.py` ExemplarBank).
- **Hybrid** (`hybrid_router.py`): ML runs first on all samples; low-confidence ones (below `ml_confidence_threshold` in config) escalate to LLM. `HybridRouter` wraps both.

## Key Design Decisions

- **Data splits are grouped by `prompt_hash`** (MD5 of lowered/stripped original prompt) — all variants of the same original prompt stay in the same split. No overlap between train/val/test.
- **Held-out attacks** (Emoji Smuggling + Pruthi) are entirely excluded from train/val/test and placed in `test_unseen.parquet` for generalization testing.
- **Config-driven**: All labels, thresholds, model params, and split ratios live in `configs/default.yaml`.
- scikit-learn 1.8: `LogisticRegression` no longer accepts `multi_class` parameter.

## Data Flow

```
HuggingFace dataset → preprocess.py → full_dataset.parquet → build_splits.py → {train,val,test,test_unseen}.parquet
                                                                                      ↓
                                                              ml_baseline.py → ml_baseline.pkl
                                                              llm_classifier.py → predictions CSV
                                                              hybrid_router.py → eval report
```

Evaluation (`evaluate.py`) computes binary/category/type metrics plus calibration. Used both as CLI (`--predictions`) and programmatically via `evaluate_dataframe()`.
