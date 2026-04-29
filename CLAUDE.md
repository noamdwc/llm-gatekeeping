# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- **Python environment**: `/Users/noamc/miniconda3/envs/llm_gate/bin/python` (Python 3.14)
- **Activate**: `conda activate llm_gate`
- **Dependencies**: `pip install -r requirements.txt`
- **LLM provider**: Set `LLM_PROVIDER` env var to `nim` (default) or `openai`. NIM model names in config are auto-translated to OpenAI equivalents.
- **NVIDIA API key**: Required when `LLM_PROVIDER=nim` (default); stored in `.env` as `NVIDIA_API_KEY` (gitignored). Get one at https://integrate.api.nvidia.com
- **OpenAI API key**: Required when `LLM_PROVIDER=openai`; stored in `.env` as `OPENAI_API_KEY` (gitignored).
- **Provider switch**: `./run_llm_provider_refresh.sh [--provider nim|openai]` re-runs LLM-dependent DVC stages (DVC does not track `LLM_PROVIDER`).
- **HuggingFace auth**: `huggingface-cli login` (dataset requires access approval)
- **Experiment tracking**: wandb (optional, disable with `--no-wandb`)

## Two Pipeline Modes

### 1. Research Pipeline (DVC)

Heavy, reproducible run via `dvc repro`. Each stage produces output files consumed by downstream stages (no redundant computation). Produces research parquets with all intermediate probabilities + evaluation reports.

```bash
dvc repro                    # Run full pipeline (all stages including LLM)
```

All stages run by default, including LLM classifier and judge. DVC caches predictions — changing thresholds (in `hybrid` config) only re-runs research/eval stages, not LLM API calls.

DVC stages (in dependency order):
- `generate_synthetic_benign@{A..F}` (foreach; one-time, API tokens; only consumed if `benign.synthetic.enabled=true`)
- `preprocess → build_splits`
- training: `ml_model`, `deberta_model`, `llm_classifier`, `llm_classifier_val`
- routing/risk: `research`, `research_val`, `train_risk_model` (consumed by `research`/`research_val`), `risk_model` (post-hoc evaluation)
- external: `research_external_llm@{ds} → research_external@{ds} → eval_new_external@{ds}`
- reports: `eval_new`

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
python -m src.cli.deberta_classifier --research --no-wandb  # Train DeBERTa + evaluate on all splits
python -m src.cli.deberta_classifier --train-only                  # Train only, skip prediction
python -m src.cli.deberta_classifier --predict-only                # Predict only from saved model
python -m src.llm_classifier.llm_classifier --split test --research  # LLM classifier (API tokens)
python -m src.llm_classifier.llm_classifier --split test --research --dynamic  # LLM with dynamic few-shot
python -m src.research --split test                         # Merge predictions + hybrid routing + margin trace
python -m src.research --split val                          # Margin trace on val (input to risk model training)
python -m src.cli.train_risk_model                                 # Fit risk_model.pkl from val trace + DeBERTa val preds
python -m src.cli.benign_risk_model --train-trace data/processed/research/hybrid_margin_trace_val.parquet --trace data/processed/research/hybrid_margin_trace_test.parquet --split test  # Post-hoc evaluation
python -m src.cli.eval_new --split test --config configs/default.yaml  # Canonical markdown reports
python -m src.cli.research_external --dataset deepset             # External dataset research (SKIP_LLM=1 by default)
SKIP_LLM=0 python -m src.cli.research_external --dataset deepset  # Include LLM predictions
python -m src.cli.run_baseline --baseline sentinel_v2 --split test      # Run HF baseline on internal split
python -m src.cli.run_baseline --baseline all --external all            # Run all HF baselines on all external sets
python -m src.cli.eval_baselines                                         # Compare HF baselines vs ML/hybrid
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
  synthetic_benign/              # generate_synthetic_benign@{A..F}
    synthetic_benign_{A..F}.parquet
  splits/                        # build_splits
    train.parquet, val.parquet, test.parquet, test_unseen.parquet
  models/
    ml_baseline.pkl              # ml_model
    risk_model.pkl               # train_risk_model
  predictions/                   # ml_model + deberta_model + llm_classifier(_val)
    ml_predictions_{split}.parquet
    deberta_predictions_{split}.parquet
    llm_predictions_{split}.parquet
  predictions_external/          # research_external_llm@{dataset}
    llm_predictions_external_{dataset}.parquet
  research/                      # research / research_val / risk_model
    research_{split}.parquet
    hybrid_margin_trace_{split}.parquet
    posthoc_benign_risk_predictions.parquet
    posthoc_benign_risk_summary.csv
  research_external/             # research_external@{dataset}
    research_external_{dataset}.parquet
  baselines/                     # HF baseline predictions
    {baseline_key}_{dataset_key}.parquet

artifacts/
  deberta_classifier/            # deberta_model stage
    model/, tokenizer/, label_mapping.json, train_history.json

reports/
  deberta_classifier/            # deberta_model stage
    metrics.json, classification_report.json, summary.md
  research/                      # eval_new stage
    eval_report_ml.md, eval_report_hybrid.md, eval_report_llm.md, summary_report.md
  research_external/             # eval_new_external@{dataset}
    research_external_{dataset}.md
  baselines/
    comparison_report.md
  posthoc_benign_risk_model.md   # risk_model stage
  artifacts/                     # risk_model stage
    benign_risk_roc.png, benign_risk_pr.png, benign_risk_calibration.png
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

Classifier backends sharing this hierarchy:
- **ML** (`ml_classifier/ml_baseline.py`): Char n-gram TF-IDF + handcrafted Unicode features → LogisticRegression per level. `MLBaseline` class handles fit/predict/save/load.
- **DeBERTa** (`cli/deberta_classifier.py`, `models/`): Per-level fine-tuning of `microsoft/deberta-v3-base`. Strong neural baseline; produces `deberta_predictions_{split}.parquet` and consumed by the risk model.
- **LLM** (`llm_classifier/llm_classifier.py`): Classifier + conditional judge chat calls with JSON mode (NIM by default; OpenAI when `LLM_PROVIDER=openai`). `HierarchicalLLMClassifier` predicts binary + derived category and can invoke judge on low-confidence cases. Supports static and dynamic few-shot (via `embeddings.py` ExemplarBank). Hard benign examples controlled by `llm.few_shot.include_hard_benign` config flag (default: false).
- **Hybrid** (`hybrid_router.py`): ML runs first on all samples; low-confidence ones (below `ml_confidence_threshold` in config) escalate to LLM. `HybridRouter` wraps both.
- **Benign risk model** (`benign_risk_model.py` + `cli/train_risk_model.py` + `cli/benign_risk_model.py`): Post-hoc LogisticRegression trained on the hybrid val margin trace + DeBERTa val probabilities. Used by `research`/`research_val` to flag false-positive-prone benigns; `risk_model` stage produces ROC/PR/calibration plots and `reports/posthoc_benign_risk_model.md`.
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
