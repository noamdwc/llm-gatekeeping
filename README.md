# LLM Security Gatekeeper

Hierarchical classifier for detecting adversarial prompt injection and jailbreak attacks. Uses a three-level classification scheme (binary → category → type) with ML, LLM, and hybrid routing approaches.

Built on the [Mindgard evaded prompt injection dataset](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples) (~11.3k adversarial samples across 20 attack types + ~2k synthetic benign samples).

## Classification Hierarchy

```
Level 0: Binary     →  adversarial | benign
Level 1: Category   →  unicode_attack | nlp_attack
Level 2: Type       →  12 unicode sub-types (NLP collapsed — sub-types indistinguishable)
```

NLP-based attacks (TextFooler, BERT-Attack, BAE, etc.) all perform word-level substitutions and are not separable from each other (17.9% sub-type accuracy). Unicode-based attacks (homoglyphs, zero-width chars, diacritics, etc.) classify cleanly at 88-100%.

## Results

Metrics change based on split, sample limit, thresholds, and whether LLM stages were run.
The canonical latest outputs are:

- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_hybrid.md`
- `reports/research/eval_report_llm.md`
- `reports/research_external/research_external_<dataset>.md`

## Setup

```bash
# Create conda environment
conda create -n llm_gate python=3.14
conda activate llm_gate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (required for LLM and hybrid modes)
echo "OPENAI_API_KEY=sk-..." > .env

# Authenticate with HuggingFace (dataset requires access approval)
huggingface-cli login
```

## Two Pipeline Modes

### Research pipeline (reproducible, DVC)

This is the canonical way to run the project. It produces **research parquets** with all intermediate probabilities plus **reports**.

```bash
# ML-only research run (LLM stage frozen by default)
dvc repro

# Full research run (unfreezes LLM stage, runs, re-freezes)
./run_llm.sh
```

#### DVC runbook (what runs, what gets recomputed)

- **Stage graph**:
  - `preprocess` → `build_splits` → `ml_model` → `research`
  - `llm_classifier` is **frozen by default** (API cost); when unfrozen it produces `data/processed/predictions/llm_predictions_{split}.parquet` and `research` will include LLM metrics/report.
  - `research_external@{dataset}` stages run via DVC `foreach` and are **independent** per dataset.

- **Run a single stage**:

```bash
dvc repro ml_model
dvc repro research
dvc repro research_external@deepset
```

- **Add a new external dataset (no recompute for existing ones)**:
  - Add the dataset config under `external_datasets` in `configs/default.yaml`
  - Add the dataset key to the `foreach` list under `research_external` in `dvc.yaml`
  - Run:

```bash
dvc repro research_external@new_dataset
```

  DVC will compute **only** the new stage; existing `research_external@...` outputs remain cached.

- **What triggers recomputation** (high level):
  - Changing `configs/default.yaml:dataset|labels|benign` → recomputes `preprocess` and everything downstream.
  - Changing `configs/default.yaml:splits|labels.held_out_attacks` → recomputes `build_splits` and everything downstream.
  - Changing `configs/default.yaml:ml` → recomputes `ml_model` and all downstream stages.
  - Changing `configs/default.yaml:hybrid.ml_confidence_threshold` → recomputes `research` and all `research_external@{dataset}` stages, but **not** `ml_model`.
  - Changing one external dataset config `external_datasets.<key>` → recomputes **only** `research_external@<key>`.
  - Adding a new dataset key → recomputes **only** the new `research_external@<key>` stage.

### Inference pipeline (lightweight, bash)

Fast CLI for running just what you need and producing reports. Assumes you already have splits + a trained ML model (from a prior `dvc repro`).

```bash
./run_inference.sh --mode ml --split test
./run_inference.sh --mode hybrid --split test --limit 100
./run_inference.sh --mode llm --split test --limit 50
./run_inference.sh --mode ml --split test_unseen
```

## Pipeline (module-level commands)

Run the full pipeline step by step:

```bash
# 1. Preprocess: load dataset, build benign set, add hierarchical labels
python -m src.preprocess

# 2. Build splits: grouped by prompt hash, held-out attack types
python -m src.build_splits

# 3. Train ML baseline + write research prediction parquets
python -m src.ml_classifier.ml_baseline --research

# 4. Run LLM classifier (requires OpenAI API key)
python -m src.llm_classifier.llm_classifier --split test --limit 100 --research

# 5. Merge predictions + hybrid routing + reports (research stage)
python -m src.research --split test

# 6. Hybrid router evaluation (optional, API tokens)
python -m src.hybrid_router --limit 100
```

### External datasets (additive + cached with DVC)

External datasets are configured in `configs/default.yaml` under `external_datasets`. DVC runs them as independent `foreach` stages (e.g. `research_external@deepset`) so adding a new dataset only computes the new stage.

To add a dataset:
- Add its config under `external_datasets` in `configs/default.yaml`
- Add the key to the `foreach` list in `dvc.yaml`
- Run `dvc repro` (only the new dataset stage runs; existing ones are cached)

## Prediction CLI

```bash
# ML-only (no API calls, instant)
echo "some suspicious text" | python -m src.cli.predict --mode ml --pretty

# LLM-only (requires API key)
echo "some suspicious text" | python -m src.cli.predict --mode llm --pretty

# Hybrid: ML first, escalate to LLM if uncertain (recommended)
echo "some suspicious text" | python -m src.cli.predict --mode hybrid --pretty

# From file (one text per line)
python -m src.cli.predict --mode ml --input texts.txt --pretty
```

Output:
```json
{
  "text": "some suspicious text...",
  "label_binary": "adversarial",
  "label_category": "unicode_attack",
  "label_type": "Homoglyphs",
  "confidence_binary": 0.95,
  "confidence_category": 0.92,
  "confidence_type": 0.88,
  "routed_to": "ml"
}
```

## Experiment Tracking

All training and evaluation scripts support [Weights & Biases](https://wandb.ai/) logging:

```bash
# Login to wandb
wandb login

# Runs log automatically; disable with --no-wandb
python -m src.ml_classifier.ml_baseline --no-wandb
python -m src.llm_classifier.llm_classifier --no-wandb
python -m src.hybrid_router --no-wandb
```

Tracked metrics include per-level accuracy/F1, LLM token usage, latency, routing stats, and threshold sweep results. Model artifacts are saved as wandb Artifacts.

## Project Structure

```
configs/default.yaml        # All configuration (labels, splits, thresholds)
src/
  preprocess.py             # Dataset loading + benign set construction
  build_splits.py           # Grouped train/val/test splits
  ml_classifier/ml_baseline.py      # Character-level ML classifier
  llm_classifier/llm_classifier.py  # Classifier + judge LLM classifier (binary + category)
  hybrid_router.py          # ML gate + LLM escalation
  evaluate.py               # Metrics at all hierarchy levels
  predict.py                # CLI prediction tool
data/processed/
  full_dataset.parquet      # Combined adversarial + benign
  splits/                   # Splits (no prompt hash overlap)
    train.parquet
    val.parquet
    test.parquet
    test_unseen.parquet
  models/
    ml_baseline.pkl          # Trained ML model
  predictions/
    ml_predictions_*.parquet # ML prediction parquets (research)
    llm_predictions_*.parquet# LLM prediction parquets (research)
  research/
    research_*.parquet       # Wide research parquets (merged)
  research_external/
    research_external_*.parquet  # Per-external-dataset research parquets
reports/
  research/                 # Main-dataset evaluation reports
  research_external/        # Per-external-dataset reports
```

## ML Features

The ML baseline extracts character-level features that are highly discriminative for Unicode-based attacks:

- **TF-IDF char n-grams** (2-5 chars, `char_wb` analyzer)
- **Unicode category distribution** (Lu, Ll, Mn, Cf, So ratios)
- **Non-ASCII ratio**
- **Zero-width / BiDi / tag / fullwidth / combining character counts**
- **Character entropy**
- **Unique script count**
