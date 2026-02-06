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

| Metric | ML-only | LLM-only | Hybrid |
|--------|---------|----------|--------|
| Binary accuracy | **86%** | 69% | 79% |
| False-negative rate | **5%** | 22% | 9.4% |
| Category accuracy | 85% | 89% | **96.1%** |
| Unicode type accuracy | 93-100% | 78% | **100%** |
| LLM calls / 100 samples | 0 | 225 | **75** |

The hybrid router gets the best of both worlds: ML handles 60% of traffic instantly and for free, while uncertain samples are escalated to the LLM for detailed classification.

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

## Pipeline

Run the full pipeline step by step:

```bash
# 1. Preprocess: load dataset, build benign set, add hierarchical labels
python -m src.preprocess

# 2. Build splits: grouped by prompt hash, held-out attack types
python -m src.build_splits

# 3. Train ML baseline: char n-gram TF-IDF + unicode features + logistic regression
python -m src.ml_baseline

# 4. Run LLM classifier (requires OpenAI API key)
python -m src.llm_classifier --split test --limit 100

# 5. Run hybrid router (requires OpenAI API key + trained ML model)
python -m src.hybrid_router --limit 100
```

## Prediction CLI

```bash
# ML-only (no API calls, instant)
echo "some suspicious text" | python -m src.predict --mode ml --pretty

# LLM-only (requires API key)
echo "some suspicious text" | python -m src.predict --mode llm --pretty

# Hybrid: ML first, escalate to LLM if uncertain (recommended)
echo "some suspicious text" | python -m src.predict --mode hybrid --pretty

# From file (one text per line)
python -m src.predict --mode ml --input texts.txt --pretty
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
python -m src.ml_baseline --no-wandb
python -m src.llm_classifier --no-wandb
python -m src.hybrid_router --no-wandb
```

Tracked metrics include per-level accuracy/F1, LLM token usage, latency, routing stats, and threshold sweep results. Model artifacts are saved as wandb Artifacts.

## Project Structure

```
configs/default.yaml        # All configuration (labels, splits, thresholds)
src/
  preprocess.py             # Dataset loading + benign set construction
  build_splits.py           # Grouped train/val/test splits
  ml_baseline.py            # Character-level ML classifier
  llm_classifier.py         # 3-stage hierarchical LLM classifier
  hybrid_router.py          # ML gate + LLM escalation
  evaluate.py               # Metrics at all hierarchy levels
  predict.py                # CLI prediction tool
data/processed/
  full_dataset.parquet      # Combined adversarial + benign
  train/val/test.parquet    # Splits (no prompt hash overlap)
  test_unseen.parquet       # Held-out attack types
  ml_baseline.pkl           # Trained ML model
reports/
  eval_report_llm.md        # LLM classifier evaluation
  eval_report_hybrid.md     # Hybrid router evaluation
```

## ML Features

The ML baseline extracts character-level features that are highly discriminative for Unicode-based attacks:

- **TF-IDF char n-grams** (2-5 chars, `char_wb` analyzer)
- **Unicode category distribution** (Lu, Ll, Mn, Cf, So ratios)
- **Non-ASCII ratio**
- **Zero-width / BiDi / tag / fullwidth / combining character counts**
- **Character entropy**
- **Unique script count**
