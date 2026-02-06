# Revised Project Plan — LLM Security Gatekeeper

## Context

The original plan proposed 20-class attack type classification across 7 steps. EDA revealed this is misguided: NLP-based attacks (8 types) are indistinguishable from each other (17.9% accuracy) because they all perform similar word-level substitutions. Meanwhile, Unicode-based attacks separate cleanly (88.5%). The dataset also contains only adversarial samples — no benign examples exist yet.

This revised plan restructures around what the data actually supports: a **hierarchical classifier** (binary → category → type) that focuses effort where it matters.

## Revised Label Scheme

```
Level 0: Binary       → adversarial | benign
Level 1: Category     → unicode_attack | nlp_attack  (94% stage-1 accuracy)
Level 2: Specific     → 12 unicode sub-types only    (88.5% accuracy)
                        NLP stays collapsed as "nlp_attack" (not worth sub-typing)
```

---

## Step 1 — Build Benign Set + Proper Splits ✅

**Why**: The dataset has ~11k adversarial samples but zero benign ones. Binary detection (the most useful real-world capability) requires benign examples.

**Approach**:
- Use the 623 unique `original_prompt` values as the benign seed set
- Augment with resampling to reach ~2k benign samples
- Grouped split by prompt hash (original + its modified variants stay together)
- Hold-out split: Emoji Smuggling + Pruthi reserved for unseen-attack generalization testing

**Results**:
- 13,313 total samples (11,313 adversarial + 2,000 benign)
- train: 8,562 / val: 1,926 / test: 1,999 / test_unseen: 826
- Zero prompt hash overlap between splits (verified)

**Deliverables**:
- `src/preprocess.py` — loading, normalization, benign set construction
- `src/build_splits.py` — grouped train/val/test splits
- `data/processed/` — processed parquet with hierarchical labels
- `configs/default.yaml` — split ratios, normalization settings

---

## Step 2 — LLM Classifier (Hierarchical, Few-Shot) ✅

**Why**: Establish the core LLM-as-a-classifier with the revised label scheme. Build on the two-stage approach from EDA that already showed promise.

**Approach**:
- Stage 0: Binary gate — adversarial vs benign
- Stage 1: Category — unicode vs nlp (94% baseline exists)
- Stage 2: Specific type — only for unicode attacks (88.5% baseline)
- Attack descriptions in system prompt
- Few-shot: 2 examples for unicode types, 5 for NLP types
- Structured JSON output with confidence scores
- Model: gpt-4o-mini

**Results** (100 test samples):
- Binary: 69% accuracy, 22% false-negative rate
- Category: 89.4% accuracy (41/44 unicode, 18/22 NLP correct)
- Unicode sub-types: 78% accuracy, 95% precision
- 225 LLM calls, ~197s total, ~0.87s avg per call

**Evaluation**:
- Binary: precision, recall, F1 (especially false-negative rate — missed attacks)
- Category: accuracy, confusion matrix
- Per-type: macro F1, per-class breakdown (unicode types only)
- Calibration: confidence vs accuracy buckets
- Cost tracking: tokens in/out, latency per sample

**Deliverables**:
- `src/llm_classifier.py` — hierarchical classifier with configurable stages
- `src/evaluate.py` — metrics computation + reporting
- `reports/eval_report_llm.md`

---

## Step 3 — ML Baseline + Hybrid Router ✅

**Why**: ML models are cheap and fast. Character-level features (n-grams, unicode stats) are extremely effective at the binary gate, since adversarial samples have distinctive character patterns (homoglyphs, zero-width chars, unusual unicode ranges).

### ML Baseline

**Features**: char n-grams (2-5) TF-IDF, unicode category distribution, non-ASCII ratio, zero-width char count, BiDi char count, tag char count, fullwidth count, combining char ratio, entropy, unique scripts.

**Model**: Logistic regression at each hierarchy level.

**Results** (test set):
- Binary: 86% accuracy (adversarial recall 95%, benign recall 49%)
- Category: 85% accuracy (unicode 100%, NLP 96%)
- Type: 81% accuracy (unicode sub-types 93-100%)
- Weakness: benign detection poor (seed set = resampled original prompts, not diverse enough)

### Hybrid Router

**Routing**: ML binary gate → if confidence < 0.85 → escalate to LLM → if LLM confidence < 0.7 → abstain.

**Results** (100 test samples, threshold 0.85):
- 60% handled by ML alone (instant, free)
- 40% escalated to LLM (75 calls vs 225 for LLM-only = 67% reduction)
- Binary: 79% accuracy, 9.4% false-negative rate
- Category: 96.1% accuracy (best of any approach)
- Unicode sub-types: 100% accuracy
- Total latency: 73s (vs 197s for LLM-only)

**Threshold sweep** (ML-only, no LLM):
| Threshold | ML-handled | ML accuracy on handled |
|-----------|------------|----------------------|
| 0.50 | 100% | 85.0% |
| 0.70 | 77% | 92.2% |
| 0.85 | 60% | 98.3% |
| 0.95 | 50% | 100% |
| 0.99 | 48% | 100% |

**Deliverables**:
- `src/ml_baseline.py` — feature extraction + model training
- `src/hybrid_router.py` — routing logic with configurable thresholds
- `reports/eval_report_hybrid.md`

---

## Step 4 — Dynamic Few-Shot + Error Analysis

**Why**: Static few-shot leaves performance on the table. Dynamic selection per attack family should help, especially for the unicode sub-types where there's real signal.

**Approach**:
- Build a few-shot exemplar bank with embeddings
- Selection: retrieve most similar examples from the same category
- Include hard negatives (confusable cross-category pairs)
- Ablation: static vs dynamic few-shot

**Error analysis** (research value):
- Characterize failure modes: where does the LLM fail and why?
- Confusion clusters within unicode types
- False negatives on the binary gate (missed attacks)
- NLP attack samples that look most benign

**Deliverables**:
- Dynamic few-shot logic integrated into `src/llm_classifier.py`
- `reports/error_analysis.md`

---

## Step 5 — Reporting + Polish

**Deliverables**:
- Final evaluation across all approaches with comparison tables
- `src/predict.py` CLI — reads input, outputs predictions + metadata ✅
- `requirements.txt` ✅
- Clean notebooks showing the research narrative

---

## Comparison Summary

| Metric | ML-only | LLM-only | Hybrid |
|--------|---------|----------|--------|
| Binary accuracy | **86%** | 69% | 79% |
| False-negative rate | **5%** | 22% | 9.4% |
| Category accuracy | 85% | 89% | **96.1%** |
| Unicode type accuracy | 93-100% | 78% | **100%** |
| LLM calls per 100 samples | 0 | 225 | **75** |
| Latency (100 samples) | <1s | 197s | **73s** |

---

## Verification Plan

1. ✅ Run `src/preprocess.py` → verified processed parquet has correct schema, benign + adversarial labels, no leakage across splits
2. ✅ Run `src/llm_classifier.py` on test split → verified hierarchical predictions with metrics
3. ✅ Run `src/ml_baseline.py` → verified ML baseline trains and evaluates
4. ✅ Run `src/hybrid_router.py` → verified hybrid improves cost-efficiency vs LLM-only
5. ✅ Run `src/predict.py` on sample input → verified end-to-end prediction pipeline
