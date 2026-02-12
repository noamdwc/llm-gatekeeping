# Pipeline Breakdown: Hierarchical Adversarial Prompt Detection

## Executive Summary

This document provides a detailed research-oriented walkthrough of the LLM Security Gatekeeper pipeline — a hierarchical classification system for detecting adversarial prompt injection and jailbreak attacks. The pipeline combines a fast character-level ML classifier with a multi-stage LLM classifier through a confidence-based routing mechanism. Every design decision is explained in terms of the research motivation behind it.

The core research question: **How can we detect and classify adversarial prompts that have been deliberately modified to evade safety filters, while balancing accuracy, latency, and cost?**

---

## Table of Contents

1. [Problem Definition and Dataset](#1-problem-definition-and-dataset)
2. [Preprocessing: Label Hierarchy Design](#2-preprocessing-label-hierarchy-design)
3. [Benign Set Construction](#3-benign-set-construction)
4. [Data Splitting Strategy](#4-data-splitting-strategy)
5. [ML Baseline: Character-Level Classification](#5-ml-baseline-character-level-classification)
6. [LLM Classifier: Three-Stage Hierarchical Classification](#6-llm-classifier-three-stage-hierarchical-classification)
7. [Dynamic Few-Shot with Embedding Retrieval](#7-dynamic-few-shot-with-embedding-retrieval)
8. [Hybrid Router: Confidence-Based ML/LLM Routing](#8-hybrid-router-confidence-based-mllm-routing)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Results and Analysis](#10-results-and-analysis)
11. [Design Decision Summary](#11-design-decision-summary)

---

## 1. Problem Definition and Dataset

### The Threat Model

LLM applications are vulnerable to **prompt injection** and **jailbreak attacks** — adversarial inputs crafted to bypass safety filters and manipulate model behavior. Attackers modify their prompts using a variety of techniques (Unicode manipulation, word substitution, encoding tricks) to make malicious content appear benign to simple filters while preserving semantic meaning.

This pipeline acts as a **gatekeeper** — a pre-processing layer that classifies incoming prompts before they reach the target LLM, detecting whether a prompt has been adversarially modified and, if so, identifying the specific attack technique used.

### Source Dataset

**Dataset**: [Mindgard/evaded-prompt-injection-and-jailbreak-samples](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples) (HuggingFace)

This dataset contains ~11,300 adversarial samples spanning 20 attack types. Each sample has:
- `modified_sample` — the adversarially modified prompt (this is the input text for classification)
- `original_sample` — the unmodified original prompt (used for benign set construction and grouped splitting)
- `attack_name` — the attack technique used to create the modification

**Why this dataset?** It is one of the few publicly available datasets that provides both the modified attack text and the original benign prompt it was derived from, which is critical for:
1. Understanding the transformation each attack applies
2. Building a realistic benign comparison set from the same domain
3. Grouping related samples (all modifications of one original) during splitting

### Attack Types (20 total)

The dataset contains two fundamentally different families of attacks:

**Unicode-based attacks (12 types):**
- Diacritics, Underline Accent Marks, Upside Down Text, Bidirectional Text, Full Width Text, Emoji Smuggling, Spaces, Homoglyphs, Deletion Characters, Unicode Tags Smuggling, Zero Width, Numbers

**NLP-based attacks (8 types):**
- BAE, BERT-Attack, Deep Word Bug, Alzantot, Pruthi, PWWS, TextBugger, TextFooler

These two families differ in a fundamental way:
- **Unicode attacks** modify the *encoding/representation* of characters — they are structurally detectable through character-level features
- **NLP attacks** modify the *word choices* using synonym substitution or character-level perturbations — they are semantically similar to the original and much harder to distinguish from each other

---

## 2. Preprocessing: Label Hierarchy Design

> **Module**: `src/preprocess.py`
> **Config**: `configs/default.yaml` → `labels` section

### The Three-Level Hierarchy

```
Level 0 (Binary):    adversarial | benign
Level 1 (Category):  unicode_attack | nlp_attack | benign
Level 2 (Type):      12 unicode sub-types | nlp_attack (collapsed) | benign
```

### Why a Hierarchy?

A flat 21-class classifier (20 attacks + benign) would be the naive approach, but it performs poorly because:

1. **The binary decision matters most operationally.** A security gatekeeper's primary job is to decide whether to block or allow a prompt. Getting the adversarial/benign distinction right is the highest-priority task.

2. **Attack categories have fundamentally different signatures.** Unicode attacks manifest in character-level features (unusual code points, invisible characters, encoding anomalies). NLP attacks manifest in semantic-level features (word substitutions, near-synonyms). A single model conflates these two detection strategies.

3. **NLP sub-types are indistinguishable from each other.** Empirical analysis showed that NLP attack sub-types (BAE vs. TextFooler vs. BERT-Attack, etc.) classify at only ~17.9% accuracy — barely above random chance for 8 classes. These attacks all perform essentially the same operation (word-level substitution with similar candidates), making their outputs nearly identical. Attempting to separate them adds noise without useful signal.

4. **Unicode sub-types are highly separable.** In contrast, Unicode attack sub-types (diacritics vs. homoglyphs vs. zero-width, etc.) classify at 93–100% accuracy because each technique leaves a distinctive character-level fingerprint.

### The NLP Collapse Decision

All 8 NLP attack types are collapsed into a single `nlp_attack` label at Level 2. This is a deliberate design choice based on empirical evidence:

- NLP attacks (TextFooler, BAE, BERT-Attack, Alzantot, PWWS, TextBugger, Deep Word Bug, Pruthi) all operate by replacing words with semantically similar alternatives
- The resulting texts are nearly indistinguishable — they read as slightly awkward but grammatically plausible paraphrases
- Forcing the model to separate these sub-types introduces classification noise that degrades overall performance
- Collapsing them allows the model to focus on the answerable question: *"Was this text word-substituted?"* rather than the unanswerable *"Which specific word-substitution algorithm was used?"*

### How Labels Are Assigned

In `preprocess.py:add_hierarchical_labels()`:
- **Level 0**: Everything in the raw dataset is adversarial (the dataset only contains attack samples). Benign samples are constructed separately.
- **Level 1**: Mapped by looking up the `attack_name` against the configured `unicode_attacks` and `nlp_attacks` lists.
- **Level 2**: For unicode attacks, the specific `attack_name` is preserved. For NLP attacks, it's collapsed to `"nlp_attack"`.

---

## 3. Benign Set Construction

> **Module**: `src/preprocess.py` → `build_benign_set()`
> **Config**: `configs/default.yaml` → `benign` section

### The Problem

The Mindgard dataset contains only adversarial samples. To train a binary classifier (adversarial vs. benign), we need benign examples. The challenge is creating a benign set that is *representative of the same domain* — the benign samples should look like the kind of prompts users would actually submit, not random unrelated text.

### The Solution: Original Prompts as Seeds

Each adversarial sample in the dataset was created by modifying an `original_sample`. These originals are perfectly suited as benign examples because:

1. They represent the **same prompt distribution** the adversarial samples came from
2. They are the **unmodified versions** of the adversarial prompts, so the binary classifier is explicitly learning to distinguish "manipulated" from "original"
3. They cover the same **topics and intents** as the attack data

### Construction Process

1. **De-duplication**: Extract unique `original_sample` values (many attack variants share the same original)
2. **Target count**: 2,000 benign samples (configurable via `benign.target_count`)
3. **Augmentation**: If unique originals < target count, sample with replacement to reach the target. The config also supports LLM paraphrasing (`benign.paraphrase_model: gpt-4o-mini`) for richer augmentation, though the current implementation uses bootstrap resampling.
4. **Label assignment**: All benign samples receive `label_binary=benign`, `label_category=benign`, `label_type=benign`

### Why 2,000?

The adversarial set has ~11,300 samples. A 2,000-sample benign set creates a roughly 85/15 adversarial/benign ratio. This intentional imbalance reflects the real-world prior: in a security gatekeeper, the base rate of actual attacks is expected to be much lower than benign traffic. However, having enough benign samples is critical for learning the "normal" distribution.

### Prompt Hashing

Every sample (adversarial and benign) gets a `prompt_hash` — an MD5 hash of the lowercased, stripped original prompt text. This hash is the key mechanism for grouped splitting (see next section).

```python
def build_prompt_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]
```

---

## 4. Data Splitting Strategy

> **Module**: `src/build_splits.py`
> **Config**: `configs/default.yaml` → `splits` and `labels.held_out_attacks`

### Grouped Splitting by Prompt Hash

This is one of the most important design decisions in the pipeline. Standard random splitting would create **data leakage**: different modifications of the same original prompt could end up in both training and test sets. Since these share the same semantic content (just different encoding tricks), the model would effectively be tested on inputs it has "seen" semantically.

**Solution**: Split by `prompt_hash` groups. All variants of a given original prompt — across all attack types — stay together in the same split.

```
Original prompt: "How do I hack a computer?"
  ├── Diacritics variant      → all in TRAIN
  ├── Homoglyphs variant      → all in TRAIN
  ├── Zero Width variant      → all in TRAIN
  ├── TextFooler variant      → all in TRAIN
  └── Benign (original)       → all in TRAIN
```

This means:
- **No prompt overlap** between train/val/test
- The model must generalize to **new prompts**, not just new encodings of seen prompts
- Split ratios (70/15/15) apply to **groups**, not individual samples

### Held-Out Attack Types for Generalization Testing

Two attack types are **entirely excluded** from train/val/test and placed in a separate `test_unseen.parquet` split:
- **Emoji Smuggling** (a unicode attack)
- **Pruthi** (an NLP attack — character-level perturbations)

**Why?** This tests a critical real-world scenario: *Can the system detect attacks it has never been trained on?* New attack techniques are constantly being developed. A robust gatekeeper should generalize to novel attack patterns, not just memorize known ones.

The choice of held-out attacks is deliberate:
- **Emoji Smuggling** is a relatively unique unicode technique (encoding text within emoji sequences), testing whether the system can detect novel encoding schemes
- **Pruthi** uses character-level typo-like perturbations, sitting at the boundary between unicode and NLP attacks, testing generalization in the ambiguous middle ground

### Split Sizes (Typical)

| Split | Samples | Purpose |
|-------|---------|---------|
| train | ~9,300 | Model training (ML) and few-shot exemplar source (LLM) |
| val | ~2,000 | Hyperparameter tuning, threshold selection |
| test | ~2,000 | Primary evaluation |
| test_unseen | ~800 | Generalization to novel attack types |

---

## 5. ML Baseline: Character-Level Classification

> **Module**: `src/ml_baseline.py`
> **Config**: `configs/default.yaml` → `ml` section

### Why Character-Level Features?

The key insight driving the ML approach is that **unicode-based attacks leave detectable fingerprints at the character level** that are invisible at the word/token level. A word-level model (bag-of-words, word embeddings) would see "hello" and "h̲e̲l̲l̲o̲" as near-identical, but a character-level model sees fundamentally different character distributions.

### Feature Engineering

The ML baseline combines two feature sets:

#### 1. Character N-Gram TF-IDF

```python
TfidfVectorizer(
    analyzer="char_wb",    # Character n-grams at word boundaries
    ngram_range=(2, 5),    # Bigrams through 5-grams
    max_features=50000,    # Top 50k features by TF-IDF score
    sublinear_tf=True,     # Apply log(1 + tf) scaling
)
```

**Why `char_wb`?** The `char_wb` analyzer generates character n-grams only within word boundaries (padded with spaces). This captures character-level patterns while respecting word structure — important because many attacks operate within words (e.g., adding diacritics to individual characters) rather than across word boundaries.

**Why 2-5 grams?** Bigrams capture character-pair patterns (e.g., combining marks always follow base characters). Trigrams and above capture common attack signatures (e.g., zero-width character sequences, fullwidth letter patterns). Above 5 characters, the feature space explodes without proportional signal gain.

**Why `sublinear_tf`?** Applies `log(1 + tf)` to dampen the effect of term frequency. A character appearing 100 times is not 100x more informative than appearing once — the presence matters more than the exact count.

#### 2. Handcrafted Unicode Features (17 features)

These features are designed based on domain knowledge of how each attack technique modifies text:

| Feature | What It Detects | Attack Types |
|---------|----------------|--------------|
| `non_ascii_ratio` | Fraction of characters outside ASCII range | Most unicode attacks |
| `zero_width_count/ratio` | ZWSP, ZWNJ, ZWJ, LRM, RLM, Word Joiner, BOM | Zero Width, Spaces |
| `bidi_count` | Bidirectional override characters (LRE, RLE, LRO, RLO, etc.) | Bidirectional Text |
| `control_count` | Characters in Unicode categories Cc, Cf, Co, Cs | Deletion Characters, BiDi |
| `tag_count` | Characters in U+E0000–U+E007F range | Unicode Tags Smuggling |
| `fullwidth_count` | Characters in U+FF01–U+FF5E range | Full Width Text |
| `combining_count/ratio` | Characters in Unicode "M" (Mark) category | Diacritics, Underline Accents |
| `char_entropy` | Shannon entropy of character distribution | All (high entropy = unusual distribution) |
| `unique_scripts` | Number of distinct Unicode script names | Homoglyphs (mix Latin + Cyrillic, etc.) |
| `text_length` | Raw character count | Control for length-dependent features |
| `cat_Lu/Ll/Mn/Cf/So` | Ratios of specific Unicode general categories | Various |

**Why handcrafted + TF-IDF?** The handcrafted features encode expert knowledge about specific attack signatures (e.g., "zero-width characters indicate a zero-width attack"). TF-IDF captures patterns the expert might not have anticipated. The combination outperforms either alone.

### Model Architecture

```python
LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
```

**Three independent models**, one per hierarchy level:
1. `label_binary` → adversarial vs. benign (2 classes)
2. `label_category` → unicode_attack vs. nlp_attack vs. benign (3 classes)
3. `label_type` → 12 unicode types + nlp_attack + benign (14 classes)

**Why Logistic Regression?**
- Produces well-calibrated probability estimates (crucial for the hybrid router's confidence thresholding)
- Fast inference (microseconds per sample — important for a gatekeeper)
- Interpretable coefficients (can inspect which features drive decisions)
- L2 regularization (C=1.0) prevents overfitting on high-dimensional TF-IDF features
- Performs surprisingly well on this task — character-level features are highly linearly separable for unicode attacks

**Why independent models per level?** Rather than a cascading hierarchy (predict binary, then category if adversarial, then type if unicode), all three models predict independently. This means:
- Each model is optimized for its specific task
- No error propagation — a wrong binary prediction doesn't force wrong category/type predictions
- The hybrid router can use confidence from any level

### Feature Combination

TF-IDF features (sparse matrix, up to 50k dimensions) and handcrafted features (dense matrix, 17 dimensions) are combined using `scipy.sparse.hstack`. This preserves the sparsity of TF-IDF while adding the dense feature columns.

### Why Not Deep Learning?

Several reasons:
1. **Dataset size**: ~13k samples is small for deep learning. Logistic regression generalizes better with limited data.
2. **Latency**: A gatekeeper must be near-instant. Logistic regression + TF-IDF runs in microseconds. A transformer model adds milliseconds to seconds.
3. **Interpretability**: Feature importance in logistic regression is transparent. This matters for security applications where you need to explain *why* a prompt was blocked.
4. **The features are the innovation**: The real work is in the character-level feature engineering. Once those features are extracted, a linear model is sufficient because the attack patterns are highly linearly separable in this feature space.

---

## 6. LLM Classifier: Three-Stage Hierarchical Classification

> **Module**: `src/llm_classifier.py`
> **Config**: `configs/default.yaml` → `llm` section

### Architecture

The LLM classifier uses OpenAI's chat completions API (model: `gpt-4o-mini`) with a three-stage sequential pipeline:

```
Input text
    │
    ▼
┌─────────────┐
│  Stage 0:   │  "Is this adversarial or benign?"
│   Binary    │──── benign ────→ STOP (return benign)
│             │
└──────┬──────┘
       │ adversarial
       ▼
┌─────────────┐
│  Stage 1:   │  "Unicode-based or NLP-based attack?"
│  Category   │
└──────┬──────┘
       │
       ├── nlp_attack ──→ STOP (return nlp_attack)
       │
       ▼ unicode_attack
┌─────────────┐
│  Stage 2:   │  "Which specific unicode technique?"
│    Type     │
└─────────────┘
```

### Why Sequential Stages?

1. **Early exit**: Benign samples skip Stages 1–2 entirely. NLP attacks skip Stage 2. This reduces API calls and cost.
2. **Focused prompts**: Each stage has a single, well-defined question with a small label set. The LLM performs better on focused binary/ternary decisions than on flat 14-class classification.
3. **Separate system prompts**: Each stage uses a custom system prompt with task-specific domain knowledge and guidance.

### Stage 0: Binary Classification

**System prompt design**: Describes the adversarial prompt threat model and lists specific attack indicators (unusual Unicode, invisible characters, homoglyphs, synonym substitutions, reversed text, encoded content). The prompt explicitly contrasts these with "normal, unmanipulated text."

**Output**: JSON with `label` (adversarial/benign) and `confidence` (0.0–1.0). Uses OpenAI's `response_format={"type": "json_object"}` for guaranteed valid JSON.

**Max tokens**: 30 (minimal — only needs a few tokens for the JSON response).

### Stage 1: Category Classification

**System prompt design**: Provides detailed descriptions of both attack families:
- Unicode attacks: "special characters, invisible characters, visual tricks, homoglyphs, full-width, diacritics"
- NLP attacks: "replacing words with synonyms or similar-meaning words, character-level typos"

The prompt emphasizes the **observable signals** for each category to guide the LLM's attention.

### Stage 2: Type Classification (Unicode Sub-Types)

This is the most complex stage. The system prompt includes:

1. **Descriptions of all 12 unicode attack types** with concrete examples:
   ```
   - Diacritics: Adds diacritical marks above/below letters, e.g., 'hello' → 'héllö'
   - Homoglyphs: Replaces letters with visually identical chars from other scripts, e.g., Latin 'a' → Cyrillic 'а'
   - Zero Width: Inserts zero-width characters (ZWSP, ZWNJ, ZWJ) between letters
   ...
   ```

2. **Few-shot examples** (static or dynamic — see Section 7) showing text → label pairs for each attack type. This grounds the LLM's classification with concrete reference points.

**Why are attack descriptions crucial?** Without descriptions, the LLM must infer what "Unicode Tags Smuggling" means from the label name alone. The descriptions make the classification task explicit: *look for invisible Unicode tag characters in the U+E0000 range*.

### Temperature and Token Settings

- **Temperature: 0** — Deterministic outputs for reproducibility. In a classification task, we want the most likely label, not creative variation.
- **Max tokens**: 30 for binary/category (tiny JSON responses), 50 for type (slightly larger to accommodate longer type names).
- **JSON mode**: All responses use `response_format={"type": "json_object"}` to ensure parseable output.

### Few-Shot Example Selection

Static few-shot examples are sampled from the training set:
- **2 examples per unicode type** (24 total for 12 types) — fewer needed because unicode attacks are visually distinctive
- **5 examples per NLP type** — more needed because NLP attacks are subtler and more varied

Examples are selected with `random_state=42` for reproducibility.

### Cost and Usage Tracking

Every API call is tracked via the `UsageStats` dataclass:
- Total calls, prompt/completion tokens
- Latency per call
- Calls broken down by stage

This enables cost analysis: the three-stage design means a benign sample costs 1 API call, an NLP attack costs 2, and a unicode attack costs 3. The average cost depends on the data distribution.

---

## 7. Dynamic Few-Shot with Embedding Retrieval

> **Module**: `src/embeddings.py`
> **Config**: `configs/default.yaml` → `llm.few_shot` section

### Motivation

Static few-shot examples are randomly sampled from training data. They may not be representative of the specific input being classified. **Dynamic few-shot** retrieves the most *similar* training examples for each input, providing the LLM with more relevant reference points.

### ExemplarBank Architecture

The `ExemplarBank` pre-computes and stores embeddings for a subset of training examples, organized by attack type:

```
ExemplarBank
├── Diacritics:     15 texts + embeddings
├── Homoglyphs:     15 texts + embeddings
├── Zero Width:     15 texts + embeddings
├── ...             (all 20 attack types)
└── TextFooler:     15 texts + embeddings
```

**Embedding model**: `text-embedding-3-small` (OpenAI) — lightweight, fast, and sufficient for similarity matching.

**Bank size per type**: 15 exemplars (configurable). This balances diversity of examples against embedding storage and retrieval cost.

### Retrieval Process (at inference time)

For each input text during Stage 2 (unicode type classification):

1. **Embed the query**: Compute the embedding of the input text
2. **For each unicode attack type**: Retrieve the `k` most similar exemplars using cosine similarity
3. **Construct few-shot messages**: Add the retrieved examples as user/assistant message pairs in the LLM prompt

```python
def cosine_similarity(a, b):
    a_norm = a / (norm(a) + 1e-9)
    b_norm = b / (norm(b, axis=1) + 1e-9)
    return dot(b_norm, a_norm)
```

**Dynamic k**: 2 examples per type (configurable). With 12 unicode types, this provides 24 dynamically selected examples, each chosen for its similarity to the specific input.

### Why Dynamic Few-Shot?

Static examples are "one-size-fits-all" — the same examples are used regardless of the input. Dynamic retrieval provides:

1. **Input-specific context**: A homoglyphs attack using Cyrillic substitutions will retrieve Cyrillic-specific examples, not Greek ones
2. **Better coverage of within-type variation**: Attack types contain diverse sub-patterns (e.g., "Spaces" includes non-breaking spaces, zero-width spaces, thin spaces). Dynamic retrieval surfaces the most relevant sub-pattern.
3. **Reduced confusion between similar types**: By showing the most discriminative examples for each type, the LLM can better distinguish similar-looking attacks

### Trade-offs

- **Additional API cost**: Each dynamic few-shot query requires an embedding call (~$0.00002 per query with `text-embedding-3-small`)
- **Latency**: One additional API round-trip per sample (embedding computation)
- **Pre-computation**: The ExemplarBank must be built once from training data (stored as pickle)

---

## 8. Hybrid Router: Confidence-Based ML/LLM Routing

> **Module**: `src/hybrid_router.py`
> **Config**: `configs/default.yaml` → `hybrid` section

### The Core Insight

ML and LLM have complementary strengths:

| Dimension | ML Baseline | LLM Classifier |
|-----------|-------------|----------------|
| **Latency** | Microseconds | ~500ms per call |
| **Cost** | Free (local inference) | ~$0.001/sample |
| **Unicode detection** | Excellent (93-100% type acc) | Good (80% type acc) |
| **NLP detection** | Moderate (word-level signals are weak) | Good (semantic understanding) |
| **Calibration** | Well-calibrated (logistic regression) | Overconfident (self-reported) |
| **Binary accuracy** | 86% | 69% |
| **Category accuracy** | 85% | 89% |

The hybrid router exploits this complementarity: **use ML for easy cases (high confidence), escalate to LLM for hard cases (low confidence)**.

### Routing Logic

```
Input text
    │
    ▼
┌──────────────────────┐
│  ML predicts all     │  (batch inference — fast)
│  three levels +      │
│  confidence scores   │
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     │             │
 ML conf ≥ 0.85  ML conf < 0.85
     │             │
     ▼             ▼
 ┌────────┐  ┌──────────────────┐
 │ Return │  │  LLM three-stage │
 │ ML     │  │  classification  │
 │ result │  └────────┬─────────┘
 └────────┘           │
                ┌─────┴──────┐
                │             │
          LLM conf ≥ 0.7  LLM conf < 0.7
                │             │
                ▼             ▼
          ┌────────┐    ┌──────────┐
          │ Return │    │ Abstain: │
          │ LLM    │    │ "needs   │
          │ result │    │  review" │
          └────────┘    └──────────┘
```

### Threshold Selection

**ML confidence threshold: 0.85**

This threshold was selected via a **threshold sweep** that simulates routing at various thresholds without making LLM calls:

```python
thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
```

For each threshold, the sweep computes:
- How many samples ML handles (high-confidence)
- How many would be escalated to LLM
- ML accuracy on the high-confidence subset

At 0.85: ~60% of samples are handled by ML with high accuracy, 40% are escalated. This balances cost savings (60% fewer LLM calls) against accuracy (the 40% escalated are genuinely harder cases).

**LLM confidence threshold: 0.7**

If even the LLM is uncertain (confidence < 0.7), the sample is marked for human review. This is a safety net for truly ambiguous inputs.

### Why ML First?

1. **Speed**: ML inference is instant. Running ML on all samples adds negligible latency.
2. **Cost**: ML is free. Every sample ML handles avoids a ~$0.001 LLM API call.
3. **ML is better at the easy cases**: Logistic regression with character features excels at clear-cut unicode attacks. These don't need LLM understanding.
4. **Well-calibrated confidence**: Logistic regression's probability estimates are well-calibrated (unlike LLM self-reported confidence), making them reliable for routing decisions.

### Batch Efficiency

The router first runs ML prediction on **all samples at once** (vectorized TF-IDF transform + matrix multiplication), then only calls the LLM sequentially on the subset that needs escalation. This is much more efficient than running samples one at a time.

### RouterStats Tracking

The `RouterStats` dataclass tracks:
- `ml_handled`: Samples resolved by ML alone
- `llm_escalated`: Samples sent to LLM
- `abstained`: Samples where even LLM was uncertain
- `ml_rate` / `llm_rate` / `abstain_rate`: Proportions

In practice: **60% ML, 40% LLM, 0% abstain** (at default thresholds on the test set).

---

## 9. Evaluation Framework

> **Module**: `src/evaluate.py`

### Multi-Level Metrics

The evaluation framework computes metrics at each hierarchy level, reflecting the hierarchical classification design:

#### Binary Level (Most Important)

| Metric | Why It Matters |
|--------|----------------|
| Accuracy | Overall correctness |
| Adversarial precision | Of samples flagged as attacks, how many truly are? (false alarm rate) |
| Adversarial recall | Of actual attacks, how many are caught? (miss rate) |
| **False-negative rate** | **Fraction of adversarial prompts that slip through as "benign" — the critical security metric** |
| Benign precision/recall | How often benign prompts are correctly passed through |

**The false-negative rate is the single most important metric.** A false negative means an adversarial prompt bypasses the gatekeeper — a security failure. False positives (blocking benign prompts) are annoying but not dangerous.

#### Category Level

- Accuracy and Macro F1 on the unicode_attack vs. nlp_attack distinction
- Confusion matrix between the two categories
- Evaluated only on adversarial samples (benign samples are excluded since they don't have a meaningful category)

#### Type Level

- Per-type precision/recall/F1 for all 12 unicode sub-types (NLP is excluded since sub-types are collapsed)
- Macro F1 to weight all types equally regardless of frequency
- Evaluated only on unicode attack samples

#### Calibration

Confidence vs. accuracy in 10 equal-width bins (0.0–0.1, 0.1–0.2, ..., 0.9–1.0). This measures whether the model's confidence scores are meaningful:
- A well-calibrated model should have ~90% accuracy on samples where it reports 0.9 confidence
- Overconfident models report high confidence even on wrong predictions
- Calibration is especially important for the hybrid router, which uses confidence for routing decisions

### Report Generation

The evaluation produces a Markdown report with:
- Tables for each metric level
- Confusion matrices
- Per-type breakdowns
- Calibration bucketed analysis
- Cost/usage statistics (for LLM and hybrid modes)

Reports are saved to `reports/eval_report_{mode}.md`.

---

## 10. Results and Analysis

### Comparative Results

| Metric | ML-only | LLM-only | Hybrid |
|--------|---------|----------|--------|
| Binary accuracy | **86%** | 69% | 80% |
| False-negative rate | **5%** | 21.2% | **9.4%** |
| Category accuracy | 85% | 89.6% | **96.1%** |
| Unicode type accuracy | 93-100% | 80% | **100%** |
| LLM calls per 100 samples | 0 | 225 | **74** |
| Cost per 100 samples | $0 | ~$0.22 | ~$0.07 |

### Key Findings

**1. ML excels at binary detection and unicode typing**

The character-level features are near-perfect at detecting unicode attacks (93-100% per-type accuracy) and good at the binary adversarial/benign split (86% accuracy, only 5% false-negative rate). This makes sense — unicode attacks literally change the character distribution in ways that are trivially detectable with the right features.

**2. LLM struggles with binary detection but excels at category classification**

The LLM's 69% binary accuracy and 21.2% false-negative rate are surprisingly poor. This is because the LLM processes *tokens*, and many unicode attacks are invisible at the token level (zero-width characters, Unicode tags). However, when the LLM knows something is adversarial, it's excellent at distinguishing unicode from NLP attacks (89.6% category accuracy) because it can reason about the *nature* of the modification.

**3. The hybrid router gets the best of both worlds**

By letting ML handle the easy 60% (mostly clear-cut unicode attacks and obvious benign prompts), the hybrid router reserves LLM calls for the genuinely ambiguous cases. The result: 96.1% category accuracy and 100% unicode type accuracy, while using only 74 LLM calls instead of 225.

**4. LLM confidence is poorly calibrated**

The calibration tables show the LLM reports 0.85–0.95 confidence even on wrong predictions. The ML model's logistic regression probabilities are more reliable, which is why ML confidence (not LLM confidence) is used for the primary routing decision.

**5. The hybrid router eliminates LLM type-classification errors**

LLM-only achieves 80% type accuracy. Hybrid achieves 100%. This is because the ML model perfectly classifies the "easy" unicode types (which are most of them), and only escalates the hard cases where the LLM also performs well with better contextual few-shot examples.

### Cost Analysis

From the hybrid evaluation report:
- 74 LLM calls (vs. 225 for LLM-only) = **67% cost reduction**
- Breakdown: 40 binary calls + 29 category calls + 5 type calls
- Total tokens: ~108k prompt + ~1.2k completion
- Average latency: 533ms per LLM call
- ML handles 60% of traffic instantly, at zero marginal cost

---

## 11. Design Decision Summary

| Decision | Alternatives Considered | Why This Choice |
|----------|------------------------|-----------------|
| Three-level hierarchy | Flat 21-class classifier | Hierarchy matches task structure; binary is the priority; NLP types are inseparable |
| NLP collapse to single label | Keep 8 NLP sub-types | 17.9% sub-type accuracy = random; adds noise |
| Grouped splitting by prompt hash | Random splitting | Prevents data leakage from same-prompt variants |
| Held-out attack types | No held-out set | Tests real-world scenario of novel attacks |
| Char n-gram TF-IDF + handcrafted features | Word-level BoW; deep learning | Character features capture unicode fingerprints; fast inference; small data |
| Logistic regression | Random forest; SVM; neural net | Calibrated probabilities; fast; interpretable; sufficient accuracy |
| Three-stage LLM pipeline | Single flat LLM call | Early exit saves cost; focused prompts improve accuracy |
| gpt-4o-mini | gpt-4o; open-source | Best cost/quality trade-off for JSON classification tasks |
| Temperature 0 | Higher temperature | Deterministic classification; reproducibility |
| ML-first hybrid routing | LLM-first; parallel | ML is free + fast; well-calibrated for confidence routing |
| 0.85 ML confidence threshold | Other thresholds | Threshold sweep: 60% ML coverage at high accuracy |
| Dynamic few-shot | Static few-shot only | Input-specific examples improve type classification |
| Benign set from originals | Random benign text; web scraping | Same-domain; paired with adversarial variants |

---

## Appendix: Full Pipeline Execution

```bash
# Step 1: Load dataset from HuggingFace, add hierarchical labels, build benign set
python -m src.preprocess
# Output: data/processed/full_dataset.parquet (~13.3k samples)

# Step 2: Group by prompt_hash, hold out attack types, split 70/15/15
python -m src.build_splits
# Output: data/processed/{train,val,test,test_unseen}.parquet

# Step 3: Train ML models (char TF-IDF + unicode features → LogisticRegression × 3)
python -m src.ml_baseline
# Output: data/processed/ml_baseline.pkl + wandb metrics

# Step 4: Run LLM three-stage classifier on test split
python -m src.llm_classifier --split test --limit 100
# Output: data/processed/predictions_test.csv + wandb metrics

# Step 5: Run hybrid ML→LLM router on test split
python -m src.hybrid_router --limit 100
# Output: reports/eval_report_hybrid.md + wandb metrics

# Step 6: Evaluate any predictions CSV
python -m src.evaluate --predictions data/processed/predictions_test.csv
# Output: reports/eval_report_llm.md
```

### Optional: Dynamic Few-Shot

```bash
# Build exemplar bank (one-time, requires embedding API calls)
# Then run LLM classifier with dynamic retrieval
python -m src.llm_classifier --split test --limit 100 --dynamic
```

### Prediction CLI

```bash
# ML-only (instant, no API)
echo "suspicious text" | python -m src.cli.predict --mode ml --pretty

# Hybrid (recommended for production)
echo "suspicious text" | python -m src.cli.predict --mode hybrid --pretty
```
