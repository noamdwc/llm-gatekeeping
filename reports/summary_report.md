# LLM Gatekeeping — Summary Report

## Overview

Three classifier backends evaluated on the Mindgard in-distribution test set and four external datasets.
LLM backend: **NVIDIA NIM** (`meta/llama-3.1-8b-instruct` classifier, `meta/llama-3.1-70b-instruct` judge).

---

## 1. In-Distribution Performance (Mindgard test set, n=1,690)

| Mode | Accuracy | Adv Recall | Benign Recall | Adv F1 | FNR |
|------|----------|-----------|---------------|--------|-----|
| ML | 94.85% | **99.94%** | 8.51% | 97.35% | 0.06% |
| Hybrid | 94.67% | 99.75% | 8.51% | 97.25% | 0.25% |
| LLM *(100 samples)* | 79.00% | 87.64% | 9.09% | 88.14% | 12.36% |

**Category classification** (adversarial samples only):

| Mode | Accuracy | Macro F1 |
|------|----------|----------|
| ML | 99.62% | 99.78% |
| Hybrid | 98.81% | 99.00% |
| LLM | 53.93% | 46.36% |

**Unicode sub-type classification**: ML and Hybrid both achieve 100% accuracy. LLM achieves 0% — it does not attempt type classification.

**Hybrid routing**: 1,664/1,690 samples (98.5%) resolved by ML; only 26 escalated to LLM, keeping API cost near zero.

---

## 2. External Dataset Generalization (ML only)

| Dataset | n | Adv% | Accuracy | Adv Recall | Benign Recall | FNR |
|---------|---|------|----------|-----------|---------------|-----|
| deepset/prompt-injections | 116 | 51.7% | 51.72% | **100%** | 0% | 0% |
| jackhhao/jailbreak-classification | 262 | 53.1% | 48.09% | 90.65% | 0% | 9.35% |
| xTRam1/safe-guard-prompt-injection | 2,049 | 31.6% | 29.04% | 91.82% | 0% | 8.18% |
| reshabhs/SPML_Chatbot_Prompt_Injection | 15,917 | 78.8% | 78.60% | **99.75%** | 0% | 0.25% |

**Dataset characteristics**:
- **deepset / jackhhao**: Small sets (~100–260 samples), NLP jailbreak-style adversarial prompts, balanced classes
- **safeguard**: Medium set (2,049), prompt injection attacks (~32% adversarial), general NLP tasks as benign
- **SPML**: Large set (15,917), prompt injection gamified challenges (~79% adversarial), general chatbot queries as benign

**Accuracy is driven almost entirely by class balance**: the model catches adversarial samples at 91–100% across all datasets, but classifies every benign sample as adversarial. Accuracy tracks directly with the adversarial base rate (SPML at 79% adversarial → 79% accuracy; safeguard at 32% adversarial → 29% accuracy).

---

## 3. Key Findings

### Strength: adversarial recall
The ML model reliably catches attacks across all datasets (90–100% recall). False negative rates are uniformly low (0–9%), which is the priority for a security gatekeeper.

### Critical weakness: benign recall = 0% on all external datasets
The model classifies virtually every external benign sample as adversarial with high confidence (mean ~0.95). This is a direct consequence of the training benign set — resampled original prompts and paraphrases — which is too narrow to represent real-world benign diversity (general QA, task instructions, NLP benchmarks, chatbot conversations).

The **safeguard dataset** makes this especially clear: 1,401 diverse benign prompts (general NLP tasks) are all classified as adversarial with mean confidence 0.952. The calibration failure is severe — in the 0.9–1.0 confidence bin, accuracy is only 4.3%. The model is maximally wrong precisely when it is most confident.

The same pattern holds across all external datasets. In the 0.9–1.0 confidence bin:
- deepset: 49.5% accuracy
- jackhhao: 12.5% accuracy
- safeguard: **4.3% accuracy**
- SPML: 43.9% accuracy (benign base rate only 21%, so accuracy here is dominated by correct adversarial predictions)

### LLM vs ML tradeoff
On the small in-distribution sample the LLM shows slightly better benign recall (9% vs 8.5%) but worse adversarial recall (88% vs 100%) and much slower throughput (~5s/sample vs instant). Category classification is significantly worse (54% vs 100%). The LLM's main value is as a judge for borderline cases in the hybrid router, not as a standalone classifier.

### Hybrid is the right default
The hybrid router gets adversarial recall (99.75%) close to pure ML while routing only 1.6% of traffic to the LLM. This keeps API cost low while adding a safety net for low-confidence ML predictions.

### Root cause of out-of-distribution failure
All external datasets share benign samples that look superficially similar to adversarial inputs: task instructions, structured prompts, role-play setups. The ML model trained exclusively on Mindgard benigns (paraphrases of attack prompts) has learned that "instruction-like text = adversarial". Fixing this requires fundamentally diversifying the training benign set.

---

## 4. Recommendations

| Priority | Issue | Fix |
|----------|-------|-----|
| High | Benign recall = 0% on all external datasets | Augment training benign set with diverse real-world prompts (general QA, task instructions, chatbot queries, NLP benchmarks) |
| High | ML over-confident on out-of-distribution benign | Apply temperature scaling or isotonic regression calibration; add OOD detection |
| Medium | LLM category classification poor | Improve few-shot examples; Llama 3.1-8B struggles with the unicode/NLP taxonomy |
| Low | LLM sample size too small (n=100) | Run full test set for a proper LLM standalone evaluation |

---

## 5. Report Sources

| Report | Path |
|--------|------|
| ML evaluation | `reports/research/eval_report_ml.md` |
| Hybrid evaluation | `reports/research/eval_report_hybrid.md` |
| LLM evaluation | `reports/research/eval_report_llm.md` |
| External — deepset | `reports/research_external/research_external_deepset.md` |
| External — jackhhao | `reports/research_external/research_external_jackhhao.md` |
| External — safeguard | `reports/research_external/research_external_safeguard.md` |
| External — spml | `reports/research_external/research_external_spml.md` |
