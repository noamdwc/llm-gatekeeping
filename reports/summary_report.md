# LLM Gatekeeping — Summary Report

## Overview

Three classifier backends evaluated on the Mindgard in-distribution test set and four external datasets.
LLM backend: **NVIDIA NIM** (`meta/llama-3.1-8b-instruct` classifier, `meta/llama-3.1-70b-instruct` judge).

---

## 1. In-Distribution Performance (Mindgard test set)

| Mode | n | Accuracy | Adv Recall | Benign Recall | Adv F1 | FNR |
|------|---|----------|-----------|---------------|--------|-----|
| ML | 1,690 | 94.85% | **99.94%** | 8.51% | 97.35% | 0.06% |
| Hybrid | 1,690 | 94.91% | 99.56% | **15.96%** | 97.37% | 0.44% |
| LLM *(100 samples)* | 100 | 75.00% | 74.16% | **81.82%** | 84.08% | 25.84% |

**Category classification** (adversarial samples only):

| Mode | Accuracy | Macro F1 |
|------|----------|----------|
| ML | 99.62% | 99.78% |
| Hybrid | 99.31% | 99.60% |
| LLM | 32.58% | 36.92% |

**Unicode sub-type classification**: ML and Hybrid both achieve 100% accuracy. LLM achieves 0% — it does not attempt type classification.

**Hybrid routing**: 1,664/1,690 samples (98.5%) resolved by ML; 26 escalated to LLM. The LLM's improved benign recall (81.82%) lifts hybrid benign recall from 8.51% (pure ML) to 15.96% — a meaningful gain from just 1.5% LLM traffic.

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
- SPML: 59.0% accuracy (benign base rate only 21%, so accuracy here is dominated by correct adversarial predictions)

### LLM behavior has shifted: benign-favoring
Compared to the previous run, the LLM now strongly favors benign recall (**81.82%**, up from 18.18%) at the cost of adversarial recall (**74.16%**, down from 93.26%). FNR has increased from 6.74% to **25.84%** — meaning the LLM misses one in four attacks. This makes the LLM unsuitable as a standalone classifier for a security gatekeeper but highly effective as a false-positive reducer for borderline ML cases.

Category classification has also degraded (32.58% vs prior 44.94%); the LLM is not reliably distinguishing unicode from NLP attacks.

### Hybrid is the right default
The hybrid router balances both failure modes:
- **99.56% adversarial recall** (only 0.44% FNR)
- **15.96% benign recall** — nearly double pure ML (8.51%), at the cost of routing 26 samples (1.5%) to the LLM

The benign recall improvement in hybrid is now more pronounced than before, reflecting the LLM's shift toward benign-favoring behavior. The tradeoff: FNR increases from 0.06% (ML) to 0.44% (Hybrid). For a strict security gatekeeper, the additional 6 missed attacks per 1,690 samples may be unacceptable — pure ML is safer on FNR.

### Root cause of out-of-distribution failure
All external datasets share benign samples that look superficially similar to adversarial inputs: task instructions, structured prompts, role-play setups. The ML model trained exclusively on Mindgard benigns (paraphrases of attack prompts) has learned that "instruction-like text = adversarial". Fixing this requires fundamentally diversifying the training benign set.

---

## 4. Recommendations

| Priority | Issue | Fix |
|----------|-------|-----|
| **High** | Benign recall = 0% on all external datasets | Augment training benign set with diverse real-world prompts (general QA, task instructions, chatbot queries, NLP benchmarks) — synthetic benign pipeline is in place for this |
| **High** | ML over-confident on out-of-distribution benign | Apply temperature scaling or isotonic regression calibration; add OOD detection |
| **High** | LLM FNR = 25.84% (misses 1 in 4 attacks) | Do not use LLM standalone; use hybrid only; investigate LLM prompt/config causing benign-favoring shift |
| **Medium** | LLM category classification very poor (32.58%) | Improve few-shot examples; Llama 3.1-8B struggles with the unicode/NLP taxonomy |
| **Medium** | Hybrid FNR (0.44%) vs ML FNR (0.06%) | Evaluate whether benign recall gain (15.96% vs 8.51%) justifies 7x increase in FNR for the target use case |
| **Low** | LLM sample size too small (n=100) | Run full test set for a proper LLM standalone evaluation |

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
