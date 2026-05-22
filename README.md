# LLM Security Gatekeeper

A multi-stage classifier that detects prompt-injection and jailbreak attacks at **94.6% accuracy while calling an LLM judge on only 4.7% of prompts** — a 95% reduction in inference cost versus judging every prompt.

DeBERTa-v3 binary classifier → LightGBM escalation router → selective LLM judge (NVIDIA NIM). Trained on 11.3k adversarial samples (Mindgard) + 2k synthetic benigns; evaluated on held-out attack families and three external datasets (deepset, jackhhao, safeguard) for generalization.

## Headline results

On 6,405 prompts across in-distribution and external datasets (from `reports/pipeline_final_verdict_report.md`):

| Slice                        | Rows  | Accuracy | Adv recall | Benign recall | Judge calls |
|------------------------------|------:|---------:|-----------:|--------------:|------------:|
| In-distribution              | 6,027 | **95.4%**| 96.3%      | 94.7%         | 3.8%        |
| External (deepset, jackhhao) |   378 | 81.5%    | 66.8%      | 97.8%         | 19.3%       |
| **Overall**                  | 6,405 | **94.6%**| 94.4%      | 94.8%         | **4.7%**    |

The cascade calls the LLM judge on only **4.7%** of prompts — a **95% reduction** versus judging every prompt — while preserving near-baseline accuracy. External-dataset performance is reported as a deliberate generalization stress test, not headline performance; the FNR gap on `deepset` is the most informative target for future work.

Per-split detail:

| Split             | Rows | Judge rate | Accuracy | Adv recall | Benign recall | Adv precision |
|-------------------|-----:|-----------:|---------:|-----------:|--------------:|--------------:|
| test              | 2581 |  2.48%     | 96.09%   | 98.61%     | 91.53%        | 95.45%        |
| unseen_test       | 1894 |  5.39%     | 92.19%   | 93.80%     | 89.80%        | 93.14%        |
| safeguard_test    | 1552 |  4.06%     | 98.32%   | 89.35%     | 99.42%        | 94.97%        |
| external_deepset  |  116 | 29.31%     | 60.34%   | 25.00%     | 98.21%        | 93.75%        |
| external_jackhhao |  262 | 14.89%     | 90.84%   | 84.89%     | 97.56%        | 97.52%        |

> External datasets are out-of-distribution stress tests — different label conventions and attack distributions. They are not the production target.

## Architecture

`DeBERTa-v3 binary classifier` ⟶ `LightGBM escalation router` (decides which prompts the cheap classifier is uncertain about) ⟶ `selective LLM judge` (NVIDIA NIM / OpenAI) on the routed subset.

Splits are grouped by prompt hash, and two attack families (Emoji Smuggling, Pruthi) are held out so generalization can be measured without leakage. Evaluation also covers three external datasets (`deepset`, `jackhhao`, `safeguard`).

Built on the [Mindgard evaded prompt injection dataset](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples) (~11.3k adversarial samples across 20 attack types) plus ~2k synthetic benign samples generated and filter-validated in-house.

## Stack

DeBERTa-v3-base · LightGBM · scikit-learn · DVC · Weights & Biases · PyTorch · HuggingFace Transformers · NVIDIA NIM / OpenAI APIs

## Classification Hierarchy

```
Level 0: Binary     →  adversarial | benign
Level 1: Category   →  unicode_attack | nlp_attack
Level 2: Type       →  12 unicode sub-types (NLP sub-types collapsed)
```

NLP sub-types (TextFooler, BERT-Attack, BAE, etc.) are collapsed into a single `nlp_attack` label because word-substitution attacks are difficult to separate reliably in this dataset (observed sub-type accuracy ~17.9%). Unicode-based attacks (homoglyphs, zero-width chars, diacritics, etc.) classify cleanly at 88–100%.

## Classifier backends

- **ML** (`src/ml_classifier/ml_baseline.py`) — char n-gram TF-IDF + handcrafted Unicode features, LogisticRegression per hierarchy level. Instant, no API.
- **DeBERTa** (`src/cli/deberta_classifier.py`, `src/models/`) — fine-tuned `microsoft/deberta-v3-base` per hierarchy level.
- **LLM** (`src/llm_classifier/llm_classifier.py`) — classifier + conditional judge calls via NVIDIA NIM (or OpenAI), with static and dynamic few-shot retrieval. The classifier model runs in Colab to recover token-level `logprobs` (hosted NIM/OpenAI endpoints no longer expose them); hosted providers still serve judge calls.
- **Escalating model** (`src/escalating_model.py`) — LightGBM router that joins the local LLM classifier's logprob features with DeBERTa predictions and decides which rows to escalate to the judge.

## ML Features

The ML baseline extracts character-level features that are highly discriminative for Unicode-based attacks:

- **TF-IDF char n-grams** (2-5 chars, `char_wb` analyzer)
- **Unicode category distribution** (Lu, Ll, Mn, Cf, So ratios)
- **Non-ASCII ratio**
- **Zero-width / BiDi / tag / fullwidth / combining character counts**
- **Character entropy**
- **Unique script count**

## Setup

```bash
conda env create -f environment.yml
conda activate llm-gate
make test
```

Full setup details, the DVC + Colab pipeline, the experiment-tracking guide, and the project-structure tree live in [docs/PIPELINE.md](docs/PIPELINE.md).

## License

Released under the MIT License — see [LICENSE](LICENSE).
