# LLM Security Gatekeeper

A multi-stage detector for adversarial prompts, prompt injection, and jailbreak attempts — designed to catch attacks while keeping inference cost low by routing only uncertain cases to an LLM judge.

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

## Setup

Conda owns the Python environment for this project; `uv` is used as the
command runner inside that active environment. This keeps local runs tied to
one explicit environment while still giving fast, reproducible command
execution. Python dependencies are managed by `environment.yml`, not by
`pyproject.toml` or `uv.lock`.

```bash
# Create and activate the Conda environment
conda env create -f environment.yml
conda activate llm-gate

# Pick an LLM provider (default: nim) and set the matching API key
echo "LLM_PROVIDER=nim"        >  .env   # or "openai"
echo "NVIDIA_API_KEY=nvapi-..." >> .env  # required when LLM_PROVIDER=nim
echo "OPENAI_API_KEY=sk-..."    >> .env  # required when LLM_PROVIDER=openai

# Authenticate with HuggingFace (dataset requires access approval)
huggingface-cli login
```

Always run project commands through `uv run --active --no-project ...` after activating the
Conda environment. The `--active` flag tells uv to use the already-active Conda
environment, while `--no-project` prevents uv from selecting or creating a
repo-local `.venv`. Do not use `uv sync` for this repository.

Common local commands are wrapped by the Makefile:

```bash
make lint     # uv run --active --no-project black --check --diff .
make format   # uv run --active --no-project black .
make test     # uv run --active --no-project pytest
make test-v   # verbose pytest with faulthandler
make repro    # uv run --active --no-project dvc repro
```

For one-off or stage-specific commands, call uv directly:

```bash
uv run --active --no-project dvc repro final_verdict_report
uv run --active --no-project python -m src.cli.final_verdict_report
uv run --active --no-project pytest tests/test_final_verdict_report.py -q
```

NIM model names in `configs/default.yaml` are auto-translated to OpenAI equivalents when `LLM_PROVIDER=openai`. After switching providers, run `./run_llm_provider_refresh.sh` to force re-execution of LLM-dependent DVC stages (DVC does not track `LLM_PROVIDER` itself).

## Canonical Pipeline

The canonical end-to-end path is DVC-driven with one manual handoff: the
local LLM classifier runs in Colab, then its classifier-only parquet outputs
are downloaded back into this repo. The final documented artifact is:

```bash
reports/pipeline_final_verdict_report.md
```

### 1. Prepare Local DVC Inputs

```bash
uv run --active --no-project dvc repro build_splits
uv run --active --no-project dvc repro ml_model
uv run --active --no-project dvc repro deberta_model
uv run --active --no-project dvc repro deberta_external@deepset
uv run --active --no-project dvc repro deberta_external@jackhhao
```

These stages produce the split files, ML model, and DeBERTa predictions
required by the Colab handoff and escalation model.

### 2. Run The Colab Local LLM Classifier

The classifier model runs in Google Colab to get a cheap GPU for local
inference; hosted NIM/OpenAI endpoints no longer expose `logprobs`, and a
Colab GPU is the most cost-effective way to recover token-level confidence
without paying for dedicated GPU hosting.

Open and run:

```bash
notebooks/colab_local_llm_classifier.ipynb
```

Download the classifier-only artifacts into these exact paths:

```bash
data/processed/predictions/llm_predictions_val_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_test_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_unseen_val_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_unseen_test_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_safeguard_test_colab_local_classifier.parquet
data/processed/predictions_external/llm_predictions_external_deepset.parquet
data/processed/predictions_external/llm_predictions_external_jackhhao.parquet
```

### 3. Validate The Handoff

```bash
uv run --active --no-project dvc repro -s validate_colab_handoff
```

This validates that every configured Colab artifact exists, is
classifier-only, has `llm_stages_run == 1`, and joins cleanly with the
corresponding DVC-produced DeBERTa predictions. Missing or malformed handoff
artifacts are errors with exact paths. The pipeline does not fall back to
legacy hosted LLM classifier outputs.

### 4. Resume Through Final Verdict

```bash
uv run --active --no-project dvc repro final_verdict_report
```

This trains the escalation model from validated Colab classifier outputs plus
DVC-produced DeBERTa predictions, runs selective judge stages, and writes:

```bash
reports/pipeline_final_verdict_report.md
```

Every `train_escalating_model` input is either DVC-produced or a validated
manual Colab handoff artifact. Missing configured external judged artifacts
also fail before final report generation.

### Main DVC Graph

```
preprocess -> build_splits
                |
                +-> ml_model
                +-> deberta_model -> deberta_external@{deepset,jackhhao}
                                      |
Colab notebook -> *_colab_local_classifier.parquet
                                      |
                         validate_colab_handoff
                                      |
                         train_escalating_model
                                      |
      judge_colab_local_predictions@{test,unseen_test,safeguard_test}
      judge_colab_local_predictions_external@{deepset,jackhhao}
                                      |
                         final_verdict_report
```

External dataset stages that are DVC `foreach` stages are driven by
`external_datasets` in `configs/default.yaml`. Aggregate stages with explicit
DVC `deps`, such as `validate_colab_handoff`, `train_escalating_model`, and
`final_verdict_report`, cannot dynamically expand those dependency lists from
the config. When adding a new external dataset, update those aggregate stage
deps/outs and the Colab artifact path list, or run/report the new dataset
separately.

### Non-Canonical Runtime Paths

Legacy hosted-LLM research stages, component markdown reports, post-hoc
abstain-risk-model reports, and baseline comparison scripts are not the
canonical run path. Use the DVC + Colab handoff flow above for start-to-finish
project execution.

## Experiment Tracking

All training and evaluation scripts support [Weights & Biases](https://wandb.ai/) logging:

```bash
# Login to wandb
uv run --active --no-project wandb login

# Runs log automatically; disable with --no-wandb
uv run --active --no-project python -m src.ml_classifier.ml_baseline --no-wandb
uv run --active --no-project python -m src.cli.deberta_classifier --no-wandb
uv run --active --no-project python -m src.llm_classifier.llm_classifier --no-wandb
```

Tracked metrics include per-level accuracy/F1, LLM token usage, latency, and threshold sweep results. Model artifacts are saved as wandb Artifacts.

## Project Structure

```
configs/default.yaml              # All configuration (labels, splits, thresholds, ml/deberta/llm/hybrid models)
src/
  preprocess.py                   # Dataset loading + benign set construction
  build_splits.py                 # Grouped train/val/test splits
  evaluate.py                     # Metrics at all hierarchy levels
  external_datasets.py            # External dataset loading helpers
  embeddings.py                   # ExemplarBank for dynamic few-shot retrieval
  escalating_model.py             # Judge-escalation model
  synthetic_benign.py             # Synthetic benign prompt generation (categories A–F)
  validators.py                   # HeuristicBenignValidator, JudgeBenignValidator, DeduplicateFilter
  ml_classifier/ml_baseline.py    # Char-level ML classifier
  llm_classifier/llm_classifier.py# Classifier + judge LLM classifier
  models/                         # DeBERTa model wrappers + training utilities
  baselines/                      # Legacy HF baseline helpers
  cli/
    deberta_classifier.py         # Train/predict DeBERTa per hierarchy level
    validate_colab_handoff.py     # Validate manual Colab classifier handoff artifacts
    train_escalating_model.py     # Fit escalating_model.pkl from Colab/local + DeBERTa predictions
    final_verdict_report.py       # Generate the canonical final-verdict report
    judge_colab_local_predictions.py # Run selective judge calls from escalation scores
    generate_synthetic_benign.py  # CLI for synthetic benign generation pipeline
data/processed/
  full_dataset.parquet            # Combined adversarial + benign
  synthetic_benign/               # Generated benign parquets per category (A–F)
  splits/                         # train.parquet, val.parquet, test.parquet, unseen_val.parquet, unseen_test.parquet, safeguard_test.parquet
  models/
    ml_baseline.pkl
    escalating_model.pkl          # Judge-escalation model
  predictions/
    ml_predictions_*.parquet
    deberta_predictions_*.parquet
    llm_predictions_*_colab_local_classifier.parquet
    llm_predictions_*_colab_local_judged.parquet
  predictions_external/
    deberta_predictions_external_<ds>.parquet
    llm_predictions_external_<ds>_colab_local_classifier.parquet
    llm_predictions_external_<ds>_colab_local_judged.parquet
  research/
    escalating_model_eval_<split>.parquet
    escalating_model_summary.csv
artifacts/
  deberta_classifier/             # model/, tokenizer/, label_mapping.json, train_history.json
reports/
  deberta_classifier/             # metrics.json, classification_report.json, summary.md
  colab_handoff_validation.json
  escalating_model_poc.md
  pipeline_final_verdict_report.md
```

## ML Features

The ML baseline extracts character-level features that are highly discriminative for Unicode-based attacks:

- **TF-IDF char n-grams** (2-5 chars, `char_wb` analyzer)
- **Unicode category distribution** (Lu, Ll, Mn, Cf, So ratios)
- **Non-ASCII ratio**
- **Zero-width / BiDi / tag / fullwidth / combining character counts**
- **Character entropy**
- **Unique script count**
