# LLM Security Gatekeeper

A research project on **cost-efficient, leakage-aware detection of adversarial prompts, prompt injection, and jailbreak attempts**. The goal is not a single best model but a clear-eyed view of the tradeoffs between recall, false positives, latency/cost, and robustness on both internal and external datasets.

The system pairs a fine-tuned DeBERTa classifier with a local/Colab LLM classifier handoff, then escalates selected low-confidence cases to an LLM judge. Splits are grouped by prompt hash and a subset of attack families is held out so generalization can be measured without leakage.

Built on the [Mindgard evaded prompt injection dataset](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples) (~11.3k adversarial samples across 20 attack types + ~2k synthetic benign samples), with additional evaluation on external datasets (`deepset`, `jackhhao`, `safeguard`).

## Classification Hierarchy

```
Level 0: Binary     →  adversarial | benign
Level 1: Category   →  unicode_attack | nlp_attack
Level 2: Type       →  12 unicode sub-types (NLP sub-types collapsed; see note below)
```

NLP sub-types (TextFooler, BERT-Attack, BAE, etc.) are currently collapsed into a single `nlp_attack` label because the existing word-substitution attacks are difficult to separate reliably in this dataset — observed sub-type accuracy is around 17.9%, which is dataset- and model-specific rather than a universal claim about NLP attacks. Unicode-based attacks (homoglyphs, zero-width chars, diacritics, etc.) classify cleanly at roughly 88–100% in our experiments.

## Classifier backends

- **ML** (`src/ml_classifier/ml_baseline.py`) — char n-gram TF-IDF + handcrafted Unicode features, LogisticRegression per hierarchy level. Instant, no API.
- **DeBERTa** (`src/cli/deberta_classifier.py`, `src/models/`) — fine-tuned `microsoft/deberta-v3-base` per hierarchy level. Strong neural baseline; produces `deberta_predictions_*.parquet`.
- **LLM** (`src/llm_classifier/llm_classifier.py`) — classifier + conditional judge calls via NVIDIA NIM (or OpenAI). Supports static and dynamic few-shot retrieval.
- **Hybrid** (`src/hybrid_router.py`) — routes each sample through the configured cascade: fast ML first, DeBERTa and/or LLM for uncertain cases, with abstention when confidence remains insufficient.
- **Benign risk model** (`src/benign_risk_model.py`) — legacy post-hoc LogisticRegression path for hybrid abstain analysis. It is not part of the canonical final-verdict pipeline.
- **Escalating model** (`src/escalating_model.py`) — LightGBM classifier that joins Colab/local LLM classifier predictions with DeBERTa predictions and estimates whether the cheap/local LLM output should be escalated to the stronger judge.

> **Status note:** Hosted NVIDIA NIM endpoints no longer expose `logprobs`, which the LLM classifier path uses for token-level confidence. The planned direction is to run the classifier model locally to restore logprob-based confidence, while retaining hosted providers (NIM/OpenAI) for judge calls. This migration has not landed yet.

## Results

Metrics change based on split, sample limit, thresholds, and whether LLM stages were run. The numbers below are historical snapshots kept for context; refresh from `reports/pipeline_final_verdict_report.md` for the canonical final-verdict path.

### Representative results

Historical main test split snapshot from the legacy component-report path.

#### Unicode/type-scope ML metrics

Evaluated only on samples within the ML classifier's unicode-attack scope:

| Model              | Rows | Accuracy | Adv F1 | Benign F1 | FPR    | FNR    |
|--------------------|------|----------|--------|-----------|--------|--------|
| ML (unicode scope) | 1070 | 0.9888   | 0.9921 | 0.9804    | 0.0000 | 0.0156 |

#### Full binary test-split metrics

Evaluated on the entire binary test split (adversarial vs. benign across all attack families):

| Model  | Rows | Accuracy | Adv F1 | Benign F1 | FPR    | FNR    |
|--------|------|----------|--------|-----------|--------|--------|
| Hybrid | 1618 | 0.9580   | 0.9743 | 0.8844    | 0.1333 | 0.0212 |
| LLM    | 1618 | 0.8072   | 0.8709 | 0.6195    | 0.1533 | 0.2018 |

> **Note:** The unicode/type-scope ML metrics and the full binary metrics answer different evaluation questions and should not be read as a strict ranking. The ML row is restricted to the subset of inputs within ML's scope; the Hybrid and LLM rows cover the full split.

Hybrid routing breakdown on the same run: `ml=747`, `deberta=600`, `llm=91`, `abstain=180`. On the validated synthetic-benign slice, hybrid FPR is 0.0000 across all 220 clean benigns.

Post-hoc benign risk model (train on val margin trace, evaluate on test):

| Model                       | ROC-AUC | PR-AUC | Brier  |
|-----------------------------|---------|--------|--------|
| Isotonic (margin only)      | 0.5000  | 0.3420 | 0.2253 |
| Logistic (all features)     | 0.9425  | 0.8684 | 0.0813 |

External (unseen) generalization, combined across `deepset`, `jackhhao`, `safeguard`:

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR    | FNR    |
|-----------:|------:|---------:|-------:|----------:|-------:|-------:|
| 2427       | 34.9% | 0.7672   | 0.6231 | 0.8316    | 0.1171 | 0.4486 |

External numbers should be read as a *generalization stress test*, not as headline performance — the FNR gap vs. the in-distribution test split is a deliberate signal that this is the area to improve.

### Canonical report output

- `reports/pipeline_final_verdict_report.md` (escalation-model final verdict)

Older component, baseline, and post-hoc report paths are historical and are
not the documented final artifact.

## Setup

```bash
# Create conda environment
conda create -n llm_gate python=3.14
conda activate llm_gate

# Install dependencies
pip install -r requirements.txt

# Pick an LLM provider (default: nim) and set the matching API key
echo "LLM_PROVIDER=nim"        >  .env   # or "openai"
echo "NVIDIA_API_KEY=nvapi-..." >> .env  # required when LLM_PROVIDER=nim
echo "OPENAI_API_KEY=sk-..."    >> .env  # required when LLM_PROVIDER=openai

# Authenticate with HuggingFace (dataset requires access approval)
huggingface-cli login
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
dvc repro build_splits
dvc repro ml_model
dvc repro deberta_model
dvc repro deberta_external@deepset
dvc repro deberta_external@jackhhao
```

These stages produce the split files, ML model, and DeBERTa predictions
required by the Colab handoff and escalation model.

### 2. Run The Colab Local LLM Classifier

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
data/processed/predictions_external/llm_predictions_external_deepset_colab_local_classifier.parquet
data/processed/predictions_external/llm_predictions_external_jackhhao_colab_local_classifier.parquet
```

### 3. Validate The Handoff

```bash
dvc repro -s validate_colab_handoff
```

This validates that every configured Colab artifact exists, is
classifier-only, has `llm_stages_run == 1`, and joins cleanly with the
corresponding DVC-produced DeBERTa predictions. Missing or malformed handoff
artifacts are errors with exact paths. The pipeline does not fall back to
legacy hosted LLM classifier outputs.

### 4. Resume Through Final Verdict

```bash
dvc repro final_verdict_report
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

### Current Handoff Status

The canonical pipeline is being rerun from the start. Treat the previous
downloaded Deepset Colab handoff failure as stale until fresh Colab classifier
artifacts are produced and `dvc repro -s validate_colab_handoff` is run again.
If validation still fails on the fresh handoff, fix the handoff artifact rather
than weakening validation or falling back to legacy hosted LLM outputs.

### Non-Canonical Runtime Paths

Legacy hosted-LLM research stages, component markdown reports, post-hoc
abstain-risk-model reports, and baseline comparison scripts are not the
canonical run path. Use the DVC + Colab handoff flow above for start-to-finish
project execution.

## Experiment Tracking

All training and evaluation scripts support [Weights & Biases](https://wandb.ai/) logging:

```bash
# Login to wandb
wandb login

# Runs log automatically; disable with --no-wandb
python -m src.ml_classifier.ml_baseline --no-wandb
python -m src.cli.deberta_classifier --no-wandb
python -m src.llm_classifier.llm_classifier --no-wandb
python -m src.hybrid_router --no-wandb
```

Tracked metrics include per-level accuracy/F1, LLM token usage, latency, routing stats, and threshold sweep results. Model artifacts are saved as wandb Artifacts.

## Project Structure

```
configs/default.yaml              # All configuration (labels, splits, thresholds, ml/deberta/llm/hybrid models)
src/
  preprocess.py                   # Dataset loading + benign set construction
  build_splits.py                 # Grouped train/val/test splits
  evaluate.py                     # Metrics at all hierarchy levels
  external_datasets.py            # External dataset loading helpers
  embeddings.py                   # ExemplarBank for dynamic few-shot retrieval
  hybrid_router.py                # ML gate + LLM escalation
  benign_risk_model.py            # Legacy post-hoc benign risk model
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
    infer_split.py                # Lightweight inference over a split
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
