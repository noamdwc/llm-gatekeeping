# LLM Security Gatekeeper

A research project on **cost-efficient, leakage-aware detection of adversarial prompts, prompt injection, and jailbreak attempts**. The goal is not a single best model but a clear-eyed view of the tradeoffs between recall, false positives, latency/cost, and robustness on both internal and external datasets.

The system pairs a fast char-level ML detector with a fine-tuned DeBERTa classifier, escalates only low-confidence cases to an LLM (selective routing), and applies a post-hoc benign risk model to suppress false positives on clean inputs. Splits are grouped by prompt hash and a subset of attack families is held out so generalization can be measured without leakage.

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
- **Benign risk model** (`src/benign_risk_model.py`) — post-hoc LogisticRegression trained on hybrid margin traces + DeBERTa probabilities to flag false-positive-prone benigns. Outputs `data/processed/models/risk_model.pkl` and `reports/posthoc_benign_risk_model.md`.

> **Status note:** Hosted NVIDIA NIM endpoints no longer expose `logprobs`, which the LLM classifier path uses for token-level confidence. The planned direction is to run the classifier model locally to restore logprob-based confidence, while retaining hosted providers (NIM/OpenAI) for judge calls. This migration has not landed yet.

## Results

Metrics change based on split, sample limit, thresholds, and whether LLM stages were run. The numbers below are illustrative of the current pipeline; refresh from the canonical reports for any external use.

### Representative results

Main test split snapshot from `reports/research/summary_report.md`.

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

### Canonical report outputs

- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_hybrid.md` (strict LLM coverage only)
- `reports/research/eval_report_llm.md`
- `reports/research/summary_report.md` (combined main + unseen-external metrics)
- `reports/research_external/research_external_<dataset>.md`
- `reports/deberta_classifier/summary.md`
- `reports/posthoc_benign_risk_model.md`
- `reports/baselines/comparison_report.md`

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

## Two Pipeline Modes

### Research pipeline (reproducible, DVC)

The canonical way to run the project. Produces **research parquets** with all intermediate probabilities plus **markdown reports**.

```bash
# Full reproducible run (all stages, including LLM and DeBERTa)
dvc repro
```

DVC caches predictions and model artifacts. Threshold-only changes (e.g. `hybrid.ml_confidence_threshold`) re-run only the research/eval stages and never re-hit the LLM API.

#### DVC stage graph

```
generate_synthetic_benign@{A..F}        (one-time; API tokens)
                ↓
preprocess → build_splits
                ↓
        ┌──────┼─────────────┐
        ↓      ↓             ↓
     ml_model  deberta_model llm_classifier(+_val)
                ↓
        research(+_val) → train_risk_model → risk_model
                ↓
research_external_llm@{ds} → research_external@{ds} → eval_new_external@{ds}
                ↓
              eval_new   (writes the canonical eval reports)
```

`research`/`research_val` produce `research_<split>.parquet` plus `hybrid_margin_trace_<split>.parquet`. `train_risk_model` consumes the val trace + DeBERTa val predictions to produce `risk_model.pkl`, which is then used by `research` (post-hoc benign filter). `risk_model` is the post-hoc evaluation stage that writes `reports/posthoc_benign_risk_model.md` and ROC/PR/calibration plots under `reports/artifacts/`.

#### Run a single stage

```bash
dvc repro preprocess
dvc repro ml_model
dvc repro deberta_model
dvc repro llm_classifier
dvc repro research
dvc repro train_risk_model
dvc repro risk_model
dvc repro research_external@deepset
dvc repro eval_new
dvc repro eval_new_external@deepset
```

`eval_new` writes:
- `reports/research/eval_report_ml.md`
- `reports/research/eval_report_llm.md`
- `reports/research/eval_report_hybrid.md` (strict LLM coverage only)
- `reports/research/summary_report.md`

`eval_new_external@{dataset}` writes `reports/research_external/research_external_<dataset>.md`.

#### Add a new external dataset (no recompute for existing ones)

1. Add the dataset config under `external_datasets` in `configs/default.yaml`.
2. Run:

```bash
dvc repro research_external@new_dataset
```

DVC computes only the new stage; existing `research_external@...` outputs remain cached.

#### What triggers recomputation

- `configs/default.yaml:dataset|labels|benign` → `preprocess` and everything downstream.
- `configs/default.yaml:splits|labels.held_out_attacks` → `build_splits` and downstream.
- `configs/default.yaml:ml` → `ml_model` and downstream.
- `configs/default.yaml:deberta` → `deberta_model` and downstream.
- `configs/default.yaml:llm` → `llm_classifier(+_val)` and `research_external_llm@*` and downstream.
- `configs/default.yaml:hybrid.ml_confidence_threshold` → `research`, `research_val`, all `research_external@{ds}`, `eval_new`, `eval_new_external@{ds}`. Does **not** re-run training or LLM stages.
- `configs/default.yaml:hybrid.risk_model` → `train_risk_model`, `research`, `research_val`, `risk_model`, and `eval_new`.
- One external dataset config `external_datasets.<key>` → only `research_external_llm@<key>` and `research_external@<key>` and `eval_new_external@<key>`.

### Inference pipeline (lightweight, bash)

Fast CLI for running just what you need. Assumes splits + a trained ML model already exist (from a prior `dvc repro`).

```bash
./run_inference.sh --mode ml --split test
./run_inference.sh --mode hybrid --split test --limit 100
./run_inference.sh --mode llm --split test --limit 50
./run_inference.sh --mode ml --split test_unseen
./run_inference.sh --mode llm --split test --dynamic        # dynamic few-shot
```

## Pipeline (module-level commands)

Run the pipeline step by step:

```bash
# 1. Generate synthetic benign prompts (optional; categories A–F)
python -m src.cli.generate_synthetic_benign --category all --limit 100

# 2. Preprocess: load dataset, build benign set, add hierarchical labels
python -m src.preprocess

# 3. Build splits: grouped by prompt hash, held-out attack types
python -m src.build_splits

# 4. Train ML baseline + write research prediction parquets
python -m src.ml_classifier.ml_baseline --research

# 5. Train DeBERTa classifier (per-level fine-tuning + research predictions)
python -m src.cli.deberta_classifier --research --no-wandb
python -m src.cli.deberta_classifier --train-only          # skip prediction
python -m src.cli.deberta_classifier --predict-only        # predict from saved model

# 6. Run LLM classifier (requires API key for current LLM_PROVIDER)
python -m src.llm_classifier.llm_classifier --split test --research
python -m src.llm_classifier.llm_classifier --split test --research --dynamic

# 7. Merge predictions + hybrid routing + margin trace
python -m src.research --split test
python -m src.research --split val

# 8. Train and evaluate the post-hoc benign risk model
python -m src.cli.train_risk_model
python -m src.cli.benign_risk_model \
    --train-trace data/processed/research/hybrid_margin_trace_val.parquet \
    --trace data/processed/research/hybrid_margin_trace_test.parquet \
    --split test

# 9. Generate canonical evaluation reports (main + external)
python -m src.cli.eval_new --split test --config configs/default.yaml
```

### External datasets (additive + cached with DVC)

External datasets are configured in `configs/default.yaml` under `external_datasets`. DVC runs them as independent `foreach` stages (`research_external_llm@<ds>`, `research_external@<ds>`, `eval_new_external@<ds>`) so adding a new dataset only computes new stages.

To add a dataset:
- Add its config under `external_datasets` in `configs/default.yaml`.
- Run `dvc repro` (only the new dataset stages run; existing ones are cached).

### HF baselines

Run published HuggingFace prompt-injection guard models against internal splits and external datasets, and compare them to ML/hybrid:

```bash
python -m src.cli.run_baseline --baseline all --split all --external all
python -m src.cli.eval_baselines
```

Per-baseline predictions land in `data/processed/baselines/`; the comparison report is written to `reports/baselines/comparison_report.md`. Available baselines: `sentinel_v2`, `protectai_v2`.

### Synthetic benign generation

LLM-generated benign prompts across 6 categories (A–F: instructions, factual, creative, code, conversational, ambiguous-but-benign), validated by a three-layer pipeline: heuristic regex → LLM judge → embedding-based dedup (cosine sim > 0.95). Controlled by `benign.synthetic.enabled` (default: false). Validated parquets land in `data/processed/synthetic_benign/synthetic_benign_<category>.parquet` and are picked up automatically by `preprocess` when enabled.

## Prediction CLI

```bash
# ML-only (no API calls, instant)
echo "some suspicious text" | python -m src.cli.predict --mode ml --pretty

# LLM-only (requires API key for current LLM_PROVIDER)
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
python -m src.cli.deberta_classifier --no-wandb
python -m src.llm_classifier.llm_classifier --no-wandb
python -m src.hybrid_router --no-wandb
```

Tracked metrics include per-level accuracy/F1, LLM token usage, latency, routing stats, and threshold sweep results. Model artifacts are saved as wandb Artifacts.

## Project Structure

```
configs/default.yaml              # All configuration (labels, splits, thresholds, ml/deberta/llm/hybrid/risk_model)
src/
  preprocess.py                   # Dataset loading + benign set construction
  build_splits.py                 # Grouped train/val/test splits
  research.py                     # Merge predictions + hybrid routing + margin trace
  evaluate.py                     # Metrics at all hierarchy levels
  eval_external.py                # Binary-only evaluation for external datasets
  embeddings.py                   # ExemplarBank for dynamic few-shot retrieval
  hybrid_router.py                # ML gate + LLM escalation
  benign_risk_model.py            # Post-hoc benign risk model (LogisticRegression)
  synthetic_benign.py             # Synthetic benign prompt generation (categories A–F)
  validators.py                   # HeuristicBenignValidator, JudgeBenignValidator, DeduplicateFilter
  ml_classifier/ml_baseline.py    # Char-level ML classifier
  llm_classifier/llm_classifier.py# Classifier + judge LLM classifier
  models/                         # DeBERTa model wrappers + training utilities
  baselines/                      # Threshold helpers + HF baseline runners
  cli/
    predict.py                    # Classify arbitrary text (stdin or file) → JSON
    deberta_classifier.py         # Train/predict DeBERTa per hierarchy level
    train_risk_model.py           # Fit risk_model.pkl from val trace + DeBERTa
    benign_risk_model.py          # Post-hoc evaluation of the benign risk model
    research_external.py          # Research artifacts for one external dataset
    eval_new.py                   # Generate canonical markdown reports
    run_baseline.py               # Run HF guard models on internal/external splits
    eval_baselines.py             # Compare HF baselines vs ML/hybrid
    generate_synthetic_benign.py  # CLI for synthetic benign generation pipeline
    infer_split.py                # Lightweight inference over a split
    margin_calibration_fit.py     # Fit calibration on hybrid margins
    margin_calibration_report.py  # Calibration report
    margin_crossfit_eval.py       # Cross-fit evaluation of margin calibration
data/processed/
  full_dataset.parquet            # Combined adversarial + benign
  synthetic_benign/               # Generated benign parquets per category (A–F)
  splits/                         # train.parquet, val.parquet, test.parquet, test_unseen.parquet
  models/
    ml_baseline.pkl
    risk_model.pkl                # Post-hoc benign risk model
  predictions/
    ml_predictions_*.parquet
    deberta_predictions_*.parquet
    llm_predictions_*.parquet
  predictions_external/
    llm_predictions_external_<ds>.parquet
  research/
    research_<split>.parquet
    hybrid_margin_trace_<split>.parquet
    posthoc_benign_risk_predictions.parquet
    posthoc_benign_risk_summary.csv
  research_external/
    research_external_<ds>.parquet
  baselines/                      # Per-baseline prediction parquets
artifacts/
  deberta_classifier/             # model/, tokenizer/, label_mapping.json, train_history.json
reports/
  research/                       # eval_report_ml|llm|hybrid.md, summary_report.md
  research_external/              # Per-external-dataset reports
  deberta_classifier/             # metrics.json, classification_report.json, summary.md
  baselines/                      # comparison_report.md
  artifacts/                      # benign_risk_roc.png, benign_risk_pr.png, benign_risk_calibration.png
  posthoc_benign_risk_model.md
```

## ML Features

The ML baseline extracts character-level features that are highly discriminative for Unicode-based attacks:

- **TF-IDF char n-grams** (2-5 chars, `char_wb` analyzer)
- **Unicode category distribution** (Lu, Ll, Mn, Cf, So ratios)
- **Non-ASCII ratio**
- **Zero-width / BiDi / tag / fullwidth / combining character counts**
- **Character entropy**
- **Unique script count**
