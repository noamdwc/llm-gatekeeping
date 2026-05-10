# Escalating Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `escalating_model` that uses Colab/local small-LLM classifier output plus DeBERTa output to decide whether a sample should be escalated to the stronger judge.

**Architecture:** The first version is an offline POC model, not a replacement for the existing abstain `risk_model`. It joins Colab/local classifier predictions with DeBERTa predictions by `sample_id`, trains a probabilistic model for `P(cheap path is wrong)`, and evaluates that score across IID, unseen-attack, and safeguard splits. Calibration, threshold sweeps, and router integration are explicitly later phases.

**Tech Stack:** Python, pandas, scikit-learn `Pipeline`, `StandardScaler`, `LogisticRegression`, parquet artifacts, pytest.

---

## Naming Contract

Use these names consistently:

```text
Concept:        Escalating model
Config key:     hybrid.escalating_model
Python class:   EscalatingModel
Dataset class:  EscalatingDataset
CLI:            src.cli.train_escalating_model
Artifact:       data/processed/models/escalating_model.pkl
Report:         reports/escalating_model_poc.md
Score column:   escalation_score
Decision col:    escalate_to_judge (later phase, not POC)
```

Keep this separate from:

```text
risk_model
```

The existing risk model resolves hybrid abstain cases from router trace features. The escalating model decides whether the cheap Colab/local LLM path should be trusted or escalated to judge.

---

## Inputs

The POC training dataset joins these files on `sample_id`:

```text
data/processed/predictions/llm_predictions_val_colab_local_classifier.parquet
data/processed/predictions/deberta_predictions_val.parquet
```

The POC evaluates on these split pairs:

```text
test:
  data/processed/predictions/llm_predictions_test_colab_local_classifier.parquet
  data/processed/predictions/deberta_predictions_test.parquet

unseen_val:
  data/processed/predictions/llm_predictions_unseen_val_colab_local_classifier.parquet
  data/processed/predictions/deberta_predictions_unseen_val.parquet

unseen_test:
  data/processed/predictions/llm_predictions_unseen_test_colab_local_classifier.parquet
  data/processed/predictions/deberta_predictions_unseen_test.parquet

safeguard_test:
  data/processed/predictions/llm_predictions_safeguard_test_colab_local_classifier.parquet
  data/processed/predictions/deberta_predictions_safeguard_test.parquet
```

Use an inner join on `sample_id` for every split and report dropped rows. This matters for `safeguard_test`, where the current local artifacts have 1552 Colab rows and 1555 DeBERTa rows.

Required Colab/local classifier columns:

```text
sample_id
label_binary
llm_pred_binary
llm_conf_binary
clf_confidence
clf_token_logprobs
```

Required DeBERTa columns:

```text
sample_id
deberta_proba_binary_adversarial
```

Optional columns that may be useful for diagnostics:

```text
modified_sample
attack_name
label_category
label_type
prompt_hash
benign_source
is_synthetic_benign
clf_label
clf_category
clf_nlp_attack_type
```

Do not depend on hybrid-router trace columns in this model:

```text
route
is_abstain
margin_source_stage
is_judge_stage
top1_logprob
top2_logprob
```

Those belong to the existing abstain risk model and are not available before judge-routing decisions.

---

## Output

The model emits a continuous score:

```text
escalation_score = P(cheap path is wrong / needs judge escalation)
```

The policy layer converts that into a decision:

```text
escalate_to_judge = escalation_score >= threshold
```

Example:

```text
escalation_score = 0.08 -> trust cheap/local classifier
escalation_score = 0.64 -> run judge
```

Expected artifacts:

```text
data/processed/models/escalating_model.pkl
data/processed/research/escalating_model_eval_test.parquet
data/processed/research/escalating_model_eval_unseen_val.parquet
data/processed/research/escalating_model_eval_unseen_test.parquet
data/processed/research/escalating_model_eval_safeguard_test.parquet
data/processed/research/escalating_model_summary.csv
reports/escalating_model_poc.md
```

---

## Training Target

Train the first version to answer the operational question:

```text
Should the cheap Colab/local LLM prediction be escalated to the judge?
```

Use a binary target:

```text
cheap_prediction = llm_pred_binary
truth = label_binary
needs_escalation = cheap_prediction != truth
```

So:

```text
y = 1 when the cheap path is wrong
y = 0 when the cheap path is correct
```

This is intentionally different from training:

```text
P(true_label == adversarial)
```

The escalating model’s job is not to classify adversarial vs benign directly. Its job is to decide whether the cheap prediction is trustworthy enough to skip the judge.

---

## Features

Start with a small, interpretable feature set:

```text
llm_conf_binary
clf_confidence
deberta_proba_binary_adversarial
llm_pred_is_adversarial
deberta_pred_is_adversarial
deberta_llm_disagree
llm_distance_from_uncertain
deberta_distance_from_uncertain
```

Definitions:

```text
llm_pred_is_adversarial = 1 if llm_pred_binary == "adversarial" else 0
deberta_pred_is_adversarial = 1 if deberta_proba_binary_adversarial >= 0.5 else 0
deberta_llm_disagree = 1 if llm_pred_is_adversarial != deberta_pred_is_adversarial else 0
llm_distance_from_uncertain = abs(llm_conf_binary - 0.5)
deberta_distance_from_uncertain = abs(deberta_proba_binary_adversarial - 0.5)
```

Add parsed logprob-margin features only if `clf_token_logprobs` has a stable format across Colab outputs. If parsing is unstable, skip it for the first version and document that decision in the report.

---

## Model

Start with the same simple pattern as the current risk model:

```text
StandardScaler + LogisticRegression
```

Recommended class behavior:

```text
EscalatingModel.train(X, y, feature_cols) -> EscalatingModel
EscalatingModel.save(path) -> None
EscalatingModel.load(path) -> EscalatingModel
EscalatingModel.predict_escalation_batch(df) -> np.ndarray
```

The saved pickle should include:

```text
pipeline
feature_cols
```

For the POC, do not create final decision columns and do not choose a production threshold. The model should emit `escalation_score`; later work will convert that score into a judge-escalation policy.

---

## POC Training and Evaluation

Train the model on `val` only:

```text
data/processed/predictions/llm_predictions_val_colab_local_classifier.parquet
data/processed/predictions/deberta_predictions_val.parquet
```

Evaluate independently on:

```text
test
unseen_val
unseen_test
safeguard_test
```

The POC question is:

```text
When rows are sorted by escalation_score, are cheap/local LLM mistakes concentrated near the top?
```

Core metrics:

```text
row count
joined row count
dropped Colab-only rows
dropped DeBERTa-only rows
cheap/local LLM error rate
ROC-AUC
PR-AUC
top 10 percent escalation-score bucket error rate
top 10 percent escalation-score bucket adversarial false-negative rate
bottom 50 percent escalation-score bucket error rate
```

The most important POC signal is:

```text
top 10 percent error rate > overall cheap/local LLM error rate
bottom 50 percent error rate < overall cheap/local LLM error rate
```

If ROC-AUC is near 0.5 and high-score rows are not meaningfully more error-prone than average, this direction is weak. If ROC-AUC and top-score concentration are strong across `test`, `unseen_test`, and `safeguard_test`, the direction is promising.

Also report adversarial false-negative concentration in the high-score bucket:

```text
top_10pct_adversarial_fn_rate =
  count(rows in top 10 percent by escalation_score where label_binary == "adversarial" and llm_pred_binary == "benign")
  / count(rows in top 10 percent by escalation_score where label_binary == "adversarial")
```

If the denominator is zero, report the metric as null for that split. This metric answers whether the model is surfacing the most safety-critical cheap-LLM mistakes near the top of the escalation queue.

Do not implement these in the POC:

```text
calibration
threshold sweep
judge-call policy
router integration
plots
```

Later phases should add calibration and threshold sweep after the POC shows useful ranking signal. Calibration bins should be based on predicted `escalation_score`, not on raw labels or raw features. Threshold sweep should then report judge-call rate, non-escalated error rate, cheap errors caught, and cheap errors missed.

The report should include one row per evaluation split:

```text
split
rows_colab
rows_deberta
rows_joined
rows_dropped_colab_only
rows_dropped_deberta_only
cheap_error_rate
roc_auc
pr_auc
top_10pct_error_rate
top_10pct_adversarial_fn_rate
bottom_50pct_error_rate
```

---

## Runtime Use Later

The initial implementation is offline-only. After validation, runtime routing can become:

```text
1. Run DeBERTa.
2. Run cheap/local LLM classifier.
3. Build escalating model features.
4. Compute escalation_score.
5. If escalation_score is below threshold, accept cheap/local LLM result.
6. If escalation_score is at or above threshold, run judge.
7. Use judge result as final when judge is run.
```

The first integration should be guarded by config:

```yaml
hybrid:
  escalating_model:
    enabled: false
    model_path: data/processed/models/escalating_model.pkl
```

Do not enable it by default until offline validation establishes an acceptable non-escalated error rate.

---

## File Structure

Create:

```text
src/escalating_model.py
src/cli/train_escalating_model.py
tests/test_escalating_model.py
```

Modify:

```text
configs/default.yaml
dvc.yaml
README.md
```

Responsibilities:

```text
src/escalating_model.py
  Dataset construction, feature engineering, model save/load/predict, and POC evaluation summaries.

src/cli/train_escalating_model.py
  CLI entrypoint that loads config, reads val Colab/local classifier and DeBERTa predictions, trains the model, writes artifacts and reports.

tests/test_escalating_model.py
  Unit tests for feature construction, target construction, model persistence, batch prediction, split joins, and POC evaluation summaries.

configs/default.yaml
  Adds hybrid.escalating_model config block.

dvc.yaml
  Adds an optional train_escalating_model stage.

README.md
  Documents the new offline model and how it differs from the existing risk model.
```

---

## Implementation Tasks

### Task 1: Dataset and Feature Builder

**Files:**

```text
Create: src/escalating_model.py
Create: tests/test_escalating_model.py
```

- [ ] Add tests that build tiny Colab and DeBERTa DataFrames and verify the joined dataset contains one row per shared `sample_id`.
- [ ] Add tests that verify `needs_escalation` is `1` when `llm_pred_binary != label_binary`.
- [ ] Add tests that verify derived features:

```text
llm_pred_is_adversarial
deberta_pred_is_adversarial
deberta_llm_disagree
llm_distance_from_uncertain
deberta_distance_from_uncertain
```

- [ ] Implement `ESCALATING_FEATURE_COLS`.
- [ ] Implement `EscalatingDataset`.
- [ ] Run:

```bash
pytest tests/test_escalating_model.py -v
```

Expected: dataset tests pass.

### Task 2: Model Wrapper

**Files:**

```text
Modify: src/escalating_model.py
Modify: tests/test_escalating_model.py
```

- [ ] Add tests for `EscalatingModel.train`.
- [ ] Add tests for `predict_escalation_batch`.
- [ ] Add tests for `save` and `load`.
- [ ] Implement `EscalatingModel` with `StandardScaler + LogisticRegression`.
- [ ] Run:

```bash
pytest tests/test_escalating_model.py -v
```

Expected: model wrapper tests pass.

### Task 3: POC Evaluation Helpers

**Files:**

```text
Modify: src/escalating_model.py
Modify: tests/test_escalating_model.py
```

- [ ] Add tests for split-level evaluation summaries.
- [ ] Add tests that verify row counts include Colab rows, DeBERTa rows, joined rows, Colab-only dropped rows, and DeBERTa-only dropped rows.
- [ ] Add tests that verify `cheap_error_rate` is computed from `needs_escalation`.
- [ ] Add tests that verify top 10 percent and bottom 50 percent error rates are computed from sorted `escalation_score`.
- [ ] Add tests that verify `top_10pct_adversarial_fn_rate` is computed only over true-adversarial rows in the top 10 percent escalation-score bucket and returns null when that bucket has no true-adversarial rows.
- [ ] Implement POC evaluation summary helper.
- [ ] Run:

```bash
pytest tests/test_escalating_model.py -v
```

Expected: POC evaluation helper tests pass.

### Task 4: Training CLI

**Files:**

```text
Create: src/cli/train_escalating_model.py
Modify: tests/test_escalating_model.py
```

- [ ] Add tests that call the CLI main function with temporary parquet inputs and output paths.
- [ ] Verify the CLI writes:

```text
escalating_model.pkl
escalating_model_eval_test.parquet
escalating_model_eval_unseen_val.parquet
escalating_model_eval_unseen_test.parquet
escalating_model_eval_safeguard_test.parquet
escalating_model_summary.csv
escalating_model_poc.md
```

- [ ] Implement CLI args:

```text
--config
--train-colab-predictions
--train-deberta-predictions
--eval-split
--model-output
--research-output-dir
--summary-output
--report-output
```

- [ ] Default train paths should use `val`.
- [ ] Default eval splits should be `test`, `unseen_val`, `unseen_test`, and `safeguard_test`.
- [ ] Default paths should match the artifact contract in this plan.
- [ ] Run:

```bash
pytest tests/test_escalating_model.py -v
```

Expected: CLI tests pass.

### Task 5: Config and DVC Stage

**Files:**

```text
Modify: configs/default.yaml
Modify: dvc.yaml
```

- [ ] Add config:

```yaml
hybrid:
  escalating_model:
    enabled: false
    model_path: data/processed/models/escalating_model.pkl
```

- [ ] Add a DVC stage:

```yaml
train_escalating_model:
  cmd: python -m src.cli.train_escalating_model
  deps:
  - src/escalating_model.py
  - src/cli/train_escalating_model.py
  - data/processed/predictions/llm_predictions_val_colab_local_classifier.parquet
  - data/processed/predictions/deberta_predictions_val.parquet
  - data/processed/predictions/llm_predictions_test_colab_local_classifier.parquet
  - data/processed/predictions/deberta_predictions_test.parquet
  - data/processed/predictions/llm_predictions_unseen_val_colab_local_classifier.parquet
  - data/processed/predictions/deberta_predictions_unseen_val.parquet
  - data/processed/predictions/llm_predictions_unseen_test_colab_local_classifier.parquet
  - data/processed/predictions/deberta_predictions_unseen_test.parquet
  - data/processed/predictions/llm_predictions_safeguard_test_colab_local_classifier.parquet
  - data/processed/predictions/deberta_predictions_safeguard_test.parquet
  params:
  - configs/default.yaml:
    - hybrid.escalating_model
  outs:
  - data/processed/models/escalating_model.pkl
  - data/processed/research/escalating_model_eval_test.parquet
  - data/processed/research/escalating_model_eval_unseen_val.parquet
  - data/processed/research/escalating_model_eval_unseen_test.parquet
  - data/processed/research/escalating_model_eval_safeguard_test.parquet
  - data/processed/research/escalating_model_summary.csv
  - reports/escalating_model_poc.md:
      cache: false
```

- [ ] Run:

```bash
dvc stage list
```

Expected: `train_escalating_model` appears in the stage list.

### Task 6: Documentation

**Files:**

```text
Modify: README.md
```

- [ ] Add a short section explaining:

```text
escalating_model predicts whether cheap/local LLM output should be escalated to judge.
risk_model remains the abstain-resolution model over hybrid router traces.
```

- [ ] Add command:

```bash
python -m src.cli.train_escalating_model
```

- [ ] Add expected artifacts.

### Task 7: Full Verification

**Files:**

```text
No source changes unless verification finds a defect.
```

- [ ] Run:

```bash
pytest tests/test_escalating_model.py -v
```

Expected: all escalating model tests pass.

- [ ] Run:

```bash
python -m src.cli.train_escalating_model
```

Expected: model, predictions, summary, and report artifacts are written.

- [ ] Inspect `reports/escalating_model_poc.md` and confirm it includes:

```text
input row count
training target definition
feature list
split-level POC metrics
explicit note that calibration and threshold sweep are deferred
```

- [ ] Run:

```bash
dvc stage list
```

Expected: DVC stage list includes `train_escalating_model`.

---

## Open Decisions Before Runtime Integration

These decisions should be made only after reviewing offline results:

```text
final escalation threshold
whether to use the model before or after existing DeBERTa fast path
whether non-escalated adversarial false negatives need a stricter policy
whether parsed token-logprob margin is stable enough to include
calibration method
threshold sweep operating points
```

Do not wire `escalating_model` into live routing until these are resolved.
