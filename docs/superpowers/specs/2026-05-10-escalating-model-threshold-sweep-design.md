# Escalating Model Threshold Sweep Design

## Context

The escalating model estimates `P(cheap path is wrong)` from Colab/local LLM classifier output and DeBERTa output. It is trained on the `val` split, so threshold selection must not reuse `val`.

## Design

Add a threshold sweep for the escalating model using `unseen_val` as the threshold-selection split. The existing CLI continues to train on `val` and score all evaluation splits. After scoring, it will compute threshold operating points from the `unseen_val` scored rows only.

For each threshold, define:

```text
escalate_to_judge = escalation_score >= threshold
```

The sweep reports:

```text
threshold
rows
judge_call_rate
judge_calls
trusted_rows
cheap_errors_total
cheap_errors_caught
cheap_errors_missed
cheap_error_catch_rate
non_escalated_error_rate
```

## Artifacts

The CLI writes:

```text
data/processed/research/escalating_model_threshold_sweep_unseen_val.csv
```

The markdown report includes a section summarizing the `unseen_val` sweep.

## Non-Goals

This change does not integrate the threshold into runtime routing, tune on `test`, or tune on `unseen_test`. Those remain evaluation-only splits.
