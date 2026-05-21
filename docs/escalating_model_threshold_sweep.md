# Escalating Model Threshold Sweep

Generated on 2026-05-10 with:

```bash
python -m src.cli.train_escalating_model
```

The escalating model is trained on `val`, so the threshold sweep is selected on
`unseen_val`.

## Correction Note

This report belongs to the escalating-model path, not the abstain risk model.
The related branch was initially planned as risk-model work, and that created
confusing project history: the changed/used artifact was the escalating model
for judge routing. The corrected implementation uses
`hybrid.escalating_model`, `data/processed/models/escalating_model.pkl`, and
`(calibrated_)escalation_score` when deciding which cheap Colab/local LLM
classifier rows are sent to the judge. The existing `hybrid.risk_model` path
still refers only to post-hoc abstain resolution from hybrid router traces.

Keeping this separation explicit avoids reproducing the wrong DVC stage,
comparing reports against the wrong artifact, or debugging judge-call behavior
with risk-model configs that are not involved in escalation routing.

## Run Outputs

```text
data/processed/models/escalating_model.pkl
data/processed/research/escalating_model_eval_test.parquet
data/processed/research/escalating_model_eval_unseen_val.parquet
data/processed/research/escalating_model_eval_unseen_test.parquet
data/processed/research/escalating_model_eval_safeguard_test.parquet
data/processed/research/escalating_model_summary.csv
data/processed/research/escalating_model_threshold_sweep_unseen_val.csv
reports/escalating_model_poc.md
reports/pipeline_final_verdict_report.md
```

## Split Summary

| split | rows joined | cheap error rate | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|
| test | 2,581 | 5.23% | 0.931 | 0.580 |
| unseen_val | 1,881 | 7.23% | 0.926 | 0.668 |
| unseen_test | 1,894 | 8.82% | 0.924 | 0.677 |
| safeguard_test | 1,552 | 4.32% | 0.976 | 0.912 |

## Unseen-Val Threshold Sweep

`unseen_val` contains 1,881 joined rows and 136 cheap-path errors. Calibration
uses one prompt-hash-disjoint half; threshold selection uses the other half
with 932 rows and 66 cheap-path errors. The table below shows representative
thresholds from
`data/processed/research/escalating_model_threshold_sweep_unseen_val.csv`.

| threshold | judge call rate | judge calls | cheap errors caught | cheap errors missed | catch rate | non-escalated error rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 11.05% | 103 | 42 | 24 | 63.64% | 2.90% |
| 0.10 | 7.73% | 72 | 41 | 25 | 62.12% | 2.91% |
| 0.25 | 6.22% | 58 | 39 | 27 | 59.09% | 3.09% |
| 0.50 | 4.83% | 45 | 36 | 30 | 54.55% | 3.38% |
| 0.70 | 3.65% | 34 | 30 | 36 | 45.45% | 4.01% |
| 0.80 | 0.00% | 0 | 0 | 66 | 0.00% | 7.08% |
| 0.95 | 0.00% | 0 | 0 | 66 | 0.00% | 7.08% |
| 1.00 | 0.00% | 0 | 0 | 66 | 0.00% | 7.08% |

## Interpretation

The selected operating point is **`hybrid.escalating_model.judge_threshold:
0.5`**. On the threshold-selection half of `unseen_val`, it calls the judge on
45/932 rows (4.83%), catches 36/66 cheap-path errors (54.55%), and reduces the
trusted cheap-path error rate from 7.08% to 3.38%.

This is acceptable for the current POC because it keeps judge workload below a
5% budget on the threshold-selection split while cutting the trusted cheap-path
error rate by more than half. Lower thresholds catch slightly more cheap errors
but spend more judge calls; higher thresholds quickly stop buying enough recall.

For a more conservative operating point, threshold `0.10` calls the judge on
7.73% of rows and catches 62.12% of cheap-path errors, leaving a 2.91% error
rate among non-escalated rows.

The canonical final-verdict report applies threshold `0.5` to internal and
external judged artifacts. It reports 254 judge calls over 6,405 rows overall
(3.97%), 229/6,027 internal rows (3.80%), and 25/378 external rows (6.61%).
External escalation is therefore first-class for the currently configured
`deepset` and `jackhhao` datasets.
