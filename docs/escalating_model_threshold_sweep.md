# Escalating Model Threshold Sweep

Generated on 2026-05-10 with:

```bash
python -m src.cli.train_escalating_model
```

The escalating model is trained on `val`, so the threshold sweep is selected on
`unseen_val`.

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
```

## Split Summary

| split | rows joined | cheap error rate | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|
| test | 2,581 | 5.23% | 0.931 | 0.580 |
| unseen_val | 1,881 | 7.23% | 0.926 | 0.668 |
| unseen_test | 1,894 | 8.82% | 0.924 | 0.677 |
| safeguard_test | 1,552 | 4.32% | 0.976 | 0.912 |

## Unseen-Val Threshold Sweep

`unseen_val` contains 1,881 joined rows and 136 cheap-path errors. The table
below shows representative thresholds from
`data/processed/research/escalating_model_threshold_sweep_unseen_val.csv`.

| threshold | judge call rate | judge calls | cheap errors caught | cheap errors missed | catch rate | non-escalated error rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 12.87% | 242 | 92 | 44 | 67.65% | 2.68% |
| 0.10 | 10.37% | 195 | 87 | 49 | 63.97% | 2.91% |
| 0.25 | 7.60% | 143 | 84 | 52 | 61.76% | 2.99% |
| 0.50 | 5.90% | 111 | 81 | 55 | 59.56% | 3.11% |
| 0.70 | 5.00% | 94 | 75 | 61 | 55.15% | 3.41% |
| 0.90 | 3.93% | 74 | 64 | 72 | 47.06% | 3.98% |
| 0.95 | 3.19% | 60 | 55 | 81 | 40.44% | 4.45% |
| 1.00 | 0.00% | 0 | 0 | 136 | 0.00% | 7.23% |

## Interpretation

The model gives useful cost/error tradeoffs on `unseen_val`. Calling the judge
on only 5.90% of rows at threshold `0.50` catches 59.56% of cheap-path errors
and reduces the trusted cheap-path error rate from 7.23% to 3.11%.

For a lower-cost operating point, threshold `0.95` calls the judge on 3.19% of
rows and still catches 40.44% of cheap-path errors, leaving a 4.45% error rate
among non-escalated rows.

For a more conservative operating point, threshold `0.10` calls the judge on
10.37% of rows and catches 63.97% of cheap-path errors, leaving a 2.91% error
rate among non-escalated rows.

No production threshold is selected here. Before runtime integration, evaluate
the candidate threshold on `unseen_test` and confirm the judge-call budget.
