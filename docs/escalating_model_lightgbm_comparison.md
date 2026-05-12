# Escalating Model: Logistic Regression vs LightGBM

## Summary

The escalating model estimates `P(cheap path is wrong)` so cheap Colab/local LLM classifier outputs can be ranked for escalation to the stronger judge.

The original POC used `StandardScaler + LogisticRegression`. The replacement uses `LightGBM` via `LGBMClassifier(random_state=42, verbosity=-1)` with the same feature set, training split, evaluation splits, and wrapper API.

LightGBM improves ROC-AUC and PR-AUC on every evaluated split. The largest gains are on `test`, `unseen_test`, and `safeguard_test` PR-AUC, which is the most relevant metric for ranking relatively rare cheap-path errors near the top of an escalation queue.

## Method

Both models were trained on the same joined validation prediction set:

- training rows: `2,557`
- target: `needs_escalation = llm_pred_binary != label_binary`
- features: `ESCALATING_FEATURE_COLS` from `src/escalating_model.py`
- evaluation splits: `test`, `unseen_val`, `unseen_test`, `safeguard_test`

The LightGBM model was trained from the current code. The logistic regression baseline was trained from the previous `HEAD` code in a temporary worktree, using the same parquet inputs.

## Results

| split | rows | cheap error rate | ROC-AUC LightGBM | ROC-AUC logistic | ROC-AUC delta | PR-AUC LightGBM | PR-AUC logistic | PR-AUC delta |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| test | 2581 | 0.0523 | 0.9313 | 0.8729 | +0.0583 | 0.5802 | 0.5274 | +0.0528 |
| unseen_val | 1881 | 0.0723 | 0.9257 | 0.9056 | +0.0201 | 0.6680 | 0.6482 | +0.0198 |
| unseen_test | 1894 | 0.0882 | 0.9244 | 0.8879 | +0.0365 | 0.6772 | 0.6204 | +0.0568 |
| safeguard_test | 1552 | 0.0432 | 0.9763 | 0.9650 | +0.0113 | 0.9118 | 0.8470 | +0.0648 |

## Bucket Metrics

The top and bottom bucket metrics show whether the escalation score is useful as an ordering signal.

| split | top 10% error rate LightGBM | top 10% error rate logistic | bottom 50% error rate LightGBM | bottom 50% error rate logistic |
|:--|--:|--:|--:|--:|
| test | 0.3127 | 0.3282 | 0.0008 | 0.0070 |
| unseen_val | 0.4603 | 0.4656 | 0.0021 | 0.0053 |
| unseen_test | 0.5526 | 0.5158 | 0.0063 | 0.0116 |
| safeguard_test | 0.4038 | 0.4038 | 0.0026 | 0.0026 |

LightGBM keeps the bottom half cleaner on `test`, `unseen_val`, and `unseen_test`, which is useful if the future policy accepts low-score cheap outputs without judge escalation. On `unseen_test`, LightGBM also concentrates more cheap-path errors into the top 10% than logistic regression.

The `test` and `unseen_val` top 10% error rates are slightly lower for LightGBM despite better ROC-AUC and PR-AUC. This means the overall ranking improved, but the very top bucket composition shifted. Threshold selection should therefore be done with a sweep, not by assuming the top 10% bucket is the final operating point.

## Interpretation

`cheap_error_rate` is unchanged between models because it measures the cheap classifier's actual error rate on each split. It is not affected by the escalating model, which only ranks examples by predicted escalation need.

ROC-AUC measures whether cheap-path errors are ranked above correct cheap-path outputs across all thresholds. PR-AUC is more important here because `needs_escalation` is relatively sparse, especially on `test` and `safeguard_test`. LightGBM improves both metrics consistently.

The current evidence supports replacing logistic regression with LightGBM for the offline escalating model POC. The follow-up threshold sweep is documented in `docs/escalating_model_threshold_sweep.md`; before runtime integration, evaluate the candidate threshold on `unseen_test` and confirm the judge-call budget.

## Artifacts

Canonical LightGBM outputs:

- `data/processed/models/escalating_model.pkl`
- `data/processed/research/escalating_model_summary.csv`
- `data/processed/research/escalating_model_eval_test.parquet`
- `data/processed/research/escalating_model_eval_unseen_val.parquet`
- `data/processed/research/escalating_model_eval_unseen_test.parquet`
- `data/processed/research/escalating_model_eval_safeguard_test.parquet`
- `reports/escalating_model_poc.md`
- `docs/escalating_model_threshold_sweep.md`

Comparison outputs:

- `data/processed/research/escalating_model_lightgbm_compare_summary.csv`
- `data/processed/research/escalating_model_lightgbm_compare/`
- `reports/escalating_model_lightgbm_compare.md`

The logistic regression comparison summary was generated from the previous code in a temporary worktree and written under `/private/tmp/escalating_model_logistic_compare_summary.csv`.
