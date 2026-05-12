# Pipeline Final-Verdict Report

Metrics computed on the **final pipeline verdict** for each sample, regardless
of which stage produced it. For non-escalated rows the verdict is the cheap
classifier output (`llm_pred_*`); for rows the escalating model flagged for
escalation (calibrated_escalation_score >= 0.5) the verdict is the judge output
(`judge_final_*`).

Splits evaluated: `test`, `unseen_test`, `safeguard_test`. `unseen_val` is used
to fit the escalation-score calibrator and pick the threshold and is excluded
from this report.

## Overall

- Rows: **6027** (2958 adversarial, 3069 benign)
- Judge calls: **229** (3.80% of rows)
- Binary accuracy: **95.44%**
- Adversarial recall: **96.25%**
- Benign recall: **94.66%**
- Adversarial precision: **94.55%**

## Per-split

| split          |   rows |   judge_calls | judge_call_rate   | accuracy   | adv_recall   | benign_recall   | adv_precision   |   n_adv |   n_benign |
|:---------------|-------:|--------------:|:------------------|:-----------|:-------------|:----------------|:----------------|--------:|-----------:|
| test           |   2581 |            64 | 2.48%             | 96.09%     | 98.61%       | 91.53%          | 95.45%          |    1660 |        921 |
| unseen_test    |   1894 |           102 | 5.39%             | 92.19%     | 93.80%       | 89.80%          | 93.14%          |    1129 |        765 |
| safeguard_test |   1552 |            63 | 4.06%             | 98.32%     | 89.35%       | 99.42%          | 94.97%          |     169 |       1383 |

## Category Recall

Recall of predicted category vs labelled category (only rows where
`label_category` is populated). On `test` 1041/2581 rows and on `unseen_test`
543/1894 rows have no `label_category` — those are included in binary metrics
but excluded here. `safeguard_test` has no `label_category` at all.

| split       | category       |   n | recall   |
|:------------|:---------------|----:|:---------|
| test        | unicode_attack | 869 | 73.19%   |
| test        | nlp_attack     | 423 | 20.09%   |
| test        | benign         | 248 | 70.16%   |
| unseen_test | unicode_attack | 275 | 77.45%   |
| unseen_test | nlp_attack     | 854 | 19.20%   |
| unseen_test | benign         | 222 | 65.32%   |

## Per-attack Adversarial Recall (binary)

Counts and binary-recall by `attack_name`, aggregated across the three splits.
`judge_calls` is how many of the rows for that attack were escalated.

| attack_name            |   n | binary_recall   |   judge_calls |
|:-----------------------|----:|:----------------|--------------:|
| Alzantot               |  56 | 96.43%          |             3 |
| BAE                    | 320 | 87.81%          |            40 |
| Bert-Attack            | 111 | 94.59%          |             4 |
| Bidirectional Text     |  79 | 100.00%         |             0 |
| Deep Word Bug          |  86 | 100.00%         |             0 |
| Deletion Characters    |  79 | 100.00%         |             0 |
| Diacritcs              |  79 | 100.00%         |             0 |
| Emoji Smuggling        | 275 | 100.00%         |             0 |
| Full Width Text        |  79 | 100.00%         |             0 |
| Homoglyphs             |  79 | 100.00%         |             0 |
| Numbers                |  79 | 100.00%         |             0 |
| PWWS                   |  89 | 100.00%         |             0 |
| Pruthi                 | 138 | 98.55%          |             3 |
| Spaces                 |  79 | 100.00%         |             0 |
| TextBugger             |  81 | 100.00%         |             0 |
| TextFooler             | 396 | 92.68%          |            31 |
| Underline Accent Marks |  79 | 100.00%         |             0 |
| Unicode Tags Smuggling |  79 | 100.00%         |             0 |
| Upside Down Text       |  79 | 100.00%         |             0 |
| Zero Width             |  79 | 98.73%          |             1 |

## Judge Workload Summary

- Total rows scored: 6027
- Rows escalated to judge: 229 (3.80%)
- Pre-gate baseline (judge on every row): 6027 calls
- Reduction vs. judge-everything: **96.20%** fewer judge calls
