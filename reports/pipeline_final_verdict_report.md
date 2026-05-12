# Pipeline Final-Verdict Report

Metrics computed on the **final pipeline verdict** for each sample, regardless
of which stage produced it. For non-escalated rows the verdict is the cheap
classifier output (`llm_pred_*`); for rows the escalating model flagged for
escalation, the verdict is the judge output (`judge_final_*`).

## Run Metadata

| Setting | Value |
|---|---|
| Escalation gate threshold (`hybrid.escalating_model.judge_threshold`) | `0.5` |
| Escalation-score column compared to threshold | `calibrated_escalation_score` (all splits) |
| Calibration method (`hybrid.escalating_model.calibration_method`) | `sigmoid` |
| Calibrator fit on | `unseen_val` calibration half (post-score split, seed=`splits.random_seed`) |
| Escalating model artifact | `data/processed/models/escalating_model.pkl` (LightGBM) |
| Cheap classifier model | `meta/llama-3.1-8b-instruct` (provider `transformers-local`) |
| Judge model | `meta/llama-3.1-70b-instruct` |
| Judge rate-limit (this run) | `--max-concurrency 1 --target-rpm 2 --cooldown-on-429 60` |
| LLM `judge_confidence_threshold` (legacy, unused for this gate) | `0.8` |

Splits / datasets evaluated:
- **Internal:** `test`, `unseen_test`, `safeguard_test` (`unseen_val` is used for calibration/threshold selection and is excluded).
- **External:** `deepset` (`deepset/prompt-injections`), `jackhhao` (`jackhhao/jailbreak-classification`).

## Overall (Internal + External)

- Rows: **6405** (3157 adv, 3248 benign)
- Judge calls: **254** (3.97% of rows)
- Binary accuracy: **94.66%**
- Adversarial recall: **94.46%**
- Benign recall: **94.86%**
- Adversarial precision: **94.70%**

## Internal Splits

### Aggregated

- Rows: **6027** (2958 adv, 3069 benign)
- Judge calls: **229** (3.80% of rows)
- Binary accuracy: **95.44%**
- Adversarial recall: **96.25%**
- Benign recall: **94.66%**
- Adversarial precision: **94.55%**

### Per-split

| name           |   rows |   judge_calls | judge_rate   | accuracy   | adv_recall   | benign_recall   | adv_precision   |   n_adv |   n_benign |
|:---------------|-------:|--------------:|:-------------|:-----------|:-------------|:----------------|:----------------|--------:|-----------:|
| test           |   2581 |            64 | 2.48%        | 96.09%     | 98.61%       | 91.53%          | 95.45%          |    1660 |        921 |
| unseen_test    |   1894 |           102 | 5.39%        | 92.19%     | 93.80%       | 89.80%          | 93.14%          |    1129 |        765 |
| safeguard_test |   1552 |            63 | 4.06%        | 98.32%     | 89.35%       | 99.42%          | 94.97%          |     169 |       1383 |

## External Datasets

### Aggregated

- Rows: **378** (199 adv, 179 benign)
- Judge calls: **25** (6.61% of rows)
- Binary accuracy: **82.28%**
- Adversarial recall: **67.84%**
- Benign recall: **98.32%**
- Adversarial precision: **97.83%**

### Per-dataset

| name              |   rows |   judge_calls | judge_rate   | accuracy   | adv_recall   | benign_recall   | adv_precision   |   n_adv |   n_benign |
|:------------------|-------:|--------------:|:-------------|:-----------|:-------------|:----------------|:----------------|--------:|-----------:|
| external_deepset  |    116 |            14 | 12.07%       | 56.03%     | 16.67%       | 98.21%          | 90.91%          |      60 |         56 |
| external_jackhhao |    262 |            11 | 4.20%        | 93.89%     | 89.93%       | 98.37%          | 98.43%          |     139 |        123 |

## Category Recall (Internal Only)

Recall of predicted category vs labelled category. On `test` 1041/2581 rows and on
`unseen_test` 543/1894 rows have no `label_category` — those are included in
binary metrics but excluded here. `safeguard_test` has no `label_category` at all.
External datasets use a binary `label_category` (adversarial/benign) and are not
included in this table.

| split       | category       |   n | recall   |
|:------------|:---------------|----:|:---------|
| test        | unicode_attack | 869 | 73.19%   |
| test        | nlp_attack     | 423 | 20.09%   |
| test        | benign         | 248 | 70.16%   |
| unseen_test | unicode_attack | 275 | 77.45%   |
| unseen_test | nlp_attack     | 854 | 19.20%   |
| unseen_test | benign         | 222 | 65.32%   |

## Per-attack Adversarial Recall (Internal)

Counts and binary-recall by `attack_name`, aggregated across the three internal
splits. `judge_calls` is how many of the rows for that attack were escalated.

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

- Total rows scored: 6405
- Rows escalated to judge: 254 (3.97%)
- Internal subset: 229/6027 (3.80%)
- External subset: 25/378 (6.61%)
- Pre-gate baseline (judge on every row): 6405 calls
- Reduction vs. judge-everything: **96.03%** fewer judge calls
