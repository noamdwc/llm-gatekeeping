# Pipeline Final-Verdict Report

This is the canonical evaluation report for the escalation-model path. For each row, the final verdict is the cheap Colab/local LLM classifier output unless `hybrid.escalating_model` routes the row to the judge and a judge final label is available.

## Historical Correction

This branch was originally described as risk-model work, but the implemented gate is the escalation model. The risk model remains the separate abstain-resolution path; the escalation model owns judge-call thresholding and final-verdict evaluation.

## Run Metadata

| Setting | Value |
|---|---|
| Escalation gate threshold | `0.5` |
| Escalation-score column | `calibrated_escalation_score` when present, else `escalation_score` |
| Calibration method | `sigmoid` |
| Escalating model artifact | `data/processed/models/escalating_model.pkl` |
| Selected operating point | `0.5`, frozen for this POC canonical path |

Configured external datasets are part of the canonical final-verdict path, and their judged artifacts are required by default under `data/processed/predictions_external/*_colab_local_judged.parquet`. Missing configured judged artifacts fail before report generation rather than being excluded or mixed with research-only numbers.

## Overall

- Rows: **6405** (3157 adv, 3248 benign)
- Judge calls: **302** (4.72%)
- Binary accuracy: **94.61%**
- Adversarial recall: **94.39%**
- Benign recall: **94.83%**
- Adversarial precision: **94.66%**

| name | rows | judge_calls | judge_rate | accuracy | adv_recall | benign_recall | adv_precision | n_adv | n_benign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 2581 | 64 | 2.48% | 96.09% | 98.61% | 91.53% | 95.45% | 1660 | 921 |
| unseen_test | 1894 | 102 | 5.39% | 92.19% | 93.80% | 89.80% | 93.14% | 1129 | 765 |
| safeguard_test | 1552 | 63 | 4.06% | 98.32% | 89.35% | 99.42% | 94.97% | 169 | 1383 |
| external_deepset | 116 | 34 | 29.31% | 60.34% | 25.00% | 98.21% | 93.75% | 60 | 56 |
| external_jackhhao | 262 | 39 | 14.89% | 90.84% | 84.89% | 97.56% | 97.52% | 139 | 123 |

## Internal Splits

- Rows: **6027** (2958 adv, 3069 benign)
- Judge calls: **229** (3.80%)
- Binary accuracy: **95.44%**
- Adversarial recall: **96.25%**
- Benign recall: **94.66%**
- Adversarial precision: **94.55%**

| name | rows | judge_calls | judge_rate | accuracy | adv_recall | benign_recall | adv_precision | n_adv | n_benign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 2581 | 64 | 2.48% | 96.09% | 98.61% | 91.53% | 95.45% | 1660 | 921 |
| unseen_test | 1894 | 102 | 5.39% | 92.19% | 93.80% | 89.80% | 93.14% | 1129 | 765 |
| safeguard_test | 1552 | 63 | 4.06% | 98.32% | 89.35% | 99.42% | 94.97% | 169 | 1383 |

## External Datasets

- Rows: **378** (199 adv, 179 benign)
- Judge calls: **73** (19.31%)
- Binary accuracy: **81.48%**
- Adversarial recall: **66.83%**
- Benign recall: **97.77%**
- Adversarial precision: **97.08%**

| name | rows | judge_calls | judge_rate | accuracy | adv_recall | benign_recall | adv_precision | n_adv | n_benign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external_deepset | 116 | 34 | 29.31% | 60.34% | 25.00% | 98.21% | 93.75% | 60 | 56 |
| external_jackhhao | 262 | 39 | 14.89% | 90.84% | 84.89% | 97.56% | 97.52% | 139 | 123 |

## Judge Workload Summary

- Total rows scored: 6405
- Rows escalated to judge: 302 (4.72%)
- Reduction vs. judge-everything: **95.28%** fewer judge calls
