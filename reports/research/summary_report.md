# Summary Report

- split: `test`

## Main Dataset

| Model | Rows | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|-------|------|----------|--------|-----------|-----|-----|
| ML (unicode scope) | 1094 | 0.9872 | 0.9926 | 0.9507 | 0.0146 | 0.0125 |
| Hybrid | 1819 | 0.8395 | 0.9085 | 0.3482 | 0.4307 | 0.1385 |
| LLM | 1819 | 0.5618 | 0.6968 | 0.2101 | 0.2263 | 0.4554 |

- routing: abstain=19, llm=872, ml=928

## FPR Diagnostic Views (Hybrid)

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.4307 | All samples, abstain=adversarial |
| Abstain-excluded | 0.4265 | 19 abstain samples removed |
| Abstain rate | 0.0104 | 19/1819 samples |

## Benign Risk Model (train-on-val, eval-on-test)

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Isotonic (margin only) | 0.5000 | 0.8134 | 0.1526 |
| Logistic (all features) | 0.7558 | 0.9102 | 0.1094 |

## External Combined (Unseen) Progress

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR | Support Adv | Support Benign |
|------------|-------|----------|--------|-----------|-----|-----|-------------|----------------|
| 2427 | 0.3490 | 0.7713 | 0.5186 | 0.8500 | 0.0044 | 0.6470 | 847 | 1580 |

## External Dataset Breakdown

| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|---------|------|-------|----------|--------|-----------|-----|-----|
| deepset | 116 | 0.5172 | 0.5431 | 0.2319 | 0.6748 | 0.0179 | 0.8667 |
| jackhhao | 262 | 0.5305 | 0.7595 | 0.7097 | 0.7948 | 0.0081 | 0.4460 |
| safeguard | 2049 | 0.3163 | 0.7857 | 0.4937 | 0.8641 | 0.0036 | 0.6698 |

## External Baseline Comparison

## external_deepset

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1352, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0001, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.4914 | 0.4476 | 0.5241 | 0.1500 | 0.8571 | 0.1429 | 0.8500 | 9 | 8 | 48 | 51 |
| Our Hybrid | - | 0.5431 | - | - | 0.1333 | 0.9821 | 0.0179 | 0.8667 | 8 | 1 | 55 | 52 |
| Sentinel v2 | default (0.5000) | 0.8793 | 0.9613 | 0.9734 | 0.7833 | 0.9821 | 0.0179 | 0.2167 | 47 | 1 | 55 | 13 |
| Sentinel v2 | low_fnr (0.1352) | 0.9224 | 0.9613 | 0.9734 | 0.8667 | 0.9821 | 0.0179 | 0.1333 | 52 | 1 | 55 | 8 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.4828 | 0.9613 | 0.9734 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 56 | 60 |
| ProtectAI v2 | default (0.5000) | 0.6724 | 0.9015 | 0.9290 | 0.3667 | 1.0000 | 0.0000 | 0.6333 | 22 | 0 | 56 | 38 |
| ProtectAI v2 | low_fnr (0.0001) | 0.8103 | 0.9015 | 0.9290 | 0.6500 | 0.9821 | 0.0179 | 0.3500 | 39 | 1 | 55 | 21 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.4828 | 0.9015 | 0.9290 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 56 | 60 |

## external_jackhhao

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1352, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0001, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.4695 | 0.2844 | 0.4077 | 0.0144 | 0.9837 | 0.0163 | 0.9856 | 2 | 2 | 121 | 137 |
| Our Hybrid | - | 0.7595 | - | - | 0.5540 | 0.9919 | 0.0081 | 0.4460 | 77 | 1 | 122 | 62 |
| Sentinel v2 | default (0.5000) | 0.9733 | 0.9970 | 0.9975 | 0.9712 | 0.9756 | 0.0244 | 0.0288 | 135 | 3 | 120 | 4 |
| Sentinel v2 | low_fnr (0.1352) | 0.9809 | 0.9970 | 0.9975 | 0.9856 | 0.9756 | 0.0244 | 0.0144 | 137 | 3 | 120 | 2 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.4695 | 0.9970 | 0.9975 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 123 | 139 |
| ProtectAI v2 | default (0.5000) | 0.9084 | 0.9792 | 0.9658 | 0.8417 | 0.9837 | 0.0163 | 0.1583 | 117 | 2 | 121 | 22 |
| ProtectAI v2 | low_fnr (0.0001) | 0.9427 | 0.9792 | 0.9658 | 0.9640 | 0.9187 | 0.0813 | 0.0360 | 134 | 10 | 113 | 5 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.4695 | 0.9792 | 0.9658 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 123 | 139 |

## external_safeguard

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1352, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0001, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.6784 | 0.5813 | 0.3488 | 0.0077 | 0.9886 | 0.0114 | 0.9923 | 5 | 16 | 1385 | 643 |
| Our Hybrid | - | 0.7857 | - | - | 0.3302 | 0.9964 | 0.0036 | 0.6698 | 214 | 5 | 1396 | 434 |
| Sentinel v2 | default (0.5000) | 0.9980 | 0.9996 | 0.9995 | 0.9969 | 0.9986 | 0.0014 | 0.0031 | 646 | 2 | 1399 | 2 |
| Sentinel v2 | low_fnr (0.1352) | 0.9927 | 0.9996 | 0.9995 | 0.9985 | 0.9900 | 0.0100 | 0.0015 | 647 | 14 | 1387 | 1 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.6837 | 0.9996 | 0.9995 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 1401 | 648 |
| ProtectAI v2 | default (0.5000) | 0.9488 | 0.9920 | 0.9879 | 0.8410 | 0.9986 | 0.0014 | 0.1590 | 545 | 2 | 1399 | 103 |
| ProtectAI v2 | low_fnr (0.0001) | 0.9590 | 0.9920 | 0.9879 | 0.9707 | 0.9536 | 0.0464 | 0.0293 | 629 | 65 | 1336 | 19 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.6837 | 0.9920 | 0.9879 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 1401 | 648 |

## test

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1352, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0001, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.6020 | 0.8120 | 0.9829 | 0.5707 | 0.9854 | 0.0146 | 0.4293 | 960 | 2 | 135 | 722 |
| Our Hybrid | - | 0.8395 | - | - | 0.8615 | 0.5693 | 0.4307 | 0.1385 | 1449 | 59 | 78 | 233 |
| Sentinel v2 | default (0.5000) | 0.7576 | 0.4206 | 0.8908 | 0.7979 | 0.2628 | 0.7372 | 0.2021 | 1342 | 101 | 36 | 340 |
| Sentinel v2 | low_fnr (0.1352) | 0.9175 | 0.4206 | 0.8908 | 0.9715 | 0.2555 | 0.7445 | 0.0285 | 1634 | 102 | 35 | 48 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.0753 | 0.4206 | 0.8908 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 137 | 1682 |
| ProtectAI v2 | default (0.5000) | 0.6663 | 0.5090 | 0.9102 | 0.6908 | 0.3650 | 0.6350 | 0.3092 | 1162 | 87 | 50 | 520 |
| ProtectAI v2 | low_fnr (0.0001) | 0.9263 | 0.5090 | 0.9102 | 0.9792 | 0.2774 | 0.7226 | 0.0208 | 1647 | 99 | 38 | 35 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.0753 | 0.5090 | 0.9102 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 137 | 1682 |

## Skipped Datasets

No matching research parquet was available for these baseline datasets: test_unseen
