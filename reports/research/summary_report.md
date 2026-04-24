# Summary Report

- split: `test`

## Main Dataset

| Model | Rows | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|-------|------|----------|--------|-----------|-----|-----|
| ML (unicode scope) | 1070 | 0.9888 | 0.9921 | 0.9804 | 0.0000 | 0.0156 |
| Hybrid | 1618 | 0.9580 | 0.9743 | 0.8844 | 0.1333 | 0.0212 |
| LLM | 1618 | 0.8072 | 0.8709 | 0.6195 | 0.1533 | 0.2018 |

- routing: abstain=180, deberta=600, llm=91, ml=747

## FPR Diagnostic Views (Hybrid)

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.1333 | All samples, abstain=adversarial |
| Abstain-excluded | 0.0827 | 180 abstain samples removed |
| Abstain rate | 0.1112 | 180/1618 samples |
| Clean-benign | 0.0000 | 220 validated synthetic benigns only |
| Clean-benign + abstain-excluded | 0.0000 | Clean benigns, 0 abstain removed |
| Clean-benign abstain rate | 0.0000 | 0/220 clean benign samples abstained |

## Benign Risk Model (train-on-val, eval-on-test)

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Isotonic (margin only) | 0.5000 | 0.3420 | 0.2253 |
| Logistic (all features) | 0.9425 | 0.8684 | 0.0813 |

## External Combined (Unseen) Progress

| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR | Support Adv | Support Benign |
|------------|-------|----------|--------|-----------|-----|-----|-------------|----------------|
| 2427 | 0.3490 | 0.7672 | 0.6231 | 0.8316 | 0.1171 | 0.4486 | 847 | 1580 |

## External Dataset Breakdown

| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR |
|---------|------|-------|----------|--------|-----------|-----|-----|
| deepset | 116 | 0.5172 | 0.6121 | 0.6341 | 0.5872 | 0.4286 | 0.3500 |
| jackhhao | 262 | 0.5305 | 0.8817 | 0.8905 | 0.8714 | 0.1463 | 0.0935 |
| safeguard | 2049 | 0.3163 | 0.7613 | 0.5526 | 0.8373 | 0.1021 | 0.5340 |

## External Baseline Comparison

## external_deepset

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1574, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0005, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.5086 | 0.4560 | 0.5308 | 0.0667 | 0.9821 | 0.0179 | 0.9333 | 4 | 1 | 55 | 56 |
| Our Hybrid | - | 0.6121 | - | - | 0.6500 | 0.5714 | 0.4286 | 0.3500 | 39 | 24 | 32 | 21 |
| Sentinel v2 | default (0.5000) | 0.8793 | 0.9613 | 0.9734 | 0.7833 | 0.9821 | 0.0179 | 0.2167 | 47 | 1 | 55 | 13 |
| Sentinel v2 | low_fnr (0.1574) | 0.9224 | 0.9613 | 0.9734 | 0.8667 | 0.9821 | 0.0179 | 0.1333 | 52 | 1 | 55 | 8 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.4828 | 0.9613 | 0.9734 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 56 | 60 |
| ProtectAI v2 | default (0.5000) | 0.6724 | 0.9015 | 0.9290 | 0.3667 | 1.0000 | 0.0000 | 0.6333 | 22 | 0 | 56 | 38 |
| ProtectAI v2 | low_fnr (0.0005) | 0.7586 | 0.9015 | 0.9290 | 0.5333 | 1.0000 | 0.0000 | 0.4667 | 32 | 0 | 56 | 28 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.4828 | 0.9015 | 0.9290 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 56 | 60 |

## external_jackhhao

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1574, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0005, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.4733 | 0.3501 | 0.4509 | 0.0072 | 1.0000 | 0.0000 | 0.9928 | 1 | 0 | 123 | 138 |
| Our Hybrid | - | 0.8817 | - | - | 0.9065 | 0.8537 | 0.1463 | 0.0935 | 126 | 18 | 105 | 13 |
| Sentinel v2 | default (0.5000) | 0.9733 | 0.9970 | 0.9975 | 0.9712 | 0.9756 | 0.0244 | 0.0288 | 135 | 3 | 120 | 4 |
| Sentinel v2 | low_fnr (0.1574) | 0.9771 | 0.9970 | 0.9975 | 0.9784 | 0.9756 | 0.0244 | 0.0216 | 136 | 3 | 120 | 3 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.4695 | 0.9970 | 0.9975 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 123 | 139 |
| ProtectAI v2 | default (0.5000) | 0.9084 | 0.9792 | 0.9658 | 0.8417 | 0.9837 | 0.0163 | 0.1583 | 117 | 2 | 121 | 22 |
| ProtectAI v2 | low_fnr (0.0005) | 0.9542 | 0.9792 | 0.9658 | 0.9424 | 0.9675 | 0.0325 | 0.0576 | 131 | 4 | 119 | 8 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.4695 | 0.9792 | 0.9658 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 123 | 139 |

## external_safeguard

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1574, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0005, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.6842 | 0.6942 | 0.4564 | 0.0046 | 0.9986 | 0.0014 | 0.9954 | 3 | 2 | 1399 | 645 |
| Our Hybrid | - | 0.7613 | - | - | 0.4660 | 0.8979 | 0.1021 | 0.5340 | 302 | 143 | 1258 | 346 |
| Sentinel v2 | default (0.5000) | 0.9980 | 0.9996 | 0.9995 | 0.9969 | 0.9986 | 0.0014 | 0.0031 | 646 | 2 | 1399 | 2 |
| Sentinel v2 | low_fnr (0.1574) | 0.9937 | 0.9996 | 0.9995 | 0.9985 | 0.9914 | 0.0086 | 0.0015 | 647 | 12 | 1389 | 1 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.6837 | 0.9996 | 0.9995 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 1401 | 648 |
| ProtectAI v2 | default (0.5000) | 0.9488 | 0.9920 | 0.9879 | 0.8410 | 0.9986 | 0.0014 | 0.1590 | 545 | 2 | 1399 | 103 |
| ProtectAI v2 | low_fnr (0.0005) | 0.9683 | 0.9920 | 0.9879 | 0.9414 | 0.9807 | 0.0193 | 0.0586 | 610 | 27 | 1374 | 38 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.6837 | 0.9920 | 0.9879 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 1401 | 648 |

## test

### Sentinel v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.1574, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

### ProtectAI v2 Threshold Tuning
- default: threshold=0.5000, target_metric=default, target=-, constraint_met=True
- low_fnr: threshold=0.0005, target_metric=false_negative_rate, target=0.0200, constraint_met=True
- bounded_fpr: threshold=>1.0, target_metric=false_positive_rate, target=0.0500, constraint_met=True

| Model | Threshold | Accuracy | AUROC | AUPRC | Adv Recall | Benign Recall | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our ML | - | 0.6545 | 0.9236 | 0.9817 | 0.5759 | 1.0000 | 0.0000 | 0.4241 | 759 | 0 | 300 | 559 |
| Our Hybrid | - | 0.9580 | - | - | 0.9788 | 0.8667 | 0.1333 | 0.0212 | 1290 | 40 | 260 | 28 |
| Sentinel v2 | default (0.5000) | 0.7484 | 0.2108 | 0.8880 | 0.7931 | 0.0189 | 0.9811 | 0.2069 | 686 | 52 | 1 | 179 |
| Sentinel v2 | low_fnr (0.1574) | 0.9052 | 0.2108 | 0.8880 | 0.9595 | 0.0189 | 0.9811 | 0.0405 | 830 | 52 | 1 | 35 |
| Sentinel v2 | bounded_fpr (>1.0) | 0.0577 | 0.2108 | 0.8880 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 53 | 865 |
| ProtectAI v2 | default (0.5000) | 0.6721 | 0.3525 | 0.9105 | 0.7029 | 0.1698 | 0.8302 | 0.2971 | 608 | 44 | 9 | 257 |
| ProtectAI v2 | low_fnr (0.0005) | 0.9150 | 0.3525 | 0.9105 | 0.9665 | 0.0755 | 0.9245 | 0.0335 | 836 | 49 | 4 | 29 |
| ProtectAI v2 | bounded_fpr (>1.0) | 0.0577 | 0.3525 | 0.9105 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0 | 53 | 865 |

## Skipped Datasets

No matching research parquet was available for these baseline datasets: test_unseen
