# Hybrid Router Evaluation Report (Strict LLM Coverage)

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7367 |
| adversarial_precision | 0.9755 |
| adversarial_recall | 0.7337 |
| adversarial_f1 | 0.8375 |
| benign_precision | 0.1913 |
| benign_recall | 0.7737 |
| benign_f1 | 0.3068 |
| false_positive_rate | 0.2263 |
| false_negative_rate | 0.2663 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1682 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5606
- Macro F1: 0.4402

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [937, 1]
  [275, 6]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.9916
- Macro F1: 0.9956

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 87 |
| Deletion Characters | 1.00 | 0.91 | 0.95 | 87 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 87 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 87 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 87 |
| Numbers | 1.00 | 1.00 | 1.00 | 87 |
| Spaces | 1.00 | 1.00 | 1.00 | 87 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 87 |
| Unicode Tags Smuggling | 1.00 | 1.00 | 1.00 | 87 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 87 |
| Zero Width | 1.00 | 1.00 | 1.00 | 87 |
| micro avg | 1.00 | 0.99 | 1.00 | 957 |
| macro avg | 1.00 | 0.99 | 1.00 | 957 |
| weighted avg | 1.00 | 0.99 | 1.00 | 957 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 18 | 0.544 | 0.611 |
| 0.6-0.7 | 40 | 0.651 | 0.500 |
| 0.7-0.8 | 40 | 0.754 | 0.550 |
| 0.8-0.9 | 74 | 0.856 | 0.541 |
| 0.9-1.0 | 1647 | 0.989 | 0.757 |

## Cost / Usage

- routed_ml: 928
- routed_llm: 868
- routed_abstain: 23
- ml_pred_benign_routed_ml: 0
- ml_pred_benign_routed_llm: 836
- ml_pred_benign_routed_abstain: 21
- ml_pred_adversarial_routed_ml: 928
- ml_pred_adversarial_routed_llm: 32
- ml_pred_adversarial_routed_abstain: 2

## Routing Diagnostics

- total_samples: 1819
- routed_ml: 928 (0.5102)
- routed_llm: 868 (0.4772)
- routed_abstain: 23 (0.0126)
- unicode_lane_unknown_total: 0

| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |
|---------------|-----------|------------|----------------|-----------------|
| benign | 0 | 836 | 21 | 1.0000 |
| adversarial | 928 | 32 | 2 | 0.0353 |

| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |
|--------------|-------|-------------|---------------------------|
| True | 1819 | 928 | 891 |
| False | 0 | 0 | 0 |

## FPR Diagnostic Views

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.2263 | All samples, abstain=adversarial |
| Abstain-excluded | 0.2263 | 23 abstain samples removed |
| Abstain rate | 0.0126 | 23/1819 samples |
