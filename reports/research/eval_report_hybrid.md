# Hybrid Router Evaluation Report (Strict LLM Coverage)

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.8395 |
| adversarial_precision | 0.9609 |
| adversarial_recall | 0.8615 |
| adversarial_f1 | 0.9085 |
| benign_precision | 0.2508 |
| benign_recall | 0.5693 |
| benign_f1 | 0.3482 |
| false_positive_rate | 0.4307 |
| false_negative_rate | 0.1385 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1682 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5595
- Macro F1: 0.4356

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [938, 0]
  [279, 3]

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
| 0.5-0.6 | 18 | 0.544 | 0.722 |
| 0.6-0.7 | 40 | 0.651 | 0.875 |
| 0.7-0.8 | 40 | 0.754 | 0.750 |
| 0.8-0.9 | 74 | 0.856 | 0.757 |
| 0.9-1.0 | 1647 | 0.989 | 0.846 |

## Cost / Usage

- routed_ml: 928
- routed_llm: 872
- routed_abstain: 19
- ml_pred_benign_routed_ml: 0
- ml_pred_benign_routed_llm: 838
- ml_pred_benign_routed_abstain: 19
- ml_pred_adversarial_routed_ml: 928
- ml_pred_adversarial_routed_llm: 34
- ml_pred_adversarial_routed_abstain: 0

## Routing Diagnostics

- total_samples: 1819
- routed_ml: 928 (0.5102)
- routed_llm: 872 (0.4794)
- routed_abstain: 19 (0.0104)
- unicode_lane_unknown_total: 0

| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |
|---------------|-----------|------------|----------------|-----------------|
| benign | 0 | 838 | 19 | 1.0000 |
| adversarial | 928 | 34 | 0 | 0.0353 |

| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |
|--------------|-------|-------------|---------------------------|
| True | 1819 | 928 | 891 |
| False | 0 | 0 | 0 |

## FPR Diagnostic Views

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.4307 | All samples, abstain=adversarial |
| Abstain-excluded | 0.4265 | 19 abstain samples removed |
| Abstain rate | 0.0104 | 19/1819 samples |
