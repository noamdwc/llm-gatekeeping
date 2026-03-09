# Hybrid Router Evaluation Report (Strict LLM Coverage)

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7488 |
| adversarial_precision | 0.9722 |
| adversarial_recall | 0.7497 |
| adversarial_f1 | 0.8466 |
| benign_precision | 0.1935 |
| benign_recall | 0.7372 |
| benign_f1 | 0.3065 |
| false_positive_rate | 0.2628 |
| false_negative_rate | 0.2503 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1682 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5755
- Macro F1: 0.4756

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [938, 0]
  [257, 30]

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
| 0.5-0.6 | 18 | 0.544 | 0.667 |
| 0.6-0.7 | 40 | 0.651 | 0.500 |
| 0.7-0.8 | 40 | 0.754 | 0.575 |
| 0.8-0.9 | 74 | 0.856 | 0.622 |
| 0.9-1.0 | 1647 | 0.989 | 0.766 |

## Cost / Usage

- routed_ml: 928
- routed_llm: 847
- routed_abstain: 44
- ml_pred_benign_routed_ml: 0
- ml_pred_benign_routed_llm: 816
- ml_pred_benign_routed_abstain: 41
- ml_pred_adversarial_routed_ml: 928
- ml_pred_adversarial_routed_llm: 31
- ml_pred_adversarial_routed_abstain: 3

## Routing Diagnostics

- total_samples: 1819
- routed_ml: 928 (0.5102)
- routed_llm: 847 (0.4656)
- routed_abstain: 44 (0.0242)
- unicode_lane_unknown_total: 0

| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |
|---------------|-----------|------------|----------------|-----------------|
| benign | 0 | 816 | 41 | 1.0000 |
| adversarial | 928 | 31 | 3 | 0.0353 |

| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |
|--------------|-------|-------------|---------------------------|
| True | 953 | 928 | 25 |
| False | 866 | 0 | 866 |

## FPR Diagnostic Views

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.2628 | All samples, abstain=adversarial |
| Abstain-excluded | 0.2574 | 44 abstain samples removed |
| Abstain rate | 0.0242 | 44/1819 samples |
