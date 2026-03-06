# Hybrid Router Evaluation Report (Strict LLM Coverage)

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7083 |
| adversarial_precision | 0.9791 |
| adversarial_recall | 0.7061 |
| adversarial_f1 | 0.8205 |
| benign_precision | 0.1299 |
| benign_recall | 0.7447 |
| benign_f1 | 0.2212 |
| false_positive_rate | 0.2553 |
| false_negative_rate | 0.2939 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1596 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5664
- Macro F1: 0.4754

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [884, 0]
  [190, 20]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.9889
- Macro F1: 0.9941

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 82 |
| Deletion Characters | 1.00 | 0.88 | 0.94 | 82 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 82 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 82 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 82 |
| Numbers | 1.00 | 1.00 | 1.00 | 82 |
| Spaces | 1.00 | 1.00 | 1.00 | 82 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 82 |
| Unicode Tags Smuggling | 1.00 | 1.00 | 1.00 | 82 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 82 |
| Zero Width | 1.00 | 1.00 | 1.00 | 82 |
| micro avg | 1.00 | 0.99 | 0.99 | 902 |
| macro avg | 1.00 | 0.99 | 0.99 | 902 |
| weighted avg | 1.00 | 0.99 | 0.99 | 902 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 43 | 0.547 | 0.488 |
| 0.6-0.7 | 56 | 0.646 | 0.321 |
| 0.7-0.8 | 63 | 0.753 | 0.397 |
| 0.8-0.9 | 106 | 0.860 | 0.491 |
| 0.9-1.0 | 1422 | 0.987 | 0.760 |

## Cost / Usage

- routed_ml: 886
- routed_llm: 760
- routed_abstain: 44
- ml_pred_benign_routed_ml: 0
- ml_pred_benign_routed_llm: 703
- ml_pred_benign_routed_abstain: 36
- ml_pred_adversarial_routed_ml: 886
- ml_pred_adversarial_routed_llm: 57
- ml_pred_adversarial_routed_abstain: 8

## Routing Diagnostics

- total_samples: 1690
- routed_ml: 886 (0.5243)
- routed_llm: 760 (0.4497)
- routed_abstain: 44 (0.0260)
- unicode_lane_unknown_total: 0

| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |
|---------------|-----------|------------|----------------|-----------------|
| benign | 0 | 703 | 36 | 1.0000 |
| adversarial | 886 | 57 | 8 | 0.0683 |

| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |
|--------------|-------|-------------|---------------------------|
| True | 1690 | 886 | 804 |
| False | 0 | 0 | 0 |

## FPR Diagnostic Views

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.2553 | All samples, abstain=adversarial |
| Abstain-excluded | 0.2391 | 44 abstain samples removed |
| Abstain rate | 0.0260 | 44/1690 samples |
