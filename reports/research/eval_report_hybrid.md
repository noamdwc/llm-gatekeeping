# Hybrid Router Evaluation Report (Strict LLM Coverage)

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.9289 |
| adversarial_precision | 0.9652 |
| adversarial_recall | 0.9469 |
| adversarial_f1 | 0.9560 |
| benign_precision | 0.7846 |
| benign_recall | 0.8500 |
| benign_f1 | 0.8160 |
| false_positive_rate | 0.1500 |
| false_negative_rate | 0.0531 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1318 |
| support_benign | 300 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5789
- Macro F1: 0.4644

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [763, 0]
  [110, 0]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.9896
- Macro F1: 0.9945

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 70 |
| Deletion Characters | 1.00 | 0.89 | 0.94 | 70 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 70 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 70 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 70 |
| Numbers | 1.00 | 1.00 | 1.00 | 70 |
| Spaces | 1.00 | 1.00 | 1.00 | 70 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 70 |
| Unicode Tags Smuggling | 1.00 | 1.00 | 1.00 | 70 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 70 |
| Zero Width | 1.00 | 1.00 | 1.00 | 70 |
| micro avg | 1.00 | 0.99 | 0.99 | 770 |
| macro avg | 1.00 | 0.99 | 0.99 | 770 |
| weighted avg | 1.00 | 0.99 | 0.99 | 770 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 12 | 0.558 | 0.917 |
| 0.6-0.7 | 12 | 0.656 | 1.000 |
| 0.7-0.8 | 22 | 0.742 | 0.818 |
| 0.8-0.9 | 61 | 0.866 | 0.918 |
| 0.9-1.0 | 1511 | 0.991 | 0.931 |

## Cost / Usage

- routed_ml: 747
- routed_llm: 266
- routed_abstain: 5
- ml_pred_benign_routed_ml: 0
- ml_pred_benign_routed_llm: 264
- ml_pred_benign_routed_abstain: 5
- ml_pred_adversarial_routed_ml: 747
- ml_pred_adversarial_routed_llm: 2
- ml_pred_adversarial_routed_abstain: 0

## Routing Diagnostics

- total_samples: 1618
- routed_ml: 747 (0.4617)
- routed_llm: 266 (0.1644)
- routed_abstain: 5 (0.0031)
- unicode_lane_unknown_total: 0

| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |
|---------------|-----------|------------|----------------|-----------------|
| benign | 0 | 264 | 5 | 0.3132 |
| adversarial | 747 | 2 | 0 | 0.0026 |

| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |
|--------------|-------|-------------|---------------------------|
| True | 1618 | 747 | 271 |
| False | 0 | 0 | 0 |


## FPR Diagnostic Views

| View | FPR | Notes |
|------|-----|-------|
| Standard | 0.1500 | All samples, abstain=adversarial |
| Abstain-excluded | 0.1500 | 5 abstain samples removed |
| Abstain rate | 0.0031 | 5/1618 samples |
| Clean-benign | 0.0000 | 220 validated synthetic benigns only |
| Clean-benign + abstain-excluded | 0.0000 | Clean benigns, 0 abstain removed |
| Clean-benign abstain rate | 0.0000 | 0/220 clean benign samples abstained |
