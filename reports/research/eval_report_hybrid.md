# Hybrid Router Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.6219 |
| adversarial_precision | 0.9938 |
| adversarial_recall | 0.6034 |
| adversarial_f1 | 0.7509 |
| benign_precision | 0.1221 |
| benign_recall | 0.9362 |
| benign_f1 | 0.2160 |
| false_positive_rate | 0.0638 |
| false_negative_rate | 0.3966 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1596 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5702
- Macro F1: 0.5127

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [893, 0]
  [32, 17]

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
| 0.5-0.6 | 43 | 0.547 | 0.465 |
| 0.6-0.7 | 56 | 0.646 | 0.446 |
| 0.7-0.8 | 63 | 0.753 | 0.270 |
| 0.8-0.9 | 106 | 0.860 | 0.283 |
| 0.9-1.0 | 1422 | 0.987 | 0.674 |

## Cost / Usage

- routed_ml: 1643
- routed_llm: 47
- ml_pred_benign_routed_ml: 694
- ml_pred_benign_routed_llm: 45
- ml_pred_adversarial_routed_ml: 949
- ml_pred_adversarial_routed_llm: 2

## Routing Diagnostics

- total_samples: 1690
- routed_ml: 1643 (0.9722)
- routed_llm: 47 (0.0278)

| ml_pred_label | routed_ml | routed_llm | escalation_rate |
|---------------|-----------|------------|-----------------|
| benign | 694 | 45 | 0.0609 |
| adversarial | 949 | 2 | 0.0021 |
