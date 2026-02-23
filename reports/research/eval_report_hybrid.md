# Hybrid Router Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.6213 |
| adversarial_precision | 0.9928 |
| adversarial_recall | 0.6034 |
| adversarial_f1 | 0.7506 |
| benign_precision | 0.1208 |
| benign_recall | 0.9255 |
| benign_f1 | 0.2138 |
| false_negative_rate | 0.3966 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1596 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5702
- Macro F1: 0.5124

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [893, 0]
  [33, 17]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.9900
- Macro F1: 0.9947

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 82 |
| Deletion Characters | 1.00 | 0.89 | 0.94 | 82 |
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
| 0.5-0.6 | 42 | 0.543 | 0.476 |
| 0.6-0.7 | 57 | 0.648 | 0.439 |
| 0.7-0.8 | 61 | 0.752 | 0.262 |
| 0.8-0.9 | 108 | 0.859 | 0.278 |
| 0.9-1.0 | 1422 | 0.987 | 0.674 |

## Cost / Usage

- routed_ml: 1643
- routed_llm: 47
