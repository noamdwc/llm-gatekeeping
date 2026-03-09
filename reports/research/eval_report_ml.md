# ML Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.9872 |
| adversarial_precision | 0.9979 |
| adversarial_recall | 0.9875 |
| adversarial_f1 | 0.9926 |
| benign_precision | 0.9184 |
| benign_recall | 0.9854 |
| benign_f1 | 0.9507 |
| false_positive_rate | 0.0146 |
| false_negative_rate | 0.0125 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 957 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.9906
- Macro F1: 0.4976

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [948, 0]
  [0, 0]

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
| 0.5-0.6 | 5 | 0.546 | 0.600 |
| 0.6-0.7 | 11 | 0.649 | 0.636 |
| 0.7-0.8 | 8 | 0.748 | 0.875 |
| 0.8-0.9 | 26 | 0.855 | 0.808 |
| 0.9-1.0 | 1044 | 0.996 | 0.998 |

## Cost / Usage

- eval_scope: benign_plus_unicode_only
- nlp_rows_excluded: 725
