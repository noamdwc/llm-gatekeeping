# ML Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.9839 |
| adversarial_precision | 0.9955 |
| adversarial_recall | 0.9867 |
| adversarial_f1 | 0.9911 |
| benign_precision | 0.8824 |
| benign_recall | 0.9574 |
| benign_f1 | 0.9184 |
| false_positive_rate | 0.0426 |
| false_negative_rate | 0.0133 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 902 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.9900
- Macro F1: 0.4975

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [893, 0]
  [0, 0]

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
| 0.5-0.6 | 9 | 0.534 | 0.667 |
| 0.6-0.7 | 10 | 0.653 | 0.700 |
| 0.7-0.8 | 7 | 0.761 | 0.571 |
| 0.8-0.9 | 24 | 0.856 | 0.833 |
| 0.9-1.0 | 946 | 0.996 | 0.997 |

## Cost / Usage

- eval_scope: benign_plus_unicode_only
- nlp_rows_excluded: 694
