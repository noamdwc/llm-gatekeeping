# ML Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.9829 |
| adversarial_precision | 0.9944 |
| adversarial_recall | 0.9867 |
| adversarial_f1 | 0.9905 |
| benign_precision | 0.8812 |
| benign_recall | 0.9468 |
| benign_f1 | 0.9128 |
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
| 0.5-0.6 | 9 | 0.531 | 0.556 |
| 0.6-0.7 | 11 | 0.661 | 0.636 |
| 0.7-0.8 | 6 | 0.767 | 0.667 |
| 0.8-0.9 | 24 | 0.857 | 0.833 |
| 0.9-1.0 | 946 | 0.996 | 0.997 |

## Cost / Usage

- eval_scope: benign_plus_unicode_only
- nlp_rows_excluded: 694
