# ML Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 1.0000 |
| adversarial_precision | 1.0000 |
| adversarial_recall | 1.0000 |
| adversarial_f1 | 1.0000 |
| benign_precision | 1.0000 |
| benign_recall | 1.0000 |
| benign_f1 | 1.0000 |
| false_negative_rate | 0.0000 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 902 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 1.0000
- Macro F1: 0.5000

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [902, 0]
  [0, 0]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 1.0000
- Macro F1: 1.0000

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 82 |
| Deletion Characters | 1.00 | 1.00 | 1.00 | 82 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 82 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 82 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 82 |
| Numbers | 1.00 | 1.00 | 1.00 | 82 |
| Spaces | 1.00 | 1.00 | 1.00 | 82 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 82 |
| Unicode Tags Smuggling | 1.00 | 1.00 | 1.00 | 82 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 82 |
| Zero Width | 1.00 | 1.00 | 1.00 | 82 |
| macro avg | 1.00 | 1.00 | 1.00 | 902 |
| weighted avg | 1.00 | 1.00 | 1.00 | 902 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.8-0.9 | 7 | 0.885 | 1.000 |
| 0.9-1.0 | 989 | 0.991 | 1.000 |

## Cost / Usage

- eval_scope: benign_plus_unicode_only
- nlp_rows_excluded: 694
