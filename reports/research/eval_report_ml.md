# ML Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.9888 |
| adversarial_precision | 1.0000 |
| adversarial_recall | 0.9844 |
| adversarial_f1 | 0.9921 |
| benign_precision | 0.9615 |
| benign_recall | 1.0000 |
| benign_f1 | 0.9804 |
| false_positive_rate | 0.0000 |
| false_negative_rate | 0.0156 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 770 |
| support_benign | 300 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.9909
- Macro F1: 0.4977

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [763, 0]
  [0, 0]

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
| 0.5-0.6 | 6 | 0.545 | 0.333 |
| 0.6-0.7 | 7 | 0.659 | 0.429 |
| 0.7-0.8 | 10 | 0.741 | 0.800 |
| 0.8-0.9 | 9 | 0.873 | 1.000 |
| 0.9-1.0 | 1038 | 0.997 | 0.998 |

## Cost / Usage

- eval_scope: benign_plus_unicode_only
- nlp_rows_excluded: 548

