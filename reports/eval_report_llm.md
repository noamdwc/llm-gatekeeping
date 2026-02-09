# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.6900 |
| adversarial_precision | 0.8462 |
| adversarial_recall | 0.7765 |
| adversarial_f1 | 0.8098 |
| benign_precision | 0.1364 |
| benign_recall | 0.2000 |
| benign_f1 | 0.1622 |
| false_negative_rate | 0.2235 |
| support_adversarial | 85 |
| support_benign | 15 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.9091
- Macro F1: 0.8952

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [42, 2]
  [4, 18]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.8000
- Macro F1: 0.8624

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 3 |
| Deletion Characters | 1.00 | 0.50 | 0.67 | 4 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 6 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 7 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 3 |
| Numbers | 1.00 | 0.50 | 0.67 | 4 |
| Spaces | 1.00 | 0.83 | 0.91 | 6 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 5 |
| Unicode Tags Smuggling | 1.00 | 0.29 | 0.44 | 7 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 1 |
| Zero Width | 0.67 | 1.00 | 0.80 | 4 |
| micro avg | 0.95 | 0.80 | 0.87 | 50 |
| macro avg | 0.97 | 0.83 | 0.86 | 50 |
| weighted avg | 0.97 | 0.80 | 0.84 | 50 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.8-0.9 | 26 | 0.850 | 0.731 |
| 0.9-1.0 | 74 | 0.945 | 0.676 |
