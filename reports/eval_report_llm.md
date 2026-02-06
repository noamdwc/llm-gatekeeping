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

- Accuracy: 0.8939
- Macro F1: 0.8793

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [41, 3]
  [4, 18]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.7800
- Macro F1: 0.8422

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 3 |
| Deletion Characters | 1.00 | 0.50 | 0.67 | 4 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 6 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 7 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 3 |
| Numbers | 1.00 | 0.25 | 0.40 | 4 |
| Spaces | 1.00 | 0.83 | 0.91 | 6 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 5 |
| Unicode Tags Smuggling | 0.67 | 0.29 | 0.40 | 7 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 1 |
| Zero Width | 0.80 | 1.00 | 0.89 | 4 |
| micro avg | 0.95 | 0.78 | 0.86 | 50 |
| macro avg | 0.95 | 0.81 | 0.84 | 50 |
| weighted avg | 0.94 | 0.78 | 0.82 | 50 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.8-0.9 | 25 | 0.850 | 0.760 |
| 0.9-1.0 | 75 | 0.947 | 0.667 |
