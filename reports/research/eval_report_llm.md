# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.6500 |
| adversarial_precision | 0.9500 |
| adversarial_recall | 0.6404 |
| adversarial_f1 | 0.7651 |
| benign_precision | 0.2000 |
| benign_recall | 0.7273 |
| benign_f1 | 0.3137 |
| false_negative_rate | 0.3596 |
| support_adversarial | 89 |
| support_benign | 11 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.4607
- Macro F1: 0.3727

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [41, 0]
  [16, 0]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 68 | 0.500 | 0.779 |
| 0.7-0.8 | 12 | 0.750 | 0.333 |
| 0.8-0.9 | 20 | 0.839 | 0.400 |
