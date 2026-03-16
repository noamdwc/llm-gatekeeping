# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.6873 |
| adversarial_precision | 0.9562 |
| adversarial_recall | 0.6457 |
| adversarial_f1 | 0.7708 |
| benign_precision | 0.3585 |
| benign_recall | 0.8700 |
| benign_f1 | 0.5078 |
| false_positive_rate | 0.1300 |
| false_negative_rate | 0.3543 |
| uncertain_rate | 0.0000 |
| judge_override_rate | 0.8103 |
| support_adversarial | 1318 |
| support_benign | 300 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.3998
- Macro F1: 0.3318

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [525, 20]
  [304, 2]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 179 | 0.000 | 1.000 |
| 0.5-0.6 | 4 | 0.500 | 1.000 |
| 0.8-0.9 | 29 | 0.841 | 0.621 |
| 0.9-1.0 | 1406 | 0.976 | 0.648 |
