# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5520 |
| adversarial_precision | 0.9666 |
| adversarial_recall | 0.5339 |
| adversarial_f1 | 0.6879 |
| benign_precision | 0.1191 |
| benign_recall | 0.7737 |
| benign_f1 | 0.2064 |
| false_positive_rate | 0.2263 |
| false_negative_rate | 0.4661 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1682 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.3555
- Macro F1: 0.3325

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [592, 25]
  [275, 6]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 203 | 0.000 | 0.995 |
| 0.5-0.6 | 14 | 0.500 | 1.000 |
| 0.8-0.9 | 53 | 0.839 | 0.604 |
| 0.9-1.0 | 1549 | 0.964 | 0.488 |
