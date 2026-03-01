# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7100 |
| adversarial_precision | 0.9688 |
| adversarial_recall | 0.6966 |
| adversarial_f1 | 0.8105 |
| benign_precision | 0.2500 |
| benign_recall | 0.8182 |
| benign_f1 | 0.3830 |
| false_positive_rate | 0.1818 |
| false_negative_rate | 0.3034 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 89 |
| support_benign | 11 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.3034
- Macro F1: 0.3474

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [9, 34]
  [1, 18]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 1 | 0.000 | 1.000 |
| 0.5-0.6 | 1 | 0.500 | 1.000 |
| 0.8-0.9 | 58 | 0.880 | 0.966 |
| 0.9-1.0 | 40 | 0.912 | 0.325 |
