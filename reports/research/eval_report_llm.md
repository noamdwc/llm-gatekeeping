# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5618 |
| adversarial_precision | 0.9673 |
| adversarial_recall | 0.5446 |
| adversarial_f1 | 0.6968 |
| benign_precision | 0.1216 |
| benign_recall | 0.7737 |
| benign_f1 | 0.2101 |
| false_positive_rate | 0.2263 |
| false_negative_rate | 0.4554 |
| uncertain_rate | 0.0000 |
| judge_override_rate | 0.7246 |
| support_adversarial | 1682 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.3597
- Macro F1: 0.3315

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [602, 32]
  [279, 3]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 206 | 0.000 | 0.985 |
| 0.5-0.6 | 28 | 0.500 | 1.000 |
| 0.8-0.9 | 40 | 0.846 | 0.650 |
| 0.9-1.0 | 1545 | 0.966 | 0.495 |
