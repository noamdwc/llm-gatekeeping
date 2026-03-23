# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.8072 |
| adversarial_precision | 0.9581 |
| adversarial_recall | 0.7982 |
| adversarial_f1 | 0.8709 |
| benign_precision | 0.4885 |
| benign_recall | 0.8467 |
| benign_f1 | 0.6195 |
| false_positive_rate | 0.1533 |
| false_negative_rate | 0.2018 |
| uncertain_rate | 0.0000 |
| judge_override_rate | 0.2732 |
| support_adversarial | 1318 |
| support_benign | 300 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5023
- Macro F1: 0.5534

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [267, 366]
  [24, 395]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 91 | 0.000 | 1.000 |
| 0.5-0.6 | 10 | 0.510 | 1.000 |
| 0.8-0.9 | 786 | 0.840 | 0.950 |
| 0.9-1.0 | 731 | 0.972 | 0.627 |

