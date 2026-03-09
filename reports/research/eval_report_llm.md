# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5915 |
| adversarial_precision | 0.9653 |
| adversarial_recall | 0.5791 |
| adversarial_f1 | 0.7239 |
| benign_precision | 0.1259 |
| benign_recall | 0.7445 |
| benign_f1 | 0.2154 |
| false_positive_rate | 0.2555 |
| false_negative_rate | 0.4209 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1682 |
| support_benign | 137 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.3686
- Macro F1: 0.3623

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [590, 97]
  [257, 30]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 221 | 0.000 | 1.000 |
| 0.5-0.6 | 29 | 0.510 | 0.828 |
| 0.8-0.9 | 80 | 0.820 | 0.287 |
| 0.9-1.0 | 1489 | 0.955 | 0.543 |
