# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5349 |
| adversarial_precision | 0.9742 |
| adversarial_recall | 0.5213 |
| adversarial_f1 | 0.6792 |
| benign_precision | 0.0861 |
| benign_recall | 0.7660 |
| benign_f1 | 0.1548 |
| false_positive_rate | 0.2340 |
| false_negative_rate | 0.4787 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1596 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.3503
- Macro F1: 0.3569

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [539, 91]
  [182, 20]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 0.0000
- Macro F1: 0.0000

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 201 | 0.000 | 1.000 |
| 0.5-0.6 | 31 | 0.500 | 0.968 |
| 0.8-0.9 | 104 | 0.829 | 0.394 |
| 0.9-1.0 | 1354 | 0.953 | 0.467 |
