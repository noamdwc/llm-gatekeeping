# DeBERTa External Evaluation — deepset

- **Dataset**: `deepset/prompt-injections`
- **Split**: `test`
- **Samples**: 116

## Binary Detection

| Metric | Value |
|--------|-------|
| accuracy | 0.6034 |
| adversarial_precision | 0.8500 |
| adversarial_recall | 0.2833 |
| adversarial_f1 | 0.4250 |
| benign_precision | 0.5521 |
| benign_recall | 0.9464 |
| benign_f1 | 0.6974 |
| false_positive_rate | 0.0536 |
| false_negative_rate | 0.7167 |
| uncertain_rate | 0.0000 |
| judge_override_rate | nan |
| support_adversarial | 60 |
| support_benign | 56 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 1 | 0.560 | 1.000 |
| 0.7-0.8 | 3 | 0.744 | 0.333 |
| 0.8-0.9 | 6 | 0.848 | 0.167 |
| 0.9-1.0 | 106 | 0.993 | 0.632 |
