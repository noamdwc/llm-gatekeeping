# DeBERTa External Evaluation — deepset

- **Dataset**: `deepset/prompt-injections`
- **Split**: `test`
- **Samples**: 116

## Binary Detection

| Metric | Value |
|--------|-------|
| accuracy | 0.5690 |
| adversarial_precision | 0.7778 |
| adversarial_recall | 0.2333 |
| adversarial_f1 | 0.3590 |
| benign_precision | 0.5306 |
| benign_recall | 0.9286 |
| benign_f1 | 0.6753 |
| false_positive_rate | 0.0714 |
| false_negative_rate | 0.7667 |
| uncertain_rate | 0.0000 |
| judge_override_rate | nan |
| support_adversarial | 60 |
| support_benign | 56 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.6-0.7 | 1 | 0.677 | 1.000 |
| 0.7-0.8 | 1 | 0.730 | 0.000 |
| 0.8-0.9 | 2 | 0.861 | 0.000 |
| 0.9-1.0 | 112 | 0.997 | 0.580 |
