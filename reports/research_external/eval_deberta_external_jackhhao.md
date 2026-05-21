# DeBERTa External Evaluation — jackhhao

- **Dataset**: `jackhhao/jailbreak-classification`
- **Split**: `test`
- **Samples**: 262

## Binary Detection

| Metric | Value |
|--------|-------|
| accuracy | 0.8359 |
| adversarial_precision | 0.8429 |
| adversarial_recall | 0.8489 |
| adversarial_f1 | 0.8459 |
| benign_precision | 0.8279 |
| benign_recall | 0.8211 |
| benign_f1 | 0.8245 |
| false_positive_rate | 0.1789 |
| false_negative_rate | 0.1511 |
| uncertain_rate | 0.0000 |
| judge_override_rate | nan |
| support_adversarial | 139 |
| support_benign | 123 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 1 | 0.511 | 0.000 |
| 0.6-0.7 | 1 | 0.678 | 0.000 |
| 0.7-0.8 | 2 | 0.739 | 1.000 |
| 0.8-0.9 | 10 | 0.856 | 0.100 |
| 0.9-1.0 | 248 | 0.992 | 0.871 |
