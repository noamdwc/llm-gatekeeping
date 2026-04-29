# DeBERTa External Evaluation — jackhhao

- **Dataset**: `jackhhao/jailbreak-classification`
- **Split**: `test`
- **Samples**: 262

## Binary Detection

| Metric | Value |
|--------|-------|
| accuracy | 0.8855 |
| adversarial_precision | 0.8978 |
| adversarial_recall | 0.8849 |
| adversarial_f1 | 0.8913 |
| benign_precision | 0.8720 |
| benign_recall | 0.8862 |
| benign_f1 | 0.8790 |
| false_positive_rate | 0.1138 |
| false_negative_rate | 0.1151 |
| uncertain_rate | 0.0000 |
| judge_override_rate | nan |
| support_adversarial | 139 |
| support_benign | 123 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 6 | 0.561 | 0.500 |
| 0.6-0.7 | 3 | 0.647 | 0.667 |
| 0.7-0.8 | 6 | 0.742 | 0.500 |
| 0.8-0.9 | 2 | 0.851 | 0.500 |
| 0.9-1.0 | 245 | 0.986 | 0.910 |
