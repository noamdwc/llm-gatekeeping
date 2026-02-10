# External Evaluation Report — jackhhao

- **Dataset**: `jackhhao/jailbreak-classification`
- **Mode**: ml
- **Samples**: 262

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.2672 |
| adversarial_precision | 0.3627 |
| adversarial_recall | 0.5036 |
| adversarial_f1 | 0.4217 |
| benign_precision | 0.0000 |
| benign_recall | 0.0000 |
| benign_f1 | 0.0000 |
| false_negative_rate | 0.4964 |
| support_adversarial | 139 |
| support_benign | 123 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 56 | 0.551 | 0.464 |
| 0.6-0.7 | 26 | 0.645 | 0.423 |
| 0.7-0.8 | 31 | 0.743 | 0.419 |
| 0.8-0.9 | 53 | 0.859 | 0.226 |
| 0.9-1.0 | 96 | 0.933 | 0.083 |
