# External Evaluation Report — safeguard

- **Dataset**: `xTRam1/safe-guard-prompt-injection`
- **Mode**: ml
- **Samples**: 2060

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.1820 |
| adversarial_precision | 0.2098 |
| adversarial_recall | 0.5754 |
| adversarial_f1 | 0.3074 |
| benign_precision | 0.0036 |
| benign_recall | 0.0007 |
| benign_f1 | 0.0012 |
| false_negative_rate | 0.4246 |
| support_adversarial | 650 |
| support_benign | 1410 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 229 | 0.551 | 0.515 |
| 0.6-0.7 | 189 | 0.654 | 0.593 |
| 0.7-0.8 | 170 | 0.752 | 0.576 |
| 0.8-0.9 | 350 | 0.864 | 0.103 |
| 0.9-1.0 | 1122 | 0.943 | 0.010 |
