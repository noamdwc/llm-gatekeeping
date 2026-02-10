# External Evaluation Report — deepset

- **Dataset**: `deepset/prompt-injections`
- **Mode**: ml
- **Samples**: 116

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5086 |
| adversarial_precision | 0.5130 |
| adversarial_recall | 0.9833 |
| adversarial_f1 | 0.6743 |
| benign_precision | 0.0000 |
| benign_recall | 0.0000 |
| benign_f1 | 0.0000 |
| false_negative_rate | 0.0167 |
| support_adversarial | 60 |
| support_benign | 56 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 1 | 0.591 | 0.000 |
| 0.6-0.7 | 1 | 0.676 | 1.000 |
| 0.7-0.8 | 5 | 0.744 | 1.000 |
| 0.8-0.9 | 12 | 0.873 | 1.000 |
| 0.9-1.0 | 97 | 0.960 | 0.423 |
