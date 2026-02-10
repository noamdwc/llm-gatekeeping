# External Evaluation Report — safeguard

- **Dataset**: `xTRam1/safe-guard-prompt-injection`
- **Mode**: hybrid
- **Samples**: 2060

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.2854 |
| adversarial_precision | 0.2732 |
| adversarial_recall | 0.7615 |
| adversarial_f1 | 0.4021 |
| benign_precision | 0.3750 |
| benign_recall | 0.0660 |
| benign_f1 | 0.1122 |
| false_negative_rate | 0.2385 |
| support_adversarial | 650 |
| support_benign | 1410 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.8-0.9 | 373 | 0.869 | 0.343 |
| 0.9-1.0 | 1687 | 0.944 | 0.273 |

## Router Stats

- total: 2060
- ml_handled: 1367
- llm_escalated: 693
- abstained: 0
- ml_rate: 0.6636
- llm_rate: 0.3364
- abstain_rate: 0.0
