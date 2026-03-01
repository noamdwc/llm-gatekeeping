# External Evaluation Report — deepset

- **Dataset**: `deepset/prompt-injections`
- **Mode**: hybrid
- **Samples**: 116

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5345 |
| adversarial_precision | 0.5455 |
| adversarial_recall | 0.6000 |
| adversarial_f1 | 0.5714 |
| benign_precision | 0.5200 |
| benign_recall | 0.4643 |
| benign_f1 | 0.4906 |
| false_positive_rate | 0.5357 |
| false_negative_rate | 0.4000 |
| uncertain_rate | 0.0000 |
| judge_override_rate | nan |
| support_adversarial | 60 |
| support_benign | 56 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 1 | 0.600 | 1.000 |
| 0.8-0.9 | 21 | 0.860 | 0.571 |
| 0.9-1.0 | 94 | 0.939 | 0.521 |

## Router Stats

- total: 116
- ml_handled: 57
- llm_escalated: 58
- abstained: 1
- ml_rate: 0.4914
- llm_rate: 0.5
- abstain_rate: 0.0086
