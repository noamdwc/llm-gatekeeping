# External Evaluation Report — jackhhao

- **Dataset**: `jackhhao/jailbreak-classification`
- **Mode**: hybrid
- **Samples**: 262

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7405 |
| adversarial_precision | 0.7795 |
| adversarial_recall | 0.7122 |
| adversarial_f1 | 0.7444 |
| benign_precision | 0.7037 |
| benign_recall | 0.7724 |
| benign_f1 | 0.7364 |
| false_positive_rate | 0.2276 |
| false_negative_rate | 0.2878 |
| uncertain_rate | 0.0000 |
| judge_override_rate | nan |
| support_adversarial | 139 |
| support_benign | 123 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.0-0.1 | 2 | 0.000 | 1.000 |
| 0.8-0.9 | 103 | 0.877 | 0.825 |
| 0.9-1.0 | 157 | 0.916 | 0.682 |

## Router Stats

- total: 262
- ml_handled: 21
- llm_escalated: 239
- abstained: 2
- ml_rate: 0.0802
- llm_rate: 0.9122
- abstain_rate: 0.0076
