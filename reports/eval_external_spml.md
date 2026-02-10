# External Evaluation Report — spml

- **Dataset**: `reshabhs/SPML_Chatbot_Prompt_Injection`
- **Mode**: hybrid
- **Samples**: 16011

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.3912 |
| adversarial_precision | 0.6454 |
| adversarial_recall | 0.4943 |
| adversarial_f1 | 0.5598 |
| benign_precision | 0.0100 |
| benign_recall | 0.0184 |
| benign_f1 | 0.0130 |
| false_negative_rate | 0.5057 |
| support_adversarial | 12541 |
| support_benign | 3470 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.8-0.9 | 3943 | 0.863 | 0.878 |
| 0.9-1.0 | 12068 | 0.944 | 0.232 |

## Router Stats

- total: 16011
- ml_handled: 7851
- llm_escalated: 8160
- abstained: 0
- ml_rate: 0.4904
- llm_rate: 0.5096
- abstain_rate: 0.0
