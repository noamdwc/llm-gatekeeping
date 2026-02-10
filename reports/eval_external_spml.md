# External Evaluation Report — spml

- **Dataset**: `reshabhs/SPML_Chatbot_Prompt_Injection`
- **Mode**: ml
- **Samples**: 16011

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7439 |
| adversarial_precision | 0.7744 |
| adversarial_recall | 0.9497 |
| adversarial_f1 | 0.8531 |
| benign_precision | 0.0000 |
| benign_recall | 0.0000 |
| benign_f1 | 0.0000 |
| false_negative_rate | 0.0503 |
| support_adversarial | 12541 |
| support_benign | 3470 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 1235 | 0.552 | 0.708 |
| 0.6-0.7 | 1876 | 0.654 | 0.902 |
| 0.7-0.8 | 3170 | 0.753 | 0.976 |
| 0.8-0.9 | 3924 | 0.851 | 0.942 |
| 0.9-1.0 | 5806 | 0.944 | 0.439 |
