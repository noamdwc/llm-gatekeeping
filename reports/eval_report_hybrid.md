# LLM Classifier Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.7900 |
| adversarial_precision | 0.8556 |
| adversarial_recall | 0.9059 |
| adversarial_f1 | 0.8800 |
| benign_precision | 0.2000 |
| benign_recall | 0.1333 |
| benign_f1 | 0.1600 |
| false_negative_rate | 0.0941 |
| support_adversarial | 85 |
| support_benign | 15 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.9610
- Macro F1: 0.9560

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [50, 0]
  [3, 24]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 1.0000
- Macro F1: 1.0000

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 3 |
| Deletion Characters | 1.00 | 1.00 | 1.00 | 4 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 6 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 7 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 3 |
| Numbers | 1.00 | 1.00 | 1.00 | 4 |
| Spaces | 1.00 | 1.00 | 1.00 | 6 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 5 |
| Unicode Tags Smuggling | 1.00 | 1.00 | 1.00 | 7 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 1 |
| Zero Width | 1.00 | 1.00 | 1.00 | 4 |
| macro avg | 1.00 | 1.00 | 1.00 | 50 |
| weighted avg | 1.00 | 1.00 | 1.00 | 50 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.8-0.9 | 18 | 0.857 | 0.667 |
| 0.9-1.0 | 82 | 0.975 | 0.817 |

## Cost / Usage

- total_calls: 75
- prompt_tokens: 107527
- completion_tokens: 1187
- total_tokens: 108714
- total_latency_s: 73.34
- avg_latency_s: 0.978
- calls_by_stage: {'binary': 40, 'category': 30, 'type': 5}
- total: 100
- ml_handled: 60
- llm_escalated: 40
- abstained: 0
- ml_rate: 0.6
- llm_rate: 0.4
- abstain_rate: 0.0
