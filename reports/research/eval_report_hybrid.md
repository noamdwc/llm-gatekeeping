# Hybrid Router Evaluation Report

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.9491 |
| adversarial_precision | 0.9526 |
| adversarial_recall | 0.9956 |
| adversarial_f1 | 0.9737 |
| benign_precision | 0.6818 |
| benign_recall | 0.1596 |
| benign_f1 | 0.2586 |
| false_negative_rate | 0.0044 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1596 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.9931
- Macro F1: 0.9960

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [902, 0]
  [0, 683]

## Per-Type Classification (Unicode Sub-Types)

- Accuracy: 1.0000
- Macro F1: 1.0000

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Bidirectional Text | 1.00 | 1.00 | 1.00 | 82 |
| Deletion Characters | 1.00 | 1.00 | 1.00 | 82 |
| Diacritcs | 1.00 | 1.00 | 1.00 | 82 |
| Full Width Text | 1.00 | 1.00 | 1.00 | 82 |
| Homoglyphs | 1.00 | 1.00 | 1.00 | 82 |
| Numbers | 1.00 | 1.00 | 1.00 | 82 |
| Spaces | 1.00 | 1.00 | 1.00 | 82 |
| Underline Accent Marks | 1.00 | 1.00 | 1.00 | 82 |
| Unicode Tags Smuggling | 1.00 | 1.00 | 1.00 | 82 |
| Upside Down Text | 1.00 | 1.00 | 1.00 | 82 |
| Zero Width | 1.00 | 1.00 | 1.00 | 82 |
| macro avg | 1.00 | 1.00 | 1.00 | 902 |
| weighted avg | 1.00 | 1.00 | 1.00 | 902 |

## Calibration

| Bin | Count | Avg Confidence | Accuracy |
|-----|-------|----------------|----------|
| 0.5-0.6 | 28 | 0.560 | 0.607 |
| 0.6-0.7 | 29 | 0.657 | 0.655 |
| 0.7-0.8 | 148 | 0.757 | 0.811 |
| 0.8-0.9 | 278 | 0.860 | 0.896 |
| 0.9-1.0 | 1207 | 0.982 | 0.993 |

## Cost / Usage

- routed_ml: 1664
- routed_llm: 26
