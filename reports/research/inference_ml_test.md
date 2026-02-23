# ML Inference Report — test

## Binary Detection (Adversarial vs Benign)

| Metric | Value |
|--------|-------|
| accuracy | 0.5893 |
| adversarial_precision | 1.0000 |
| adversarial_recall | 0.5652 |
| adversarial_f1 | 0.7222 |
| benign_precision | 0.1193 |
| benign_recall | 1.0000 |
| benign_f1 | 0.2132 |
| false_negative_rate | 0.4348 |
| uncertain_rate | 0.0000 |
| judge_override_rate | N/A |
| support_adversarial | 1596 |
| support_benign | 94 |

## Category Classification (Unicode vs NLP)

- Accuracy: 0.5652
- Macro F1: 0.5000

Confusion matrix (rows=true, cols=pred):
Labels: ['unicode_attack', 'nlp_attack']
  [902, 0]
  [0, 0]

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
| 0.5-0.6 | 44 | 0.567 | 0.000 |
| 0.6-0.7 | 305 | 0.659 | 0.085 |
| 0.7-0.8 | 387 | 0.743 | 0.147 |
| 0.8-0.9 | 101 | 0.849 | 0.594 |
| 0.9-1.0 | 853 | 0.974 | 1.000 |


## Scope Breakdown

| Scope | Rows | Accuracy | False-negative rate |
|-------|------|----------|---------------------|
| full | 1690 | 0.5893 | 0.4348 |
| ml_scope_no_nlp | 996 | 1.0000 | 0.0000 |
| nlp_only | 694 | 0.0000 | 1.0000 |

- `ml_scope_no_nlp` excludes `label_category == nlp_attack` rows.
- `nlp_only` is out-of-scope for the unicode/character-specialist ML model.
