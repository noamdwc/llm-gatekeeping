# Logprob Threshold Analysis — Summary Report

**Date**: 2026-03-12  
**Dataset**: main test set (1819 samples)

## Signal Coverage

- Classifier logprob signal: 1589/1819 (87.4%)
- Judge logprob signal: 1305/1819 (71.7%)
- Preferred (judge when available): 1809/1819 (99.5%)

## Calibration (ECE)

| Signal | ECE |
|--------|-----|
| Self-reported (clf_confidence/100) | 0.5410 |
| Logprob prob (exp(label_start_logprob)) | 0.3215 |
| Margin (label_start_margin / 5) | 0.2782 |

## Correctness Discrimination (AUC-ROC)

| subgroup             |    n |   AUC_clf_confidence |   AUC_label_start_prob |   AUC_label_start_margin |
|:---------------------|-----:|---------------------:|-----------------------:|-------------------------:|
| All                  | 1809 |            0.509871  |               0.57826  |                 0.586773 |
| Benign GT            |  137 |            0.0652769 |               0.472611 |                 0.461199 |
| Adversarial GT       | 1672 |            0.540515  |               0.579369 |                 0.590381 |
| Hybrid→LLM           |  868 |            0.668944  |               0.687078 |                 0.698775 |
| Judge ran (stages=2) | 1318 |            0.375778  |               0.449697 |                 0.448667 |
| Clf only (stages=1)  |  491 |            0.5       |               0.744911 |                 0.743444 |

## Threshold Sweep — Best Configs

Baseline: FP=31, adv_recall=0.5311, ben_f1=0.2064

Best configs (adv_recall drop <= 1%):

| policy                 |   threshold |   accuracy |   adv_recall |   ben_f1 |   FP |   FN |
|:-----------------------|------------:|-----------:|-------------:|---------:|-----:|-----:|
| confidence_only        |         0.8 |   0.561636 |     0.544258 | 0.210945 |   31 |  762 |
| combined_conf90+margin |         0   |   0.561636 |     0.544258 | 0.210945 |   31 |  762 |
| baseline               |       nan   |   0.549475 |     0.5311   | 0.206426 |   31 |  784 |
| margin_only            |         0   |   0.549475 |     0.5311   | 0.206426 |   31 |  784 |

## Hybrid Pipeline Simulation

Current hybrid: accuracy=0.7367, adv_recall=0.7337, ben_F1=0.3068, FP=31, FN=448

|   margin_threshold |   accuracy |   adv_recall |   ben_f1 |   FP |   FN |
|-------------------:|-----------:|-------------:|---------:|-----:|-----:|
|                0   |   0.736668 |     0.73365  | 0.306802 |   31 |  448 |
|                0.5 |   0.772402 |     0.780618 | 0.307692 |   45 |  369 |
|                1   |   0.80099  |     0.814507 | 0.324627 |   50 |  312 |
|                1.5 |   0.82133  |     0.840071 | 0.332649 |   56 |  269 |
|                2   |   0.836723 |     0.858502 | 0.344371 |   59 |  238 |
|                2.5 |   0.853766 |     0.87931  | 0.357488 |   63 |  203 |
|                3   |   0.862562 |     0.894174 | 0.342105 |   72 |  178 |
|                3.5 |   0.875206 |     0.91082  | 0.345821 |   77 |  150 |
|                4   |   0.882353 |     0.921522 | 0.339506 |   82 |  132 |
|                4.5 |   0.88895  |     0.934007 | 0.312925 |   91 |  111 |
|                5   |   0.894447 |     0.941736 | 0.309353 |   94 |   98 |
|                5.5 |   0.898846 |     0.947087 | 0.313433 |   95 |   89 |
|                6   |   0.905443 |     0.954221 | 0.328125 |   95 |   77 |

## Recommendations

1. **Best logprob metric**: `label_start_margin` — the rank-1 vs rank-2 logprob delta at the label-start token. It captures model certainty at the decision point directly.
2. **Suggested threshold range**: See sweep results above. Pick the margin threshold that achieves the desired FP/adv_recall tradeoff.
3. **Implementation path**: Add `logprob_margin_threshold` to `configs/default.yaml` under `hybrid:`. In `hybrid_router.py`, after LLM prediction, check `label_start_margin >= threshold` before accepting a benign prediction.
4. **Blocker**: NIM `top_logprobs` token names are empty for the classifier stage (but present for the judge). This means clf margin is rank-1 vs rank-2 logprob delta, not a semantic `adversarial` vs `benign` comparison. The judge stage does have token names, enabling semantic margin computation.
5. **Recommendation**: Conditionally adopt — if the AUC and sweep results show improvement over self-reported confidence, integrate margin gating. If not, the logprob signal adds no value beyond what `clf_confidence` already provides.
