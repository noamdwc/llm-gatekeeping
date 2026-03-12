# Post-hoc Benign Risk Model Report

> **Note**: Model trained on val trace, evaluated on test trace.
> This provides a proper train/test separation for the risk model.

## Data

- **Eligible samples**: 568 (LLM-path rows where LLM predicted benign pre-policy)
- **True adversarial**: 462 (81.3% base rate)
- **True benign**: 106

## Model Comparison

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Isotonic (margin only) | 0.5000 ± 0.0000 | 0.8134 ± 0.0000 | 0.1526 ± 0.0000 |
| Logistic (all features) | 0.7558 ± 0.0000 | 0.9102 ± 0.0000 | 0.1094 ± 0.0000 |

## Calibration (best model)

- **ECE**: 0.0361

| Bin | Count | Avg Predicted | Avg Actual |
|-----|-------|---------------|------------|
| [0.00, 0.10) | 36 | 0.0493 | 0.0556 |
| [0.10, 0.20) | 3 | 0.1560 | 0.0000 |
| [0.20, 0.30) | 3 | 0.2784 | 0.6667 |
| [0.30, 0.40) | 2 | 0.3236 | 0.5000 |
| [0.40, 0.50) | 2 | 0.4436 | 1.0000 |
| [0.50, 0.60) | 6 | 0.5766 | 0.6667 |
| [0.60, 0.70) | 36 | 0.6497 | 0.7778 |
| [0.70, 0.80) | 70 | 0.7625 | 0.7714 |
| [0.80, 0.90) | 148 | 0.8621 | 0.8851 |
| [0.90, 1.00) | 262 | 0.9388 | 0.9084 |

## Policy Simulation — Two-Zone

|   threshold |    tpr |    fpr |   precision |   recall |     f1 |   accuracy |   n_flipped |
|------------:|-------:|-------:|------------:|---------:|-------:|-----------:|------------:|
|        0.5  | 0.9848 | 0.6321 |      0.8716 |   0.9848 | 0.9248 |     0.8697 |         522 |
|        0.6  | 0.9762 | 0.6132 |      0.874  |   0.9762 | 0.9223 |     0.8662 |         516 |
|        0.7  | 0.9156 | 0.5377 |      0.8812 |   0.9156 | 0.8981 |     0.831  |         480 |
|        0.8  | 0.7987 | 0.3868 |      0.9    |   0.7987 | 0.8463 |     0.7641 |         410 |
|        0.9  | 0.5152 | 0.2264 |      0.9084 |   0.5152 | 0.6575 |     0.5634 |         262 |
|        0.95 | 0.2035 | 0.0472 |      0.9495 |   0.2035 | 0.3351 |     0.3433 |          99 |

## Policy Simulation — Three-Zone

|   low_threshold |   high_threshold |   benign_zone_coverage |   uncertain_zone_coverage |   adversarial_zone_coverage |   benign_zone_accuracy |   adversarial_zone_accuracy |
|----------------:|-----------------:|-----------------------:|--------------------------:|----------------------------:|-----------------------:|----------------------------:|
|             0.3 |              0.7 |                 0.0739 |                    0.081  |                      0.8451 |                 0.9048 |                      0.8812 |
|             0.4 |              0.8 |                 0.0775 |                    0.2007 |                      0.7218 |                 0.8864 |                      0.9    |
|             0.5 |              0.9 |                 0.081  |                    0.4577 |                      0.4613 |                 0.8478 |                      0.9084 |

## Recommendations

1. **Is margin alone sufficient?** Check ROC-AUC of isotonic vs logistic.
   If logistic AUC is materially higher, the extra features add value.
2. **Best operating point?** Refer to the two-zone table for the threshold
   that balances FPR and recall for your use case.
3. **Three-zone viable?** If the uncertain zone has high coverage with low
   accuracy, a three-zone policy can defer hard cases to human review.
4. **Productionization**: Would require training on a held-out calibration
   set (not the test traces) and periodic recalibration as LLM behavior drifts.

## Next Steps

- Train on a dedicated calibration split (not test-derived traces)
- Evaluate on external datasets for generalization
- Consider adding text-length features if trace is joined back to raw data
- Monitor calibration drift over time
