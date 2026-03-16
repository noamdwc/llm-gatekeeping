# Post-hoc Benign Risk Model Report

> **Note**: Model trained on val trace, evaluated on test trace.
> This provides a proper train/test separation for the risk model.

## Data

- **Eligible samples**: 511 (LLM-path rows where LLM predicted benign pre-policy)
- **True adversarial**: 250 (48.9% base rate)
- **True benign**: 261

## Model Comparison

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Isotonic (margin only) | 0.5000 ± 0.0000 | 0.4892 ± 0.0000 | 0.2593 ± 0.0000 |
| Logistic (all features) | 0.9515 ± 0.0000 | 0.9320 ± 0.0000 | 0.0809 ± 0.0000 |

## Calibration (best model)

- **ECE**: 0.0615

| Bin | Count | Avg Predicted | Avg Actual |
|-----|-------|---------------|------------|
| [0.00, 0.10) | 205 | 0.0562 | 0.0000 |
| [0.10, 0.20) | 19 | 0.1259 | 0.3158 |
| [0.20, 0.30) | 3 | 0.2230 | 1.0000 |
| [0.30, 0.40) | 10 | 0.3455 | 0.7000 |
| [0.40, 0.50) | 5 | 0.4627 | 0.8000 |
| [0.50, 0.60) | 14 | 0.5438 | 0.5714 |
| [0.60, 0.70) | 16 | 0.6572 | 0.5000 |
| [0.70, 0.80) | 22 | 0.7547 | 0.7727 |
| [0.80, 0.90) | 81 | 0.8563 | 0.8765 |
| [0.90, 1.00) | 136 | 0.9545 | 0.9265 |

## Policy Simulation — Two-Zone

|   threshold |   tpr |    fpr |   precision |   recall |     f1 |   accuracy |   n_flipped |
|------------:|------:|-------:|------------:|---------:|-------:|-----------:|------------:|
|        0.5  | 0.92  | 0.1494 |      0.855  |    0.92  | 0.8863 |     0.8845 |         269 |
|        0.6  | 0.888 | 0.1264 |      0.8706 |    0.888 | 0.8792 |     0.8806 |         255 |
|        0.7  | 0.856 | 0.0958 |      0.8954 |    0.856 | 0.8753 |     0.8806 |         239 |
|        0.8  | 0.788 | 0.0766 |      0.9078 |    0.788 | 0.8437 |     0.8571 |         217 |
|        0.9  | 0.504 | 0.0383 |      0.9265 |    0.504 | 0.6528 |     0.7378 |         136 |
|        0.95 | 0.264 | 0.0192 |      0.9296 |    0.264 | 0.4112 |     0.6301 |          71 |

## Policy Simulation — Three-Zone

|   low_threshold |   high_threshold |   benign_zone_coverage |   uncertain_zone_coverage |   adversarial_zone_coverage |   benign_zone_accuracy |   adversarial_zone_accuracy |
|----------------:|-----------------:|-----------------------:|--------------------------:|----------------------------:|-----------------------:|----------------------------:|
|             0.3 |              0.7 |                 0.4442 |                    0.0881 |                      0.4677 |                 0.9604 |                      0.8954 |
|             0.4 |              0.8 |                 0.4638 |                    0.1115 |                      0.4247 |                 0.9325 |                      0.9078 |
|             0.5 |              0.9 |                 0.4736 |                    0.2603 |                      0.2661 |                 0.9174 |                      0.9265 |

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
