# Post-hoc Benign Risk Model Report

> **Note**: Model trained on val trace, evaluated on test trace.
> This provides a proper train/test separation for the risk model.

## Data

- **Eligible samples**: 386 (LLM-path rows where LLM predicted benign pre-policy)
- **True adversarial**: 132 (34.2% base rate)
- **True benign**: 254

## Model Comparison

| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| Isotonic (margin only) | 0.5000 ± 0.0000 | 0.3420 ± 0.0000 | 0.2253 ± 0.0000 |
| Logistic (all features) | 0.9425 ± 0.0000 | 0.8684 ± 0.0000 | 0.0813 ± 0.0000 |

## Calibration (best model)

- **ECE**: 0.0454

| Bin | Count | Avg Predicted | Avg Actual |
|-----|-------|---------------|------------|
| [0.00, 0.10) | 201 | 0.0163 | 0.0050 |
| [0.10, 0.20) | 17 | 0.1400 | 0.1765 |
| [0.20, 0.30) | 4 | 0.2326 | 0.0000 |
| [0.30, 0.40) | 5 | 0.3312 | 0.4000 |
| [0.40, 0.50) | 7 | 0.4460 | 0.5714 |
| [0.50, 0.60) | 14 | 0.5474 | 0.7857 |
| [0.60, 0.70) | 30 | 0.6416 | 0.6333 |
| [0.70, 0.80) | 75 | 0.7515 | 0.8133 |
| [0.80, 0.90) | 33 | 0.8099 | 0.9394 |

## Policy Simulation — Two-Zone

|   threshold |    tpr |    fpr |   precision |   recall |     f1 |   accuracy |   n_flipped |
|------------:|-------:|-------:|------------:|---------:|-------:|-----------:|------------:|
|        0.5  | 0.9242 | 0.1181 |      0.8026 |   0.9242 | 0.8592 |     0.8964 |         152 |
|        0.6  | 0.8409 | 0.1063 |      0.8043 |   0.8409 | 0.8222 |     0.8756 |         138 |
|        0.7  | 0.697  | 0.063  |      0.8519 |   0.697  | 0.7667 |     0.8549 |         108 |
|        0.8  | 0.2348 | 0.0079 |      0.9394 |   0.2348 | 0.3758 |     0.7332 |          33 |
|        0.9  | 0      | 0      |      0      |   0      | 0      |     0.658  |           0 |
|        0.95 | 0      | 0      |      0      |   0      | 0      |     0.658  |           0 |

## Policy Simulation — Three-Zone

|   low_threshold |   high_threshold |   benign_zone_coverage |   uncertain_zone_coverage |   adversarial_zone_coverage |   benign_zone_accuracy |   adversarial_zone_accuracy |
|----------------:|-----------------:|-----------------------:|--------------------------:|----------------------------:|-----------------------:|----------------------------:|
|             0.3 |              0.7 |                 0.5751 |                    0.1451 |                      0.2798 |                 0.982  |                      0.8519 |
|             0.4 |              0.8 |                 0.5881 |                    0.3264 |                      0.0855 |                 0.9736 |                      0.9394 |
|             0.5 |              0.9 |                 0.6062 |                    0.3938 |                      0      |                 0.9573 |                    nan      |

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
