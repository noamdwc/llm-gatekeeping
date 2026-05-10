# Escalating Model POC

This offline POC trains a model to estimate `P(cheap path is wrong)` from Colab/local classifier output and DeBERTa output. It does not choose a production threshold or integrate with the hybrid router.

Parsed `clf_token_logprobs` features are intentionally omitted in this version except for top-1, top-2, and top-1 minus top-2 label-token logprob features from the cheap/local LLM classifier output.

## Evaluation Summary

| split          |   rows_colab |   rows_deberta |   rows_joined |   rows_dropped_colab_only |   rows_dropped_deberta_only |   cheap_error_rate |   roc_auc |   pr_auc |   top_10pct_error_rate |   top_10pct_adversarial_fn_rate |   bottom_50pct_error_rate |
|:---------------|-------------:|---------------:|--------------:|--------------------------:|----------------------------:|-------------------:|----------:|---------:|-----------------------:|--------------------------------:|--------------------------:|
| test           |         2581 |           2581 |          2581 |                         0 |                           0 |          0.0523053 |  0.854868 | 0.490928 |               0.30888  |                        0.169082 |                0.00774593 |
| unseen_val     |         1881 |           1881 |          1881 |                         0 |                           0 |          0.0717703 |  0.878414 | 0.588433 |               0.47619  |                        0.403974 |                0.0042508  |
| unseen_test    |         1894 |           1894 |          1894 |                         0 |                           0 |          0.0876452 |  0.846176 | 0.555198 |               0.515789 |                        0.45098  |                0.0137276  |
| safeguard_test |         1552 |           1555 |          1552 |                         0 |                           3 |          0.0451031 |  0.960681 | 0.891544 |               0.423077 |                        0.169811 |                0.00386598 |
