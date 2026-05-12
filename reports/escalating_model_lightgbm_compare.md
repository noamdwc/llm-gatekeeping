# Escalating Model POC

This offline POC trains a model to estimate `P(cheap path is wrong)` from Colab/local classifier output and DeBERTa output. It does not choose a production threshold or integrate with the hybrid router.

Parsed `clf_token_logprobs` features are intentionally omitted in this version except for top-1, top-2, and top-1 minus top-2 label-token logprob features from the cheap/local LLM classifier output.

## Evaluation Summary

| split          |   rows_colab |   rows_deberta |   rows_joined |   rows_dropped_colab_only |   rows_dropped_deberta_only |   cheap_error_rate |   roc_auc |   pr_auc |   top_10pct_error_rate |   top_10pct_adversarial_fn_rate |   bottom_50pct_error_rate |
|:---------------|-------------:|---------------:|--------------:|--------------------------:|----------------------------:|-------------------:|----------:|---------:|-----------------------:|--------------------------------:|--------------------------:|
| test           |         2581 |           2581 |          2581 |                         0 |                           0 |          0.0523053 |  0.931266 | 0.580215 |               0.312741 |                        0.174129 |               0.000774593 |
| unseen_val     |         1881 |           1881 |          1881 |                         0 |                           0 |          0.072302  |  0.925655 | 0.668005 |               0.460317 |                        0.397351 |               0.0021254   |
| unseen_test    |         1894 |           1894 |          1894 |                         0 |                           0 |          0.0881732 |  0.92438  | 0.677218 |               0.552632 |                        0.493421 |               0.0063358   |
| safeguard_test |         1552 |           1555 |          1552 |                         0 |                           3 |          0.0431701 |  0.976346 | 0.911843 |               0.403846 |                        0.168317 |               0.00257732  |
