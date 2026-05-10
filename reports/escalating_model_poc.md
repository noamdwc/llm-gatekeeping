# Escalating Model POC

This offline POC trains a model to estimate `P(cheap path is wrong)` from Colab/local classifier output and DeBERTa output. It does not choose a production threshold or integrate with the hybrid router.

Parsed `clf_token_logprobs` features are intentionally omitted in this version except for top-1, top-2, and top-1 minus top-2 label-token logprob features from the cheap/local LLM classifier output.

## Evaluation Summary

| split          |   rows_colab |   rows_deberta |   rows_joined |   rows_dropped_colab_only |   rows_dropped_deberta_only |   cheap_error_rate |   roc_auc |   pr_auc |   top_10pct_error_rate |   top_10pct_adversarial_fn_rate |   bottom_50pct_error_rate |
|:---------------|-------------:|---------------:|--------------:|--------------------------:|----------------------------:|-------------------:|----------:|---------:|-----------------------:|--------------------------------:|--------------------------:|
| test           |         2581 |           2581 |          2581 |                         0 |                           0 |          0.0523053 |  0.872932 | 0.527427 |               0.328185 |                        0.173267 |                0.00697134 |
| unseen_val     |         1881 |           1881 |          1881 |                         0 |                           0 |          0.072302  |  0.905566 | 0.648186 |               0.465608 |                        0.398693 |                0.0053135  |
| unseen_test    |         1894 |           1894 |          1894 |                         0 |                           0 |          0.0881732 |  0.88792  | 0.620394 |               0.515789 |                        0.461538 |                0.0116156  |
| safeguard_test |         1552 |           1555 |          1552 |                         0 |                           3 |          0.0431701 |  0.965003 | 0.847015 |               0.403846 |                        0.158879 |                0.00257732 |
