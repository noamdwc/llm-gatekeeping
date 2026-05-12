# Escalating Model POC

This offline POC trains a model to estimate `P(cheap path is wrong)` from Colab/local classifier output and DeBERTa output. It does not choose a production threshold or integrate with the hybrid router.

Parsed `clf_token_logprobs` features are intentionally omitted in this version except for top-1, top-2, and top-1 minus top-2 label-token logprob features from the cheap/local LLM classifier output.

## Evaluation Summary

| split             |   rows_colab |   rows_deberta |   rows_joined |   rows_dropped_colab_only |   rows_dropped_deberta_only |   cheap_error_rate |   roc_auc |   pr_auc |   top_10pct_error_rate |   top_10pct_adversarial_fn_rate |   bottom_50pct_error_rate |
|:------------------|-------------:|---------------:|--------------:|--------------------------:|----------------------------:|-------------------:|----------:|---------:|-----------------------:|--------------------------------:|--------------------------:|
| test              |         2581 |           2581 |          2581 |                         0 |                           0 |          0.0523053 |  0.931266 | 0.580215 |               0.312741 |                        0.174129 |               0.000774593 |
| unseen_val        |         1881 |           1881 |          1881 |                         0 |                           0 |          0.072302  |  0.925655 | 0.668005 |               0.460317 |                        0.397351 |               0.0021254   |
| unseen_test       |         1894 |           1894 |          1894 |                         0 |                           0 |          0.0881732 |  0.92438  | 0.677218 |               0.552632 |                        0.493421 |               0.0063358   |
| safeguard_test    |         1552 |           1555 |          1552 |                         0 |                           3 |          0.0431701 |  0.976346 | 0.911843 |               0.403846 |                        0.168317 |               0.00257732  |
| external_deepset  |          116 |            116 |           116 |                         0 |                           0 |          0.405172  |  0.640302 | 0.548201 |               0.5      |                        0.333333 |               0.344828    |
| external_jackhhao |          262 |            262 |           262 |                         0 |                           0 |          0.0839695 |  0.875852 | 0.492337 |               0.481481 |                        0.7      |               0.0152672   |

## Post-score unseen_val Split Diagnostics

Calibration method: `sigmoid`.

| postscore_split   |   rows |   prompt_hash_groups |   cheap_errors |   error_rate |
|:------------------|-------:|---------------------:|---------------:|-------------:|
| calibration       |    949 |                  521 |             70 |    0.0737619 |
| threshold         |    932 |                  521 |             66 |    0.0708155 |

### Label Counts

| label_binary   |   calibration |   threshold |
|:---------------|--------------:|------------:|
| adversarial    |           570 |         551 |
| benign         |           379 |         381 |

### Attack / Benign Group Counts

| attack_or_benign_group   |   calibration |   threshold |
|:-------------------------|--------------:|------------:|
| BAE                      |           165 |         156 |
| Emoji Smuggling          |           139 |         137 |
| Pruthi                   |            65 |          66 |
| TextFooler               |           201 |         192 |
| benign                   |           379 |         381 |

Prompt hash overlap: 0

## Limitations / Statistical Power

`unseen_val` has only 136 cheap-path errors total. The calibration half has 70 cheap-path errors, and the threshold-selection half has 66 cheap-path errors. Calibration and threshold estimates are therefore noisy.

One missed cheap-path error in the threshold half changes the missed-error rate by about 1.5 percentage points. Per-attack conclusions are diagnostic only. The selected threshold is a PoC operating point, not a final production threshold. Prefer a conservative threshold from a stable plateau, not necessarily the single best sweep point.

## Threshold Sweep

Threshold operating points are selected on `unseen_val` because the escalating model is trained on `val`.

|   threshold |   rows |   judge_call_rate |   judge_calls |   trusted_rows |   cheap_errors_total |   cheap_errors_caught |   cheap_errors_missed |   cheap_error_catch_rate |   non_escalated_error_rate |
|------------:|-------:|------------------:|--------------:|---------------:|---------------------:|----------------------:|----------------------:|-------------------------:|---------------------------:|
|        0    |    932 |         1         |           932 |              0 |                   66 |                    66 |                     0 |                 1        |                nan         |
|        0.05 |    932 |         0.110515  |           103 |            829 |                   66 |                    42 |                    24 |                 0.636364 |                  0.0289505 |
|        0.1  |    932 |         0.0772532 |            72 |            860 |                   66 |                    41 |                    25 |                 0.621212 |                  0.0290698 |
|        0.15 |    932 |         0.0718884 |            67 |            865 |                   66 |                    40 |                    26 |                 0.606061 |                  0.0300578 |
|        0.2  |    932 |         0.0633047 |            59 |            873 |                   66 |                    39 |                    27 |                 0.590909 |                  0.0309278 |
|        0.25 |    932 |         0.0622318 |            58 |            874 |                   66 |                    39 |                    27 |                 0.590909 |                  0.0308924 |
|        0.3  |    932 |         0.0579399 |            54 |            878 |                   66 |                    38 |                    28 |                 0.575758 |                  0.0318907 |
|        0.35 |    932 |         0.0536481 |            50 |            882 |                   66 |                    38 |                    28 |                 0.575758 |                  0.031746  |
|        0.4  |    932 |         0.0525751 |            49 |            883 |                   66 |                    38 |                    28 |                 0.575758 |                  0.0317101 |
|        0.45 |    932 |         0.0504292 |            47 |            885 |                   66 |                    37 |                    29 |                 0.560606 |                  0.0327684 |
|        0.5  |    932 |         0.0482833 |            45 |            887 |                   66 |                    36 |                    30 |                 0.545455 |                  0.0338219 |
|        0.55 |    932 |         0.0450644 |            42 |            890 |                   66 |                    34 |                    32 |                 0.515152 |                  0.0359551 |
|        0.6  |    932 |         0.0407725 |            38 |            894 |                   66 |                    32 |                    34 |                 0.484848 |                  0.0380313 |
|        0.65 |    932 |         0.0396996 |            37 |            895 |                   66 |                    31 |                    35 |                 0.469697 |                  0.0391061 |
|        0.7  |    932 |         0.0364807 |            34 |            898 |                   66 |                    30 |                    36 |                 0.454545 |                  0.0400891 |
|        0.75 |    932 |         0.02897   |            27 |            905 |                   66 |                    25 |                    41 |                 0.378788 |                  0.0453039 |
|        0.8  |    932 |         0         |             0 |            932 |                   66 |                     0 |                    66 |                 0        |                  0.0708155 |
|        0.85 |    932 |         0         |             0 |            932 |                   66 |                     0 |                    66 |                 0        |                  0.0708155 |
|        0.9  |    932 |         0         |             0 |            932 |                   66 |                     0 |                    66 |                 0        |                  0.0708155 |
|        0.95 |    932 |         0         |             0 |            932 |                   66 |                     0 |                    66 |                 0        |                  0.0708155 |
|        1    |    932 |         0         |             0 |            932 |                   66 |                     0 |                    66 |                 0        |                  0.0708155 |
