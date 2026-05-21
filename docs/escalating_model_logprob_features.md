# Escalating Model Logprob Features

## Purpose

The escalating model estimates `P(cheap path is wrong)` so the router can later decide when a cheap Colab/local LLM classifier result should be escalated to the stronger judge. The first POC used DeBERTa agreement and confidence-style features. The new feature set adds classifier label-token logprob information from the Colab/local LLM output.

These features are intended to capture how strongly the cheap classifier favored its emitted label. If the model is uncertain between label tokens, that should be useful evidence that the cheap path may need escalation.

## New Features

The notebook now writes top-5 token alternatives into `clf_token_logprobs`. The escalating dataset builder extracts three scalar features from the classifier label-token position:

| Feature | Meaning |
|---|---|
| `clf_top1_logprob` | Logprob of the most likely token at the classifier label-token position. |
| `clf_top2_logprob` | Logprob of the second most likely token at the classifier label-token position. |
| `clf_logprob_diff` | Margin between the first and second alternatives: `clf_top1_logprob - clf_top2_logprob`. |

The margin is the most direct confidence signal: a small diff means the model was close between two token alternatives; a large diff means the chosen label token was much more likely.

## Data Flow

1. `notebooks/colab_local_llm_classifier.ipynb` runs the local classifier and saves `clf_token_logprobs`.
2. Each generated token entry now includes `top_logprobs` with five alternatives.
3. `src/escalating_model.py` parses `clf_token_logprobs`.
4. The escalating model joins Colab/local predictions with DeBERTa predictions by `sample_id`.
5. The model trains on `needs_escalation = llm_pred_binary != label_binary`.

The current saved model uses these features:

```text
llm_conf_binary
clf_confidence
deberta_proba_binary_adversarial
llm_pred_is_adversarial
deberta_pred_is_adversarial
deberta_llm_disagree
llm_distance_from_uncertain
deberta_distance_from_uncertain
clf_top1_logprob
clf_top2_logprob
clf_logprob_diff
```

## Feature Coverage

After regenerating the Colab/local prediction parquets with top-5 logprobs, the features are populated across the evaluation splits:

| Split | `clf_top1_logprob` nonzero | `clf_top2_logprob` nonzero | `clf_logprob_diff` nonzero |
|---|---:|---:|---:|
| val | 2556 / 2557 | 2557 / 2557 | 2539 / 2557 |
| test | 2580 / 2581 | 2581 / 2581 | 2576 / 2581 |
| unseen_val | 1880 / 1881 | 1881 / 1881 | 1870 / 1881 |
| unseen_test | 1894 / 1894 | 1894 / 1894 | 1883 / 1894 |
| safeguard_test | 1552 / 1552 | 1552 / 1552 | 1540 / 1552 |

## Model Impact

The table compares the selected-token-only run against the regenerated top-5 logprob run.

| Split | ROC-AUC Before | ROC-AUC After | PR-AUC Before | PR-AUC After | Top 10% Error Before | Top 10% Error After | Bottom 50% Error Before | Bottom 50% Error After |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 0.8549 | 0.8729 | 0.4909 | 0.5274 | 30.89% | 32.82% | 0.77% | 0.70% |
| unseen_val | 0.8784 | 0.9056 | 0.5884 | 0.6482 | 47.62% | 46.56% | 0.43% | 0.53% |
| unseen_test | 0.8462 | 0.8879 | 0.5552 | 0.6204 | 51.58% | 51.58% | 1.37% | 1.16% |
| safeguard_test | 0.9607 | 0.9650 | 0.8915 | 0.8470 | 42.31% | 40.38% | 0.39% | 0.26% |

## Interpretation

The new top-5 logprob features improve ranking quality on the main target metrics:

- ROC-AUC improves on every split.
- PR-AUC improves on `test`, `unseen_val`, and `unseen_test`.
- The bottom 50% error rate improves on `test`, `unseen_test`, and `safeguard_test`, which is useful for identifying cheap-path rows that are safer to trust.
- The high-score bucket remains strongly enriched for cheap-path mistakes.

The strongest evidence is on the unseen-attack splits. `unseen_val` ROC-AUC improves from `0.8784` to `0.9056`, and `unseen_test` ROC-AUC improves from `0.8462` to `0.8879`. This suggests the logprob features add generalization signal beyond DeBERTa disagreement alone.

One caveat: `safeguard_test` PR-AUC decreases from `0.8915` to `0.8470`, even though ROC-AUC and bottom-half error improve. That means the feature is not uniformly better under every ranking metric; the canonical path therefore relies on the later threshold sweep and final-verdict report rather than this feature analysis alone.
