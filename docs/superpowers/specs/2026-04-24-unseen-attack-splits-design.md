# Unseen Attack Splits — Design

**Date:** 2026-04-24
**Status:** Approved (pending implementation)
**Branch:** zero_shot_nlp_attack

## Goal

Expand the held-out attack scheme so the pipeline produces **both** `unseen_val` and `unseen_test` splits. `unseen_val` is a monitoring signal watched during experiments; `unseen_test` is the final generalization number. Neither is used to tune thresholds or hyperparameters.

## Motivation

- Only 20 attack types exist in the dataset (12 unicode + 8 NLP). Disjoint attack types between `unseen_val` and `unseen_test` would shrink training coverage too much. We therefore hold out the **same** attacks for both unseen splits and separate them by `prompt_hash`.
- NLP attacks are the open generalization weakness (unicode classifies at 93–100% per sub-type). Current held-out set (Emoji Smuggling + Pruthi) has only one NLP attack; we expand NLP hold-out.
- Today's `test_unseen` has no benigns, so FPR and benign recall are not measurable on unseen data.

## Decisions

| Topic | Decision |
|---|---|
| Role of `unseen_val` | Monitoring only; never tuned against. |
| Val/test relationship | Same held-out attack types, split by `prompt_hash` (noisier separation is acceptable for monitoring). |
| Focus | NLP attacks are the hold-out priority; Emoji Smuggling retained as a cheap unicode-unseen signal. |
| Held-out attacks | Emoji Smuggling, Pruthi, TextFooler, BAE. |
| Unseen val/test ratio | 50 / 50 of held-out `prompt_hash` values (stratified per attack). |
| Benigns in unseen sets | Yes — dedicated benign hashes, pulled from the benign pool before train/val/test are built, matched to the main-pool adv/benign ratio. |
| Safeguard dataset promotion | **Deferred.** Remains an external-only dataset for now. |

## Held-out attack rationale

Families (coarse) among NLP attacks:

- **Typo / character perturbation:** Pruthi, Deep Word Bug, TextBugger
- **Synonym / embedding substitution:** BAE, Bert-Attack, TextFooler, PWWS, Alzantot

Picks:

- **Pruthi** — kept. Covers typo family.
- **TextFooler** — largest embedding-substitution attack (789 rows, 484 hashes). Strong substitution-family probe.
- **BAE** — BERT-based substitution, mechanistically distinct from TextFooler (641 rows, 389 hashes).
- **Emoji Smuggling** — kept as unicode-unseen signal; low cost to training since unicode sub-types are abundant.

Training retains Bert-Attack, PWWS, Alzantot (substitution family) and Deep Word Bug, TextBugger (typo family), so all NLP families remain represented in train/val/test.

## Config changes (`configs/default.yaml`)

```yaml
labels:
  held_out_attacks:
    - Emoji Smuggling
    - Pruthi
    - TextFooler
    - BAE

splits:
  train: 0.7
  val: 0.15
  test: 0.15
  unseen_val_ratio: 0.5   # fraction of held-out prompt_hashes → unseen_val
  random_seed: 42
```

`unseen_val_ratio` only applies to the held-out adversarial pool. Benign allocation into unseen splits is computed from the main-pool adv/benign ratio — not configured.

## Split algorithm (`src/build_splits.py`)

1. Load `data/processed/full_dataset.parquet`.
2. Partition rows:
   - **Held-out adversarial pool:** rows where `attack_name ∈ held_out_attacks`.
   - **Main pool:** all remaining rows (non-held-out attacks + all benigns).
3. **Held-out adversarial split (stratified):**
   - For each held-out `attack_name`, take its unique `prompt_hash` values, shuffle with `random_seed`, assign the first `unseen_val_ratio` fraction to `unseen_val_adv_hashes`, the rest to `unseen_test_adv_hashes`.
   - Union across attacks → final adversarial hash sets for each unseen split.
4. **Benign allocation for unseen splits:**
   - Compute main-pool adv/benign row ratio `r = n_adv_rows_main / n_benign_rows_main` (counted before any benign hashes are removed).
   - For each unseen split: `target_benign_rows = round(n_unseen_adv_rows / r)`.
   - Shuffle benign `prompt_hash` values once (with `random_seed`) and iterate: assign hashes to `unseen_val` until its accumulated benign row count ≥ its target; then continue assigning to `unseen_test` until its target is reached. The two sets are disjoint by construction.
   - Remove all assigned benign hashes from the benign pool before step 5.
5. **Main split (train / val / test):**
   - Take remaining main-pool `prompt_hash` values (non-held-out adversarial hashes + leftover benign hashes).
   - Shuffle with `random_seed`, split by configured ratios (70 / 15 / 15).
6. Write 5 parquet files to `data/processed/splits/`:
   - `train.parquet`
   - `val.parquet`
   - `test.parquet`
   - `unseen_val.parquet`
   - `unseen_test.parquet`
7. Print per-split label distributions and per-attack counts within the unseen splits (for debugging).

**Invariants verified by code:**

- No `prompt_hash` appears in more than one split.
- Held-out attacks appear only in `unseen_val` and `unseen_test`.
- Benigns in `unseen_val` and `unseen_test` do not overlap with train/val/test benigns.

## Downstream impacts

### ML / DeBERTa / LLM classifiers

Wherever code iterates over split names, replace `test_unseen` with both `unseen_val` and `unseen_test`. No training-logic changes; each classifier simply produces predictions for two additional splits.

Touchpoints:
- `src/ml_classifier/ml_baseline.py` — prediction loop.
- `src/cli/deberta_classifier.py` — evaluation loop.
- `src/llm_classifier/llm_classifier.py` — `--split` CLI accepts the new names.
- `src/research.py` — merges predictions for both new splits.

### DVC (`dvc.yaml`)

- Rename the `test_unseen` split reference to `unseen_test` in every stage that lists splits.
- Add `unseen_val` to the same lists.
- The `research` stage (matrix over splits) gains two new items.
- Main-pool membership changes (TextFooler and BAE are no longer in train/val/test), which invalidates the `build_splits`, `ml_model`, `deberta_model`, `llm_classifier`, and `research` caches. Accepted as a one-time recompute cost.

### Reports

- `reports/research/summary_report.md` gains `unseen_val` and `unseen_test` rows.
- `reports/research/eval_report_{ml,hybrid,llm}.md` include sections for both unseen splits.
- Per-attack breakdown in unseen reports lists all four held-out attacks.

### Output parquets

- `data/processed/predictions/{ml,deberta,llm}_predictions_unseen_val.parquet`
- `data/processed/predictions/{ml,deberta,llm}_predictions_unseen_test.parquet`
- `data/processed/research/research_unseen_val.parquet`
- `data/processed/research/research_unseen_test.parquet`

### Hybrid thresholds / risk model

Thresholds are tuned on `val` (seen). **No change.** `unseen_val` is reported alongside as a monitoring overlay and must **not** feed tuning.

### External datasets

Unchanged. Safeguard remains in `external_datasets` for external-eval only.

## Non-goals

- Promoting safeguard into the main dataset (deferred).
- Any change to the benign-generation pipeline.
- Any tuning against `unseen_val` — explicitly forbidden.
- Disjoint attack types between `unseen_val` and `unseen_test` — rejected due to dataset size.

## Rollout

1. Update `configs/default.yaml`.
2. Update `src/build_splits.py` to produce the five splits with the algorithm above.
3. Update downstream split iteration in ML / DeBERTa / LLM / research modules.
4. Update `dvc.yaml` split lists.
5. Run `dvc repro` end-to-end. Verify report outputs contain both unseen splits and all four held-out attacks.
