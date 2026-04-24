# Add Safeguard Dataset to Training

## Goal

Move `xTRam1/safe-guard-prompt-injection` from external-evaluation-only to a
training data source, in order to address the documented benign-recall weakness
(currently 49% on the main test set).

Safeguard provides binary labels only (adversarial / benign); category and type
labels are masked (NaN) for safeguard rows. The hierarchical category/type
classifiers train on non-NaN rows only — existing masking pattern. The binary
classifier benefits from the additional ~7.7k benign and ~2.5k adversarial
examples.

## Non-Goals

- No change to category/type label schema or to the held-out attack-type splits
  (`unseen_val`, `unseen_test`).
- No retroactive comparison of "safeguard as external eval" numbers — that
  evaluation goes away because safeguard becomes training data.
- No change to other external eval datasets (deepset, jackhhao).

## Data Flow

```
HuggingFace:
  mindgard (existing)         →  preprocess  →  rows with full hierarchy
  safeguard train (8,236)     →  preprocess  →  rows with category=NaN, type=NaN
  synthetic_benign (existing) →  preprocess  →  unchanged

  All concatenated → dedup by prompt_hash (Mindgard wins on collision)
                                    ↓
                              build_splits  →  train/val/test (grouped by prompt_hash)
                                            +  safeguard_test.parquet (held out, from safeguard's HF test split)
                                            +  unseen_val/unseen_test (unchanged)
```

### Splits after the change

| Split | Composition |
|---|---|
| `train.parquet` | Mindgard train + synthetic train + safeguard-train portion |
| `val.parquet` | Mindgard val + synthetic val + safeguard-train portion |
| `test.parquet` | Mindgard test + synthetic test + safeguard-train portion |
| `safeguard_test.parquet` | Safeguard's HF `test` split (2,060 rows), never seen in training |
| `unseen_val.parquet` / `unseen_test.parquet` | Unchanged (held-out attack types) |

Safeguard rows carry `source="safeguard"`, `category=NaN`, `attack_type=NaN`,
binary label set, `prompt_hash=md5(lower().strip(text))`.

## Component Changes

### `configs/default.yaml`
- Add a `training_datasets.safeguard` block (name, text_col, label_col,
  label_map) — same shape as the current `external_datasets.safeguard`.
- Remove (or comment out) `safeguard` under `external_datasets` to prevent
  data leakage from evaluating on what is now training data.

### `src/preprocess.py`
- After loading Mindgard, load safeguard `train` split via HF `datasets`.
- Map labels via `label_map`, set `attack_type=NaN`, `category=NaN`,
  `prompt_hash`, `source="safeguard"`.
- Concat with Mindgard rows. Synthetic-benign integration unchanged.
- Dedup the combined frame by `prompt_hash`, keeping first occurrence.
  Mindgard rows are concatenated first so they win on collision and we don't
  lose hierarchical metadata.

### `src/build_splits.py`
- Existing grouped-by-`prompt_hash` split logic runs over the full pool.
  No change needed — safeguard rows already have a `prompt_hash`.
- New step: load safeguard's HF `test` split directly, write
  `data/processed/splits/safeguard_test.parquet` with the same schema
  (binary only, NaN category/type, `source="safeguard"`).
- Hard assertion: `prompt_hash` overlap between `safeguard_test` and the union
  of `{train, val, test, unseen_val, unseen_test}` must be empty.

### `src/ml_classifier/ml_baseline.py`
- Binary head trains on all rows (no change).
- Category and type heads must `dropna(subset=[label_col])` before fitting.
  Add the dropna if not already present.

### `src/cli/deberta_classifier.py`
- Drop NaN rows when training non-binary heads.
- Add `safeguard_test` to the per-split prediction loop so it produces
  `data/processed/predictions/deberta_predictions_safeguard_test.parquet`.

### `src/llm_classifier/llm_classifier.py` and `src/research.py`
- Recognize `safeguard_test` as a known split for prediction and reporting.

### `dvc.yaml`
- `build_splits` stage adds `safeguard_test.parquet` to outs.
- Per-split prediction/research stages include `safeguard_test` so it's
  re-evaluated on each `dvc repro`.

### `src/evaluate.py`
- Verify (no expected change): binary metrics compute on safeguard_test;
  category/type metrics auto-skip NaN rows.

## Reporting

**On `safeguard_test` (binary only):**
- accuracy, precision/recall/F1 (per class), AUROC, calibration.
- New `reports/research/eval_report_{ml,hybrid,llm}_safeguard_test.md`.

**On `test` (mixed sources):**
- Binary metrics reported **overall** AND **per-source** (Mindgard / safeguard /
  synthetic). Per-source numbers prevent a single combined number from hiding
  per-dataset behavior.
- Category/type metrics: only non-NaN rows (Mindgard + synthetic adversarials).
  Report header notes which rows are excluded.

**Sanity logs at preprocess / build_splits time:**
- Row counts per `source` in each split.
- `prompt_hash` overlap assertion.
- Class balance per split (adv / benign ratio).

**Existing report headers:** call out that the main `test` split now contains
binary-only rows from safeguard, so binary metrics are no longer directly
comparable to prior runs.

## Testing

- `test_preprocess_safeguard.py` — fixture safeguard rows, verify post-
  preprocess fields (`category=NaN`, `type=NaN`, `source="safeguard"`, valid
  `prompt_hash`, label correctly mapped).
- `test_build_splits_safeguard.py` — verify `safeguard_test.parquet` shape,
  zero overlap with training pool, ratio sanity for safeguard rows in
  train/val/test.
- `test_ml_baseline_masking.py` — fixture with NaN category/type rows; verify
  category/type heads fit only on non-NaN, binary head uses all rows.

Manual post-`dvc repro` verification:
- `reports/research/eval_report_ml_safeguard_test.md` exists with populated
  binary metrics.
- Benign recall on the main `test` split improves vs. the prior 49% baseline.

## Risk and Open Questions

- **Class shift on `test`**: adding ~1.2k binary-only rows to the test split
  changes the binary F1 number. Per-source reporting mitigates the
  interpretability cost.
- **Dedup direction**: Mindgard wins on `prompt_hash` collision so we keep
  hierarchical metadata. If safeguard ever has a more accurate binary label
  for a colliding prompt, we'd silently drop it. Acceptable trade-off.
- **DeBERTa training time** grows with the bigger training set; no action
  unless it becomes a problem.
