# Safeguard Into Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `xTRam1/safe-guard-prompt-injection` from external-eval into training data, with safeguard's HF `test` split held out as `safeguard_test.parquet`, in order to address the documented benign-recall weakness (49% on main test).

**Architecture:** Safeguard rows enter at preprocess as binary-only with `category=NaN, type=NaN, source="safeguard"`. They flow through the existing grouped-split logic alongside Mindgard + synthetic. Safeguard's HF `test` split bypasses the normal split logic and is written directly as `safeguard_test.parquet`. Hierarchical category/type training already filters out NLP rows; we add `dropna` for the new NaN rows. Reporting splits per-source on the main `test` and adds a dedicated `safeguard_test` block.

**Tech Stack:** Python 3.14, pandas, scikit-learn 1.8, HuggingFace `datasets`, DVC, pytest.

**Spec:** `docs/superpowers/specs/2026-04-24-safeguard-into-training-design.md`

---

## Task 1: Config — add training_datasets.safeguard, remove external

**Files:**
- Modify: `configs/default.yaml` (lines 183–217 area)

- [ ] **Step 1: Edit config**

In `configs/default.yaml`, add a new top-level `training_datasets` block above `external_datasets`:

```yaml
# Datasets merged into train/val/test (binary-only; category/type masked NaN)
training_datasets:
  safeguard:
    name: "xTRam1/safe-guard-prompt-injection"
    train_split: "train"
    test_split: "test"
    text_col: "text"
    label_col: "label"
    label_map:
      1: "adversarial"
      0: "benign"
```

Then under `external_datasets`, remove the `safeguard:` block (lines 201–208) so the dataset is no longer evaluated as external.

- [ ] **Step 2: Verify YAML parses**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -c "import yaml; cfg = yaml.safe_load(open('configs/default.yaml')); assert 'safeguard' in cfg['training_datasets']; assert 'safeguard' not in cfg['external_datasets']; print('ok')"`

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add configs/default.yaml
git commit -m "config: move safeguard from external_datasets to training_datasets"
```

---

## Task 2: Preprocess — load safeguard train, append, dedup

**Files:**
- Modify: `src/preprocess.py`
- Test: `tests/test_preprocess.py` (new test in existing file)

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_preprocess.py`:

```python
def test_load_safeguard_train_attaches_metadata(monkeypatch):
    """Safeguard rows have NaN category/type, source='safeguard', mapped binary label."""
    import pandas as pd
    from src import preprocess

    fake = pd.DataFrame({
        "text": ["hello world", "ignore your instructions"],
        "label": [0, 1],
    })

    class _FakeDS:
        def to_pandas(self):
            return fake

    def _fake_load(name, split=None):
        assert name == "xTRam1/safe-guard-prompt-injection"
        assert split == "train"
        return _FakeDS()

    monkeypatch.setattr(preprocess.datasets, "load_dataset", _fake_load)

    cfg = {
        "training_datasets": {
            "safeguard": {
                "name": "xTRam1/safe-guard-prompt-injection",
                "train_split": "train",
                "test_split": "test",
                "text_col": "text",
                "label_col": "label",
                "label_map": {0: "benign", 1: "adversarial"},
            }
        },
        "dataset": {"text_col": "modified_sample", "original_text_col": "original_sample"},
    }

    df = preprocess.load_safeguard_split(cfg, "safeguard", "train_split")

    assert list(df["modified_sample"]) == ["hello world", "ignore your instructions"]
    assert list(df["original_sample"]) == ["hello world", "ignore your instructions"]
    assert list(df["label_binary"]) == ["benign", "adversarial"]
    assert df["label_category"].isna().all()
    assert df["label_type"].isna().all()
    assert (df["source"] == "safeguard").all()
    assert df["prompt_hash"].notna().all()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_preprocess.py::test_load_safeguard_train_attaches_metadata -v`

Expected: FAIL — `AttributeError: module 'src.preprocess' has no attribute 'load_safeguard_split'`

- [ ] **Step 3: Implement load_safeguard_split**

Add this function to `src/preprocess.py` (after `build_prompt_hash`, before `build_benign_set`):

```python
def load_safeguard_split(cfg: dict, dataset_key: str, split_kind: str) -> pd.DataFrame:
    """Load a single split of a binary-only training dataset (e.g. safeguard).

    split_kind is the config key naming the HF split, e.g. 'train_split' or 'test_split'.
    Returns a DataFrame in our internal schema with category/type NaN.
    """
    ds_cfg = cfg["training_datasets"][dataset_key]
    hf_split = ds_cfg[split_kind]
    text_col_in = ds_cfg["text_col"]
    label_col_in = ds_cfg["label_col"]
    label_map = ds_cfg["label_map"]

    out_text_col = cfg["dataset"]["text_col"]
    out_orig_col = cfg["dataset"]["original_text_col"]

    raw = datasets.load_dataset(ds_cfg["name"], split=hf_split).to_pandas()

    df = pd.DataFrame({
        out_text_col: raw[text_col_in].astype(str).values,
        out_orig_col: raw[text_col_in].astype(str).values,
        "label_binary": raw[label_col_in].map(label_map).values,
        "label_category": pd.Series([pd.NA] * len(raw), dtype="object"),
        "label_type": pd.Series([pd.NA] * len(raw), dtype="object"),
        "attack_name": pd.Series([pd.NA] * len(raw), dtype="object"),
        "benign_source": pd.Series([pd.NA] * len(raw), dtype="object"),
        "is_synthetic_benign": False,
        "source": dataset_key,
    })
    df["prompt_hash"] = df[out_orig_col].fillna("").apply(build_prompt_hash)
    return df
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_preprocess.py::test_load_safeguard_train_attaches_metadata -v`

Expected: PASS

- [ ] **Step 5: Wire safeguard into the preprocess() pipeline**

Edit `src/preprocess.py` `preprocess()` function. After the existing combine + drop_duplicates block (around lines 168–173) and BEFORE the `df["prompt_hash"]` assignment, modify the flow so that:

1. Mindgard adversarial rows get `source="mindgard"` set after `add_hierarchical_labels`.
2. Benign rows get `source="benign"` set in `build_benign_set` / `add_hierarchical_labels_benign`.
3. After Mindgard + benign concat (with prompt_hash), append safeguard train rows for any keys in `cfg.get("training_datasets", {})`.
4. Dedup the final frame by `prompt_hash`, keeping first (Mindgard wins).

Replace the body of `preprocess()` from "Combine and drop duplicates" through the end of duplicate handling with:

```python
    # Tag sources on Mindgard + benign rows
    df_adv["source"] = "mindgard"
    if "source" not in df_benign.columns:
        df_benign["source"] = "benign"

    # Combine Mindgard + benign first
    text_col = cfg["dataset"]["text_col"]
    df = pd.concat([df_adv, df_benign], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} duplicate text rows from Mindgard+benign ({len(df)} remaining)")

    # Add prompt hash for grouped splitting
    orig_col = cfg["dataset"]["original_text_col"]
    df["prompt_hash"] = df[orig_col].fillna("").apply(build_prompt_hash)

    # Append rows from training_datasets (binary-only; category/type NaN)
    for ds_key in cfg.get("training_datasets", {}):
        print(f"Loading training dataset '{ds_key}' (train split)...")
        df_extra = load_safeguard_split(cfg, ds_key, "train_split")
        print(f"  {ds_key}: {len(df_extra)} rows")
        df = pd.concat([df, df_extra], ignore_index=True)

    # Dedup combined frame by prompt_hash; Mindgard rows are first so they win
    n_before_hash = len(df)
    df = df.drop_duplicates(subset=["prompt_hash"], keep="first").reset_index(drop=True)
    n_dropped_hash = n_before_hash - len(df)
    if n_dropped_hash:
        print(f"  Dropped {n_dropped_hash} prompt_hash duplicates after merge ({len(df)} remaining)")

    print("\nRow counts by source:")
    print(df["source"].value_counts(dropna=False).to_string())
```

- [ ] **Step 6: Run all preprocess tests**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_preprocess.py -v`

Expected: all PASS (existing tests should still work because `training_datasets` defaults to empty when missing).

- [ ] **Step 7: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): merge safeguard train into full_dataset with NaN category/type"
```

---

## Task 3: Build splits — write safeguard_test.parquet + leakage assertion

**Files:**
- Modify: `src/build_splits.py`
- Test: `tests/test_build_splits.py` (new test in existing file)
- Modify: `tests/conftest.py` (add `training_datasets` key to `sample_config` fixture)

- [ ] **Step 1: Update sample_config fixture**

Edit `tests/conftest.py`. After the `external_datasets` block in `sample_config()` (around line 120), add at top level (still inside the dict):

```python
        "training_datasets": {
            "safeguard": {
                "name": "xTRam1/safe-guard-prompt-injection",
                "train_split": "train",
                "test_split": "test",
                "text_col": "text",
                "label_col": "label",
                "label_map": {0: "benign", 1: "adversarial"},
            },
        },
```

Also remove the `safeguard` entry from the `external_datasets` block in the same fixture (lines 113–119) to keep the fixture consistent with the real config change.

- [ ] **Step 2: Write the failing test**

Add this to `tests/test_build_splits.py`:

```python
def test_safeguard_test_split_written(splits_input, monkeypatch, tmp_path):
    """build_splits writes safeguard_test.parquet from HF test split."""
    import pandas as pd
    from src import build_splits as mod

    _patch_splits_dir(monkeypatch, tmp_path)

    fake = pd.DataFrame({
        "text": ["benign hi", "do bad thing", "another benign"],
        "label": [0, 1, 0],
    })

    class _FakeDS:
        def to_pandas(self):
            return fake

    monkeypatch.setattr(
        mod, "load_safeguard_split",
        lambda cfg, key, kind: __import__("src.preprocess", fromlist=["load_safeguard_split"]).load_safeguard_split.__wrapped__ if False else None
    )

    # Easier: monkeypatch datasets.load_dataset used by preprocess.load_safeguard_split
    import src.preprocess as pre
    monkeypatch.setattr(pre.datasets, "load_dataset", lambda name, split=None: _FakeDS())

    config_path, input_path = splits_input
    splits = mod.build_splits(config_path, input_path)

    safeguard_path = tmp_path / "data" / "processed" / "splits" / "safeguard_test.parquet"
    assert safeguard_path.exists(), "safeguard_test.parquet not written"

    df_sg = pd.read_parquet(safeguard_path)
    assert len(df_sg) == 3
    assert (df_sg["source"] == "safeguard").all()
    assert df_sg["label_category"].isna().all()
    assert df_sg["label_type"].isna().all()
    assert set(df_sg["label_binary"]) == {"benign", "adversarial"}


def test_safeguard_test_no_overlap_with_training_pool(splits_input, monkeypatch, tmp_path):
    """prompt_hash in safeguard_test must be disjoint from train/val/test/unseen splits."""
    import pandas as pd
    from src import build_splits as mod
    import src.preprocess as pre

    _patch_splits_dir(monkeypatch, tmp_path)

    fake = pd.DataFrame({"text": ["safeguard only one", "safeguard only two"], "label": [0, 1]})

    class _FakeDS:
        def to_pandas(self):
            return fake

    monkeypatch.setattr(pre.datasets, "load_dataset", lambda name, split=None: _FakeDS())

    config_path, input_path = splits_input
    splits = mod.build_splits(config_path, input_path)

    df_sg = pd.read_parquet(tmp_path / "data" / "processed" / "splits" / "safeguard_test.parquet")
    sg_hashes = set(df_sg["prompt_hash"])
    for name in ("train", "val", "test", "unseen_val", "unseen_test"):
        other = set(splits[name]["prompt_hash"])
        assert sg_hashes.isdisjoint(other), f"safeguard_test overlaps with {name}"
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_build_splits.py::test_safeguard_test_split_written tests/test_build_splits.py::test_safeguard_test_no_overlap_with_training_pool -v`

Expected: FAIL — file not written / function doesn't exist.

- [ ] **Step 4: Implement in build_splits.py**

At the top of `src/build_splits.py`, add the import:

```python
from src.preprocess import load_safeguard_split
```

Inside `build_splits()`, AFTER all main splits are written (after the existing `for name, split_df in splits.items(): ... to_parquet ...` loop, before the per-attack print loop), add:

```python
    # Write held-out safeguard_test split (binary-only)
    for ds_key in cfg.get("training_datasets", {}):
        df_held = load_safeguard_split(cfg, ds_key, "test_split")
        path = SPLITS_DIR / f"{ds_key}_test.parquet"
        df_held.to_parquet(path, index=False)
        print(f"  {ds_key}_test: {len(df_held)} rows -> {path}")

        # Leakage assertion
        held_hashes = set(df_held["prompt_hash"])
        for name in ("train", "val", "test", "unseen_val", "unseen_test"):
            other = set(splits[name]["prompt_hash"])
            overlap = held_hashes & other
            assert not overlap, (
                f"prompt_hash leakage: {ds_key}_test overlaps {name} "
                f"on {len(overlap)} hashes (e.g. {list(overlap)[:3]})"
            )

        splits[f"{ds_key}_test"] = df_held
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_build_splits.py -v`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/build_splits.py tests/test_build_splits.py tests/conftest.py
git commit -m "feat(splits): write safeguard_test.parquet with leakage guard"
```

---

## Task 4: ML baseline — handle NaN category/type, train on safeguard rows for binary

**Files:**
- Modify: `src/ml_classifier/ml_baseline.py`
- Test: `tests/test_ml_baseline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ml_baseline.py`:

```python
def test_ml_baseline_trains_with_nan_category_rows(sample_config):
    """Binary head uses all rows; category/type heads drop NaN-label rows."""
    import pandas as pd
    import numpy as np
    from src.ml_classifier.ml_baseline import MLBaseline

    df = pd.DataFrame({
        "modified_sample": [
            "abc def", "xyz qwe", "lmnop", "another text",
            "safeguard benign one", "safeguard benign two",
            "safeguard adversarial one", "safeguard adversarial two",
        ],
        "original_sample": ["a"] * 8,
        "attack_name": ["Diacritcs", "Zero Width", None, None, None, None, None, None],
        "label_binary": [
            "adversarial", "adversarial", "benign", "benign",
            "benign", "benign", "adversarial", "adversarial",
        ],
        "label_category": [
            "unicode_attack", "unicode_attack", "benign", "benign",
            np.nan, np.nan, np.nan, np.nan,
        ],
        "label_type": [
            "Diacritcs", "Zero Width", "benign", "benign",
            np.nan, np.nan, np.nan, np.nan,
        ],
        "source": ["mindgard"] * 4 + ["safeguard"] * 4,
    })

    model = MLBaseline(sample_config)
    model.fit(df, "modified_sample")

    # Binary head must have learned both classes (saw safeguard rows too)
    assert set(model.label_encoders["label_binary"].classes_) == {"adversarial", "benign"}
    # Category/type heads must NOT have a NaN class
    cat_classes = set(model.label_encoders["label_category"].classes_)
    type_classes = set(model.label_encoders["label_type"].classes_)
    assert "nan" not in {str(c) for c in cat_classes}
    assert "nan" not in {str(c) for c in type_classes}
    assert pd.NA not in cat_classes
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_ml_baseline.py::test_ml_baseline_trains_with_nan_category_rows -v`

Expected: FAIL — likely a `ValueError` or NaN being encoded as a class by `LabelEncoder`.

- [ ] **Step 3: Implement NaN-dropping in fit()**

In `src/ml_classifier/ml_baseline.py`, modify the `fit()` method (around lines 286–312). Replace the per-level loop body so non-binary levels drop NaN-label rows BEFORE encoding/fitting:

```python
    def fit(self, df_train: pd.DataFrame, text_col: str):
        """Train models for all three hierarchy levels."""
        df_train = self._filter_char_attack_training_rows(df_train)

        if "label_binary" in df_train.columns and df_train["label_binary"].nunique() < 2:
            raise ValueError(
                f"ML training data has only one binary class after NLP filtering "
                f"({df_train['label_binary'].unique()}). "
                "Ensure benign samples are present in the training split."
            )

        # Binary features built once on the full training pool (incl. safeguard rows)
        X_full = self._build_features(df_train[text_col], fit=True)

        for level in ["label_binary", "label_category", "label_type"]:
            if level == "label_binary":
                df_level = df_train
                X_level = X_full
            else:
                # Drop rows with NaN labels (e.g. safeguard rows lack category/type)
                mask = df_train[level].notna()
                n_dropped = (~mask).sum()
                if n_dropped:
                    print(f"  [{level}] dropping {n_dropped} NaN-label rows before fit")
                df_level = df_train[mask]
                X_level = X_full[mask.values]

            y = df_level[level].values
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            self.label_encoders[level] = le

            if level == "label_binary":
                model, self.binary_calibrator = self._fit_binary_with_calibration(X_level, y_enc)
            else:
                model = self._fit_level_model(X_level, y_enc, level)

            self.models[level] = model
            print(f"  Trained {level}: {len(le.classes_)} classes")
```

Note: this also removes the bug where `_filter_char_attack_training_rows` would silently include NaN-category rows because the mask `df["label_category"] != "nlp_attack"` returns True for NaN values. Verify by inspecting that helper — if NaN rows survive its filter (they do, by default `!=` semantics on NaN), the new dropna in the per-level loop catches them downstream.

- [ ] **Step 4: Run all ML tests**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_ml_baseline.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml_classifier/ml_baseline.py tests/test_ml_baseline.py
git commit -m "feat(ml): drop NaN category/type rows per-level so safeguard binary-only rows train"
```

---

## Task 5: DeBERTa — predict on safeguard_test

**Files:**
- Modify: `src/cli/deberta_classifier.py`

- [ ] **Step 1: Add safeguard_test to EVAL_SPLITS**

In `src/cli/deberta_classifier.py` line 61, change:

```python
EVAL_SPLITS = ["val", "test", "unseen_val", "unseen_test"]
```

to:

```python
EVAL_SPLITS = ["val", "test", "unseen_val", "unseen_test", "safeguard_test"]
```

- [ ] **Step 2: Verify masking-aware training**

Read `src/models/deberta_classifier.py` (look for the train loop / data loading). Confirm that training data is filtered to non-NaN binary labels (it is — DeBERTa head trains on `label_binary` which is always populated). If the model also trains a category or type head and those code paths exist, ensure they `dropna(subset=[<level>])` analogously to ML. If only binary, no change needed.

If multi-head training exists and lacks dropna, add it. (In practice, the current DeBERTa model is binary-only — confirm by grepping `label_category` in `src/models/`.)

Run: `grep -n "label_category\|label_type" src/models/deberta_classifier.py`

Expected: either no matches (binary-only — no further action), or matches that already filter NaN. If matches exist that don't filter, add `df = df[df[level].notna()]` before the corresponding split's training loader.

- [ ] **Step 3: Smoke run (optional, only if model artifacts exist locally)**

Skip if you'd have to retrain. Otherwise:

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m src.cli.deberta_classifier --predict-only --no-wandb`

Expected: produces `data/processed/predictions/deberta_predictions_safeguard_test.parquet` alongside the others, no exceptions.

- [ ] **Step 4: Commit**

```bash
git add src/cli/deberta_classifier.py
git commit -m "feat(deberta): include safeguard_test in eval splits"
```

---

## Task 6: LLM classifier + research — recognize safeguard_test

**Files:**
- Modify: `src/llm_classifier/llm_classifier.py` (CLI argparse choices for `--split`)
- Modify: `src/research.py` (CLI argparse choices for `--split`)

- [ ] **Step 1: Inspect current --split choices**

Run: `grep -n "add_argument.*split\|choices=" src/llm_classifier/llm_classifier.py src/research.py`

Identify the lines defining `--split` choices (or open-ended).

- [ ] **Step 2: Add safeguard_test to choices (if enumerated)**

If either file has `choices=[...]` for `--split`, append `"safeguard_test"`. If the argument is open-ended (no `choices=`), confirm the code reads `SPLITS_DIR / f"{split}.parquet"` and would naturally accept `--split safeguard_test`. Make no change in the open-ended case — just verify behavior.

For example, if `src/research.py` has:

```python
parser.add_argument("--split", choices=["test", "val", "unseen_val", "unseen_test"])
```

change to:

```python
parser.add_argument("--split", choices=["test", "val", "unseen_val", "unseen_test", "safeguard_test"])
```

- [ ] **Step 3: Verify research.py handles missing category/type predictions gracefully**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_research.py tests/test_evaluate.py -v`

Expected: all PASS. If any test fails because evaluate now sees NaN ground truth, add masking in `src/evaluate.py` `category_metrics` / `type_metrics` to skip NaN ground-truth rows. (Check first: `grep -n "dropna\|notna" src/evaluate.py`.)

If evaluate.py needs masking, add at the top of `category_metrics(df)` and `type_metrics(df)`:

```python
df = df[df["label_category"].notna()].copy()  # or label_type for type_metrics
if len(df) == 0:
    return {"n_samples": 0}
```

- [ ] **Step 4: Commit**

```bash
git add src/llm_classifier/llm_classifier.py src/research.py src/evaluate.py
git commit -m "feat(research): accept safeguard_test split; mask NaN category/type in metrics"
```

---

## Task 7: Per-source binary metrics on main test

**Files:**
- Modify: `src/evaluate.py` (or `src/research.py` reporting section — check which writes `eval_report_*.md`)

- [ ] **Step 1: Locate the report writer for eval_report_*.md**

Run: `grep -rn "eval_report_ml.md\|generate_report\|REPORTS_RESEARCH_DIR" src/ | head`

Find the function that writes the per-mode markdown reports (likely `generate_report()` in `src/evaluate.py` or `src/cli/eval_new.py`).

- [ ] **Step 2: Add per-source binary breakdown**

In the report-generation function, after the existing overall binary-metrics section, add a per-source section. Pseudocode:

```python
if "source" in df.columns:
    lines.append("\n### Binary metrics per source\n")
    lines.append("| source | n | accuracy | adv_recall | benign_recall | f1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for src_name, sub in df.groupby("source", dropna=False):
        m = binary_metrics(sub)
        lines.append(
            f"| {src_name} | {len(sub)} | {m['accuracy']:.3f} | "
            f"{m['adversarial_recall']:.3f} | {m['benign_recall']:.3f} | {m['f1']:.3f} |"
        )
```

Adapt the column names to match what `binary_metrics()` actually returns in this codebase. Inspect by running `grep -n "def binary_metrics" src/evaluate.py` and reading the return dict.

- [ ] **Step 3: Smoke test the report**

Run any existing report-generation test, e.g.:

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_evaluate.py tests/test_research.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/evaluate.py
git commit -m "feat(report): per-source binary breakdown on test reports"
```

---

## Task 8: DVC — wire safeguard_test into stages

**Files:**
- Modify: `dvc.yaml`

- [ ] **Step 1: Add safeguard_test to build_splits outs**

In `dvc.yaml` `build_splits` stage `outs:` (lines 55–59), add:

```yaml
    - data/processed/splits/safeguard_test.parquet
```

- [ ] **Step 2: Add safeguard_test deps + outs to ml_model and deberta_model stages**

In `ml_model` stage:
- Under `deps:`, add `- data/processed/splits/safeguard_test.parquet`
- Under `outs:`, add `- data/processed/predictions/ml_predictions_safeguard_test.parquet`

In `deberta_model` stage:
- Under `deps:`, add `- data/processed/splits/safeguard_test.parquet`
- Under `outs:`, add `- data/processed/predictions/deberta_predictions_safeguard_test.parquet`

- [ ] **Step 3: Add params dep on training_datasets to preprocess + build_splits**

In `preprocess` stage `params:` block, add:

```yaml
      - training_datasets
```

In `build_splits` stage `params:` block, add:

```yaml
      - training_datasets
```

- [ ] **Step 4: Add a research stage for safeguard_test**

After the existing `research_val` stage, add a new `research_safeguard_test` stage modeled on it:

```yaml
  # ── Stage 5d: Research on safeguard_test (binary-only) ──────────────────
  research_safeguard_test:
    cmd: python -m src.research --split safeguard_test
    deps:
    - data/processed/predictions/
    - data/processed/splits/safeguard_test.parquet
    - src/evaluate.py
    - src/hybrid_router.py
    - src/research.py
    - src/utils.py
    params:
    - configs/default.yaml:
      - hybrid
    outs:
    - data/processed/research/research_safeguard_test.parquet
```

(If the LLM classifier should also predict on safeguard_test, that adds API cost — leave out unless explicitly desired. The ML + DeBERTa predictions plus hybrid routing on those are sufficient for the binary numbers.)

- [ ] **Step 5: Validate dvc.yaml syntax**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -c "import yaml; yaml.safe_load(open('dvc.yaml')); print('ok')"`

Expected: `ok`

Run (does not execute anything): `dvc status` (just to confirm DVC parses the file).

Expected: lists pending stages without an error.

- [ ] **Step 6: Commit**

```bash
git add dvc.yaml
git commit -m "build: wire safeguard_test split through ml/deberta/research DVC stages"
```

---

## Task 9: Full pipeline run + manual verification

**Files:** none (verification only).

- [ ] **Step 1: Run preprocess + build_splits**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m src.preprocess && /Users/noamc/miniconda3/envs/llm_gate/bin/python -m src.build_splits`

Expected:
- `data/processed/full_dataset.parquet` shows source counts including ~8.2k safeguard rows.
- `data/processed/splits/safeguard_test.parquet` exists with ~2.0k rows.
- Leakage assertion passes (no exception).

- [ ] **Step 2: Inspect splits**

Run:

```bash
/Users/noamc/miniconda3/envs/llm_gate/bin/python -c "
import pandas as pd
for s in ['train','val','test','safeguard_test']:
    df = pd.read_parquet(f'data/processed/splits/{s}.parquet')
    print(s, len(df))
    if 'source' in df.columns:
        print(df['source'].value_counts(dropna=False).to_dict())
    print(df['label_binary'].value_counts().to_dict())
    print()
"
```

Expected:
- `train`/`val`/`test` contain a mix of `mindgard`, `benign`, `safeguard` sources.
- `safeguard_test` is 100% safeguard, binary-only.

- [ ] **Step 3: Train ML, evaluate**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m src.ml_classifier.ml_baseline --research --no-wandb`

Expected: trains without error, prints "dropping N NaN-label rows" messages for category/type heads, reports `safeguard_test` metrics, writes `ml_predictions_safeguard_test.parquet`.

- [ ] **Step 4: Run remaining DVC stages OR manual research**

If happy with ML output, either run `dvc repro` (slow — includes DeBERTa training) or just:

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m src.research --split safeguard_test`

Expected: produces `data/processed/research/research_safeguard_test.parquet` and a binary-only report.

- [ ] **Step 5: Verify benign recall improvement**

Inspect `reports/research/eval_report_ml.md` (regenerate if needed via the eval stage). Confirm benign recall on the main `test` split is materially higher than the previous 49% baseline. Note the per-source breakdown shows where the lift comes from.

- [ ] **Step 6: Run full test suite**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/ -v`

Expected: all PASS.

- [ ] **Step 7: Final commit (if anything was tweaked during verification)**

```bash
git add -A
git commit -m "chore: post-verification fixups"
```

(Skip if no changes.)

---

## Self-Review Notes

- **Spec coverage:** Tasks 1 (config), 2 (preprocess), 3 (build_splits + safeguard_test + leakage assert), 4 (ML masking), 5 (DeBERTa), 6 (LLM/research split recognition + evaluate masking), 7 (per-source reporting), 8 (DVC), 9 (verification). All spec sections covered.
- **Synthetic benign**: spec says "keep on, unchanged" — no task needed; existing `benign.synthetic` config is untouched.
- **Removal of safeguard from external_datasets**: handled in Task 1 + conftest in Task 3.
- **Dedup direction (Mindgard wins)**: enforced by concat order (Mindgard first) + `keep="first"` in Task 2 Step 5.
