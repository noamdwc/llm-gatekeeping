# Unseen Attack Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `test_unseen` split with two new splits — `unseen_val` (monitoring signal) and `unseen_test` (final generalization number) — each containing held-out attacks (Emoji Smuggling, Pruthi, TextFooler, BAE) stratified 50/50 by `prompt_hash`, plus dedicated benigns drawn to match the main-pool adv/benign ratio.

**Architecture:** Modify `src/build_splits.py` to produce five splits instead of four, driven by expanded `held_out_attacks` in config and a new `splits.unseen_val_ratio` knob. Downstream modules (ML baseline, DeBERTa, CLIs, DVC stages) iterate `["unseen_val", "unseen_test"]` wherever they previously referenced `test_unseen`.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn 1.8, DVC, pytest. Environment: `/Users/noamc/miniconda3/envs/llm_gate/bin/python`.

**Spec:** `docs/superpowers/specs/2026-04-24-unseen-attack-splits-design.md`

---

## File Map

- **Modify:** `configs/default.yaml` — held_out_attacks list + new `unseen_val_ratio`
- **Modify:** `src/build_splits.py` — five-split algorithm with benign allocation
- **Modify:** `tests/test_build_splits.py` — new tests for the 5-split contract
- **Modify:** `tests/conftest.py` — enlarge `sample_dataframe` so held-out pool has ≥2 hashes to split
- **Modify:** `src/ml_classifier/ml_baseline.py` — iterate both unseen splits
- **Modify:** `src/cli/deberta_classifier.py` — `EVAL_SPLITS` list
- **Modify:** `src/cli/infer_split.py` — `--split` choices
- **Modify:** `src/cli/run_baseline.py` — `INTERNAL_SPLITS` list
- **Modify:** `dvc.yaml` — rename outs, add unseen_val outs to build_splits / ml_model / deberta_model
- **Modify:** `run_inference.sh` — `--split` doc comments
- **Modify:** `src/cli/README.md`, `CLAUDE.md` — doc references to split names
- **Modify:** `tests/test_cli_infer_split.py` if it enumerates choices

---

## Task 1: Update test fixtures for multi-hash held-out pool

The current `sample_dataframe` fixture has only one held-out prompt_hash (`bbb222` for Homoglyphs), insufficient for testing a 50/50 stratified split. We add two more held-out hashes so the unseen pool can actually split.

**Files:**
- Modify: `tests/conftest.py:123-183` (sample_dataframe)
- Modify: `tests/conftest.py:19-32` (sample_config labels — leave `held_out_attacks` as `["Homoglyphs"]`; we only need more held-out rows)

- [ ] **Step 1: Extend `sample_dataframe` with more Homoglyphs hashes and more benigns**

In `tests/conftest.py`, add the following rows to the list inside `sample_dataframe` (insert before the final closing `]`):

```python
        # Additional held-out Homoglyphs hashes (so the held-out pool has ≥2 hashes to split)
        {"modified_sample": "gооd mоrning", "original_sample": "good morning",
         "attack_name": "Homoglyphs", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Homoglyphs",
         "prompt_hash": "jjj000"},
        {"modified_sample": "hоw аre you", "original_sample": "how are you",
         "attack_name": "Homoglyphs", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Homoglyphs",
         "prompt_hash": "kkk111"},
        # Extra benigns so benign allocation to unseen splits is possible
        {"modified_sample": "please summarize", "original_sample": "please summarize",
         "attack_name": "benign", "label_binary": "benign",
         "label_category": "benign", "label_type": "benign",
         "prompt_hash": "lll222"},
        {"modified_sample": "list three colors", "original_sample": "list three colors",
         "attack_name": "benign", "label_binary": "benign",
         "label_category": "benign", "label_type": "benign",
         "prompt_hash": "mmm333"},
```

- [ ] **Step 2: Add `unseen_val_ratio` to `sample_config["splits"]`**

In `tests/conftest.py`, edit the `splits` block inside `sample_config`:

```python
        "splits": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
            "unseen_val_ratio": 0.5,
            "random_seed": 42,
        },
```

- [ ] **Step 3: Run existing tests to see what breaks**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_build_splits.py -v`
Expected: existing tests still pass (we only added rows and one harmless config key). If any fail, stop and investigate.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: enlarge held-out pool + benigns in sample fixture

Prepares fixture for five-split build_splits tests where the held-out
pool must split 50/50 by prompt_hash and benigns are drawn for unseen
sets from the main benign pool."
```

---

## Task 2: Update config for held-out attacks and unseen_val_ratio

**Files:**
- Modify: `configs/default.yaml:37-46`

- [ ] **Step 1: Edit `labels.held_out_attacks`**

Replace the current block in `configs/default.yaml`:

```yaml
  # Attack types held out entirely for unseen-attack generalization testing.
  # Split 50/50 by prompt_hash between unseen_val (monitoring) and unseen_test (final).
  held_out_attacks:
    - Emoji Smuggling
    - Pruthi
    - TextFooler
    - BAE
```

- [ ] **Step 2: Add `unseen_val_ratio` to splits block**

Replace the `splits:` block:

```yaml
# Data splits
splits:
  train: 0.7
  val: 0.15
  test: 0.15
  unseen_val_ratio: 0.5   # fraction of held-out prompt_hashes → unseen_val
  random_seed: 42
```

- [ ] **Step 3: Commit (no code changes yet, so no tests run)**

```bash
git add configs/default.yaml
git commit -m "config: expand held_out_attacks + add unseen_val_ratio

Holds out Emoji Smuggling, Pruthi, TextFooler, BAE for unseen_val +
unseen_test generalization splits (50/50 by prompt_hash)."
```

---

## Task 3: Write failing tests for the five-split contract (TDD)

Write the tests *before* changing `build_splits.py`. They must fail until Task 4 is complete.

**Files:**
- Modify: `tests/test_build_splits.py` (full rewrite of test class + kept helper)

- [ ] **Step 1: Replace the `TestBuildSplits` class with the new contract**

Replace the file contents of `tests/test_build_splits.py` with:

```python
"""Tests for src.build_splits — grouped splitting logic with unseen_val + unseen_test."""

import yaml
import pandas as pd
import pytest

from src.build_splits import build_splits


@pytest.fixture
def splits_input(tmp_path, sample_config, sample_dataframe):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(sample_config))
    input_path = tmp_path / "full_dataset.parquet"
    sample_dataframe.to_parquet(input_path, index=False)
    return str(config_path), str(input_path)


def _patch_splits_dir(monkeypatch, tmp_path):
    import src.build_splits as mod
    splits_dir = tmp_path / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(mod, "SPLITS_DIR", splits_dir)


class TestBuildSplits:
    """Tests for build_splits() with unseen_val + unseen_test."""

    def test_produces_five_splits(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        assert set(splits.keys()) == {"train", "val", "test", "unseen_val", "unseen_test"}

    def test_no_prompt_hash_overlap_across_all_splits(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        names = ["train", "val", "test", "unseen_val", "unseen_test"]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ha = set(splits[a]["prompt_hash"].unique())
                hb = set(splits[b]["prompt_hash"].unique())
                assert ha.isdisjoint(hb), f"{a}/{b} prompt_hash overlap"

    def test_held_out_attacks_only_in_unseen_splits(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)

        held_out = {"Homoglyphs"}  # from sample_config
        for name in ["train", "val", "test"]:
            attacks = set(splits[name]["attack_name"].unique())
            assert attacks.isdisjoint(held_out), (
                f"held-out attack leaked into {name}: {attacks & held_out}"
            )
        for name in ["unseen_val", "unseen_test"]:
            adv_rows = splits[name][splits[name]["label_binary"] == "adversarial"]
            adv_attacks = set(adv_rows["attack_name"].unique())
            assert adv_attacks.issubset(held_out), (
                f"non-held-out attack in {name}: {adv_attacks - held_out}"
            )

    def test_unseen_splits_stratified_per_attack(self, splits_input, monkeypatch, tmp_path):
        """Each held-out attack's prompt_hashes split ~50/50 across unseen_val and unseen_test."""
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)

        # For each held-out attack, at least one hash in unseen_val and one in unseen_test
        # (sample_dataframe provides 3 Homoglyphs hashes → 50/50 of 3 gives 1 and 2).
        for attack in {"Homoglyphs"}:
            v_hashes = set(splits["unseen_val"].query("attack_name == @attack")["prompt_hash"])
            t_hashes = set(splits["unseen_test"].query("attack_name == @attack")["prompt_hash"])
            assert len(v_hashes) >= 1, f"{attack} missing from unseen_val"
            assert len(t_hashes) >= 1, f"{attack} missing from unseen_test"

    def test_unseen_splits_contain_benigns(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        for name in ["unseen_val", "unseen_test"]:
            n_benign = (splits[name]["label_binary"] == "benign").sum()
            assert n_benign >= 1, f"{name} has no benign rows"

    def test_benigns_do_not_overlap_across_splits(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        b_hashes = {}
        for name in ["train", "val", "test", "unseen_val", "unseen_test"]:
            b = splits[name][splits[name]["label_binary"] == "benign"]
            b_hashes[name] = set(b["prompt_hash"].unique())
        names = list(b_hashes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert b_hashes[a].isdisjoint(b_hashes[b]), f"benign overlap {a}/{b}"

    def test_all_samples_accounted_for(self, splits_input, monkeypatch, tmp_path, sample_dataframe):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        total = sum(len(s) for s in splits.values())
        assert total == len(sample_dataframe)

    def test_reproducible_with_same_seed(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits1 = build_splits(config_path, input_path)
        splits2 = build_splits(config_path, input_path)
        for name in ["train", "val", "test", "unseen_val", "unseen_test"]:
            pd.testing.assert_frame_equal(
                splits1[name].reset_index(drop=True),
                splits2[name].reset_index(drop=True),
            )
```

- [ ] **Step 2: Run tests — expect failures**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_build_splits.py -v`
Expected: multiple FAILs (e.g. `test_produces_five_splits` fails because `test_unseen` key still exists). This is the red phase of TDD.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_build_splits.py
git commit -m "test: five-split contract for build_splits (failing, TDD red)"
```

---

## Task 4: Implement the five-split algorithm in build_splits.py

**Files:**
- Modify: `src/build_splits.py` (rewrite body of `build_splits`)

- [ ] **Step 1: Replace `build_splits` with the new algorithm**

Replace the full contents of `src/build_splits.py` with:

```python
"""
Build grouped splits: train, val, test, unseen_val, unseen_test.

unseen_val and unseen_test share the same held-out attack types, split
50/50 by prompt_hash (per-attack stratified), with dedicated benigns drawn
to match the main-pool adv/benign ratio.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, SPLITS_DIR, load_config


def _stratified_hash_split(
    df_held: pd.DataFrame,
    val_ratio: float,
    rng: np.random.RandomState,
) -> tuple[set, set]:
    """Per-attack 50/50 split of prompt_hashes → (val_hashes, test_hashes)."""
    val_hashes: set = set()
    test_hashes: set = set()
    for attack, sub in df_held.groupby("attack_name"):
        hashes = sub["prompt_hash"].unique().tolist()
        rng.shuffle(hashes)
        n_val = max(1, int(round(len(hashes) * val_ratio))) if len(hashes) >= 2 else 0
        val_hashes.update(hashes[:n_val])
        test_hashes.update(hashes[n_val:])
    return val_hashes, test_hashes


def _allocate_benign_hashes(
    df_benign: pd.DataFrame,
    target_rows_val: int,
    target_rows_test: int,
    rng: np.random.RandomState,
) -> tuple[set, set]:
    """Draw disjoint benign hash sets for unseen_val and unseen_test.

    Iterates shuffled benign hashes and assigns them to unseen_val until that
    split's accumulated benign row count meets target_rows_val, then to
    unseen_test until target_rows_test is met.
    """
    hashes = df_benign["prompt_hash"].unique().tolist()
    rng.shuffle(hashes)
    rows_per_hash = df_benign.groupby("prompt_hash").size().to_dict()

    val_hashes: set = set()
    test_hashes: set = set()
    val_rows = 0
    test_rows = 0
    for h in hashes:
        n = rows_per_hash[h]
        if val_rows < target_rows_val:
            val_hashes.add(h)
            val_rows += n
        elif test_rows < target_rows_test:
            test_hashes.add(h)
            test_rows += n
        else:
            break
    return val_hashes, test_hashes


def build_splits(config_path: str = None, input_path: str = None) -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    input_path = Path(input_path) if input_path else DATA_DIR / "full_dataset.parquet"
    df = pd.read_parquet(input_path)

    held_out = set(cfg["labels"]["held_out_attacks"])
    split_cfg = cfg["splits"]
    seed = split_cfg["random_seed"]
    unseen_val_ratio = split_cfg["unseen_val_ratio"]

    label_col = cfg["dataset"]["label_col"]

    # 1. Partition rows
    mask_held = df[label_col].isin(held_out)
    df_held = df[mask_held].copy()
    df_main = df[~mask_held].copy()

    df_main_adv = df_main[df_main["label_binary"] == "adversarial"]
    df_main_benign = df_main[df_main["label_binary"] == "benign"]

    print(f"Held-out adversarial pool ({len(held_out)} attacks): {len(df_held)} rows")
    print(f"Main adversarial pool: {len(df_main_adv)} rows")
    print(f"Benign pool: {len(df_main_benign)} rows")

    rng = np.random.RandomState(seed)

    # 2. Held-out adversarial split (stratified per attack)
    val_adv_hashes, test_adv_hashes = _stratified_hash_split(df_held, unseen_val_ratio, rng)
    df_unseen_val_adv = df_held[df_held["prompt_hash"].isin(val_adv_hashes)].copy()
    df_unseen_test_adv = df_held[df_held["prompt_hash"].isin(test_adv_hashes)].copy()

    # 3. Benign allocation for unseen splits — match main-pool adv/benign ratio
    if len(df_main_benign) > 0 and len(df_main_adv) > 0:
        r = len(df_main_adv) / len(df_main_benign)
        target_v = int(round(len(df_unseen_val_adv) / r))
        target_t = int(round(len(df_unseen_test_adv) / r))
    else:
        target_v = 0
        target_t = 0

    val_benign_hashes, test_benign_hashes = _allocate_benign_hashes(
        df_main_benign, target_v, target_t, rng
    )
    df_unseen_val_benign = df_main_benign[df_main_benign["prompt_hash"].isin(val_benign_hashes)].copy()
    df_unseen_test_benign = df_main_benign[df_main_benign["prompt_hash"].isin(test_benign_hashes)].copy()

    df_unseen_val = pd.concat([df_unseen_val_adv, df_unseen_val_benign], ignore_index=True)
    df_unseen_test = pd.concat([df_unseen_test_adv, df_unseen_test_benign], ignore_index=True)

    # 4. Main split (train/val/test) from remaining main-pool hashes
    reserved_benign = val_benign_hashes | test_benign_hashes
    df_main_remaining = df_main[~df_main["prompt_hash"].isin(reserved_benign)].copy()

    groups = df_main_remaining["prompt_hash"].unique()
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    df_train = df_main_remaining[df_main_remaining["prompt_hash"].isin(train_groups)].copy()
    df_val = df_main_remaining[df_main_remaining["prompt_hash"].isin(val_groups)].copy()
    df_test = df_main_remaining[df_main_remaining["prompt_hash"].isin(test_groups)].copy()

    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "unseen_val": df_unseen_val,
        "unseen_test": df_unseen_test,
    }

    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = SPLITS_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"  {name}: {len(split_df)} rows → {path}")

    # Diagnostics
    for name, split_df in splits.items():
        print(f"\n--- {name} label_binary distribution ---")
        print(split_df["label_binary"].value_counts().to_string())
    for name in ("unseen_val", "unseen_test"):
        print(f"\n--- {name} per-attack counts (adversarial only) ---")
        adv = splits[name][splits[name]["label_binary"] == "adversarial"]
        print(adv["attack_name"].value_counts().to_string())

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train/val/test/unseen_val/unseen_test splits")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--input", default=None, help="Path to full_dataset.parquet")
    args = parser.parse_args()
    build_splits(args.config, args.input)
```

- [ ] **Step 2: Run tests — expect all pass**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_build_splits.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 3: Run the real splitter against the real dataset and sanity-check counts**

Run:
```bash
/Users/noamc/miniconda3/envs/llm_gate/bin/python -m src.build_splits
```

Expected output includes lines like:
- `Held-out adversarial pool (4 attacks): ~2351 rows` (Emoji Smuggling + Pruthi 269 + TextFooler 789 + BAE 641)
- `unseen_val: <several hundred> rows`, `unseen_test: <several hundred> rows`
- Per-attack table showing all 4 held-out attacks present in each unseen split

If any split is empty or missing an expected held-out attack, stop and investigate.

- [ ] **Step 4: Commit**

```bash
git add src/build_splits.py
git commit -m "feat: produce unseen_val + unseen_test splits

Replaces test_unseen with two splits stratified 50/50 by prompt_hash
across the held-out attacks, with dedicated benigns drawn from the
benign pool to match main-pool adv/benign ratio."
```

---

## Task 5: Update ML baseline to evaluate both unseen splits

**Files:**
- Modify: `src/ml_classifier/ml_baseline.py:525-550` (around the block that loads `test_unseen`)

- [ ] **Step 1: Read the current unseen block**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -c "print(open('src/ml_classifier/ml_baseline.py').read())" | sed -n '520,555p'`
Confirm the block looks like:
```python
unseen_path = SPLITS_DIR / "test_unseen.parquet"
if unseen_path.exists():
    df_unseen = pd.read_parquet(unseen_path)
    evaluate_ml(model, df_unseen, text_col, "test_unseen")
    ...
    save_research_predictions(model, df_unseen, text_col, "test_unseen")
```

- [ ] **Step 2: Replace the block with a loop over both unseen splits**

Edit `src/ml_classifier/ml_baseline.py`: replace the existing `unseen_path`/`df_unseen` block with:

```python
for unseen_name in ("unseen_val", "unseen_test"):
    unseen_path = SPLITS_DIR / f"{unseen_name}.parquet"
    if unseen_path.exists():
        df_unseen = pd.read_parquet(unseen_path)
        evaluate_ml(model, df_unseen, text_col, unseen_name)
        if save_research:
            save_research_predictions(model, df_unseen, text_col, unseen_name)
```

(Preserve whatever `save_research` flag name the existing code uses — adapt if it's called something different there.)

- [ ] **Step 3: Run the ML baseline test**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_ml_baseline.py -v`
Expected: PASS (the test uses `sample_dataframe` and shouldn't depend on the unseen block directly). If it does reference `test_unseen`, update it to `unseen_test` inline.

- [ ] **Step 4: Commit**

```bash
git add src/ml_classifier/ml_baseline.py
git commit -m "feat(ml): evaluate both unseen_val and unseen_test"
```

---

## Task 6: Update DeBERTa classifier EVAL_SPLITS

**Files:**
- Modify: `src/cli/deberta_classifier.py:61`

- [ ] **Step 1: Change EVAL_SPLITS**

Edit `src/cli/deberta_classifier.py`:

```python
EVAL_SPLITS = ["val", "test", "unseen_val", "unseen_test"]
```

- [ ] **Step 2: Run DeBERTa tests**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_deberta_classifier.py -v`
Expected: PASS. If a test enumerates `test_unseen`, replace with `unseen_test` inline.

- [ ] **Step 3: Commit**

```bash
git add src/cli/deberta_classifier.py
git commit -m "feat(deberta): evaluate on unseen_val and unseen_test"
```

---

## Task 7: Update CLI split choices

**Files:**
- Modify: `src/cli/infer_split.py:132`
- Modify: `src/cli/run_baseline.py:13`
- Modify: `tests/test_cli_infer_split.py` (if it references `test_unseen`)
- Modify: `run_inference.sh:12-16` (comments)

- [ ] **Step 1: Update `infer_split.py`**

Replace:
```python
parser.add_argument("--split", default="test", choices=["test", "val", "test_unseen"])
```
with:
```python
parser.add_argument("--split", default="test", choices=["test", "val", "unseen_val", "unseen_test"])
```

- [ ] **Step 2: Update `run_baseline.py`**

Replace:
```python
INTERNAL_SPLITS = ["val", "test", "test_unseen"]
```
with:
```python
INTERNAL_SPLITS = ["val", "test", "unseen_val", "unseen_test"]
```

- [ ] **Step 3: Update `run_inference.sh` usage comments**

In `run_inference.sh`, replace the two `test_unseen` mentions in comments:

```bash
#   ./run_inference.sh --mode ml --split unseen_test           # Generalization test
...
#   --split SPLIT     test | val | unseen_val | unseen_test (default: test)
```

- [ ] **Step 4: Update `tests/test_cli_infer_split.py` if it enumerates `test_unseen`**

Run: `grep -n "test_unseen" tests/test_cli_infer_split.py`
If any matches, replace each with `unseen_test` (or add `unseen_val` where a full choices list is tested).

- [ ] **Step 5: Run CLI tests**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/test_cli_infer_split.py tests/test_baselines.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli/infer_split.py src/cli/run_baseline.py run_inference.sh tests/test_cli_infer_split.py
git commit -m "feat(cli): accept unseen_val and unseen_test split names"
```

---

## Task 8: Update DVC pipeline outputs

**Files:**
- Modify: `dvc.yaml:58, 69, 77, 89, 100` (rename `test_unseen` → `unseen_test`, add `unseen_val`)

- [ ] **Step 1: Edit `build_splits` stage outs**

Replace:
```yaml
    outs:
    - data/processed/splits/train.parquet
    - data/processed/splits/val.parquet
    - data/processed/splits/test.parquet
    - data/processed/splits/test_unseen.parquet
```
with:
```yaml
    outs:
    - data/processed/splits/train.parquet
    - data/processed/splits/val.parquet
    - data/processed/splits/test.parquet
    - data/processed/splits/unseen_val.parquet
    - data/processed/splits/unseen_test.parquet
```

- [ ] **Step 2: Edit `ml_model` stage deps + outs**

In the `ml_model` stage, replace the split deps block and the prediction outs block:

```yaml
    deps:
    - src/ml_classifier/
    - src/utils.py
    - data/processed/splits/train.parquet
    - data/processed/splits/val.parquet
    - data/processed/splits/test.parquet
    - data/processed/splits/unseen_val.parquet
    - data/processed/splits/unseen_test.parquet
```

```yaml
    outs:
    - data/processed/models/ml_baseline.pkl
    - data/processed/predictions/ml_predictions_test.parquet
    - data/processed/predictions/ml_predictions_val.parquet
    - data/processed/predictions/ml_predictions_unseen_val.parquet
    - data/processed/predictions/ml_predictions_unseen_test.parquet
```

- [ ] **Step 3: Edit `deberta_model` stage deps + outs**

Same pattern — replace `test_unseen` with `unseen_test` and add `unseen_val`:

```yaml
    deps:
    - src/models/
    - src/cli/deberta_classifier.py
    - src/utils.py
    - data/processed/splits/train.parquet
    - data/processed/splits/val.parquet
    - data/processed/splits/test.parquet
    - data/processed/splits/unseen_val.parquet
    - data/processed/splits/unseen_test.parquet
```

```yaml
    outs:
    - artifacts/deberta_classifier/model/
    - artifacts/deberta_classifier/tokenizer/
    - artifacts/deberta_classifier/label_mapping.json
    - artifacts/deberta_classifier/train_history.json
    - data/processed/predictions/deberta_predictions_test.parquet
    - data/processed/predictions/deberta_predictions_val.parquet
    - data/processed/predictions/deberta_predictions_unseen_val.parquet
    - data/processed/predictions/deberta_predictions_unseen_test.parquet
    - reports/deberta_classifier/metrics.json:
        cache: false
    - reports/deberta_classifier/classification_report.json:
        cache: false
    - reports/deberta_classifier/summary.md:
        cache: false
```

- [ ] **Step 4: Validate DVC parses the file**

Run: `dvc stage list`
Expected: prints all stages without YAML errors. If a parse error appears, re-check indentation.

- [ ] **Step 5: Commit**

```bash
git add dvc.yaml
git commit -m "build: rename test_unseen to unseen_test and add unseen_val

Renames DVC outputs for the split and every downstream prediction file,
and adds unseen_val prediction outputs for ml_model and deberta_model."
```

---

## Task 9: Update documentation references

**Files:**
- Modify: `src/cli/README.md:14` (split filename mentions)
- Modify: `CLAUDE.md` (any mentions of `test_unseen`)

- [ ] **Step 1: Grep for remaining `test_unseen` references**

Run: `grep -rn "test_unseen" src/ docs/ CLAUDE.md README.md 2>/dev/null | grep -v -E "\.pyc|__pycache__"`

- [ ] **Step 2: Replace each match**

For each remaining occurrence in `.md` files, replace `test_unseen` with `unseen_test` (and mention `unseen_val` where enumerating all splits). Do **not** modify `.pyc` files or the spec/plan docs in `docs/superpowers/` (those keep historical record).

- [ ] **Step 3: Verify no stale code references remain**

Run: `grep -rn "test_unseen" src/ dvc.yaml run_inference.sh configs/`
Expected: no matches (or only inside comments explicitly describing the old name in a historical note).

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "docs: rename test_unseen references to unseen_test / unseen_val"
```

---

## Task 10: Run the full test suite and DVC pipeline

- [ ] **Step 1: Run full test suite**

Run: `/Users/noamc/miniconda3/envs/llm_gate/bin/python -m pytest tests/ -v`
Expected: all PASS. Failures here indicate a missed rename — address inline before proceeding.

- [ ] **Step 2: Reproduce the DVC pipeline**

Run: `dvc repro`

Expected:
- `build_splits` re-runs (new outs). Output includes all 4 held-out attacks in per-attack counts.
- `ml_model`, `deberta_model`, `llm_classifier`, `research` stages re-run (main pool changed because TextFooler + BAE are no longer in train/val/test).
- New prediction files exist at `data/processed/predictions/{ml,deberta}_predictions_{unseen_val,unseen_test}.parquet`.

- [ ] **Step 3: Spot-check unseen reports**

Inspect:
- `reports/deberta_classifier/summary.md` — should show rows/metrics for both unseen_val and unseen_test.
- `reports/research/summary_report.md` — should list both unseen splits.

If reports still show `test_unseen` or miss `unseen_val`, the report generator (`src/research.py` or `src/cli/eval_new.py`) likely enumerates splits from a hard-coded list. Grep and fix inline:

```bash
grep -rn "test_unseen" src/research.py src/cli/eval_new.py
```

For each match, replace with the appropriate `unseen_val` + `unseen_test` handling. Add a commit for any such fixes.

- [ ] **Step 4: Final verification commit (if any fixes were made)**

```bash
git add -u
git commit -m "fix: include unseen_val and unseen_test in reports"
```

- [ ] **Step 5: Summarize changes in the branch**

Run: `git log --oneline main..HEAD`
Confirm the commit history reads as a coherent progression: fixture → config → tests → implementation → downstream → DVC → docs → final verification.

---

## Self-Review Notes

- **Spec coverage:**
  - Config changes (spec §Config) → Task 2 ✓
  - Split algorithm (spec §Algorithm steps 1–7) → Task 4 ✓
  - Downstream ML/DeBERTa/LLM iteration (spec §Downstream) → Tasks 5, 6, 7 ✓
  - DVC rename + new outs (spec §DVC) → Task 8 ✓
  - Reports cover both unseen splits (spec §Reports) → Task 10 spot-check + fix ✓
  - Hybrid thresholds untouched (spec §Hybrid — "No change") → no task needed ✓
  - External datasets untouched (spec §External — "Unchanged") → no task needed ✓
  - Non-goals (safeguard promotion, benign-gen changes) → explicitly not addressed ✓
- **Placeholder scan:** no TBDs. Every code step shows exact code; commands have expected output criteria.
- **Type consistency:** `unseen_val` / `unseen_test` used consistently across all tasks; `unseen_val_ratio` config key used in Task 2 and consumed in Task 4; test names match their assertions.
