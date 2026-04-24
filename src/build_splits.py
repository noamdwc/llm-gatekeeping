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
    """Split held-out prompt_hashes into disjoint val/test sets.

    Held-out attacks commonly share prompt_hash (same original prompt attacked
    multiple ways), so we assign each hash once — globally — then stratify by
    shuffling within each attack-group so both unseen splits see every attack.
    """
    all_hashes: set = set(df_held["prompt_hash"].unique())
    # Primary attack per hash = first attack encountered (stable for stratification).
    primary_attack = (
        df_held.drop_duplicates(subset=["prompt_hash"]).set_index("prompt_hash")["attack_name"]
    )

    val_hashes: set = set()
    test_hashes: set = set()
    for _, group in primary_attack.groupby(primary_attack):
        hashes = group.index.tolist()
        rng.shuffle(hashes)
        n_val = max(1, int(round(len(hashes) * val_ratio))) if len(hashes) >= 2 else 0
        val_hashes.update(hashes[:n_val])
        test_hashes.update(hashes[n_val:])

    assert val_hashes.isdisjoint(test_hashes)
    assert val_hashes | test_hashes == all_hashes
    return val_hashes, test_hashes


def _allocate_benign_hashes(
    df_benign: pd.DataFrame,
    target_rows_val: int,
    target_rows_test: int,
    rng: np.random.RandomState,
) -> tuple[set, set]:
    """Draw disjoint benign hash sets for unseen_val and unseen_test."""
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

    mask_held = df[label_col].isin(held_out)
    df_held = df[mask_held].copy()
    df_main = df[~mask_held].copy()

    df_main_adv = df_main[df_main["label_binary"] == "adversarial"]
    df_main_benign = df_main[df_main["label_binary"] == "benign"]

    print(f"Held-out adversarial pool ({len(held_out)} attacks): {len(df_held)} rows")
    print(f"Main adversarial pool: {len(df_main_adv)} rows")
    print(f"Benign pool: {len(df_main_benign)} rows")

    rng = np.random.RandomState(seed)

    val_adv_hashes, test_adv_hashes = _stratified_hash_split(df_held, unseen_val_ratio, rng)
    df_unseen_val_adv = df_held[df_held["prompt_hash"].isin(val_adv_hashes)].copy()
    df_unseen_test_adv = df_held[df_held["prompt_hash"].isin(test_adv_hashes)].copy()

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

    # Reserve benign rows for unseen by dropping only the benign variants at those
    # hashes. Non-benign rows at the same prompt_hash remain in the main pool —
    # cross-group (main vs unseen) prompt_hash overlap is expected by design
    # (same as the pre-existing test_unseen behavior).
    reserved_benign = val_benign_hashes | test_benign_hashes
    drop_mask = (df_main["label_binary"] == "benign") & df_main["prompt_hash"].isin(reserved_benign)
    df_main_remaining = df_main[~drop_mask].copy()

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

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = SPLITS_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"  {name}: {len(split_df)} rows -> {path}")

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
