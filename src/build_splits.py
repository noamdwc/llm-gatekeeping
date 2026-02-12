"""
Build grouped train/val/test splits with held-out attack types.

Usage:
    python -m src.build_splits [--config configs/default.yaml]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, SPLITS_DIR, load_config


def build_splits(config_path: str = None, input_path: str = None) -> dict[str, pd.DataFrame]:
    """
    Build grouped splits ensuring:
    1. Prompt groups (original + all its modified variants) stay together
    2. Held-out attack types go entirely into a separate test-unseen split
    3. Remaining data split into train/val/test by prompt_hash groups
    """
    cfg = load_config(config_path)
    input_path = Path(input_path) if input_path else DATA_DIR / "full_dataset.parquet"
    df = pd.read_parquet(input_path)

    held_out = set(cfg["labels"]["held_out_attacks"])
    split_cfg = cfg["splits"]
    seed = split_cfg["random_seed"]

    # --- Separate held-out attack types for unseen-attack generalization ---
    mask_held = df[cfg["dataset"]["label_col"]].isin(held_out)
    df_held = df[mask_held].copy()
    df_main = df[~mask_held].copy()

    print(f"Held-out unseen attacks ({len(held_out)} types): {len(df_held)} samples")
    print(f"Main pool: {len(df_main)} samples")

    # --- Group by prompt_hash, split groups ---
    groups = df_main["prompt_hash"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    train_ratio = split_cfg["train"]
    val_ratio = split_cfg["val"]

    n = len(groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    df_train = df_main[df_main["prompt_hash"].isin(train_groups)].copy()
    df_val = df_main[df_main["prompt_hash"].isin(val_groups)].copy()
    df_test = df_main[df_main["prompt_hash"].isin(test_groups)].copy()

    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "test_unseen": df_held,
    }

    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = SPLITS_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"  {name}: {len(split_df)} samples → {path}")

    # Print label distributions
    for name, split_df in splits.items():
        print(f"\n--- {name} label_binary distribution ---")
        print(split_df["label_binary"].value_counts().to_string())
        print(f"\n--- {name} label_category distribution ---")
        print(split_df["label_category"].value_counts().to_string())

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train/val/test splits")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--input", default=None, help="Path to full_dataset.parquet")
    args = parser.parse_args()
    build_splits(args.config, args.input)
