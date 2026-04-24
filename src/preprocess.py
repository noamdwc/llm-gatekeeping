"""
Preprocessing pipeline: load dataset, add hierarchical labels, build benign set.

Usage:
    python -m src.preprocess [--config configs/default.yaml]
"""

import argparse
import hashlib
from pathlib import Path

import datasets
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"


def load_config(path: str = None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_dataset(cfg: dict) -> pd.DataFrame:
    """Load from HuggingFace and return a DataFrame."""
    ds = datasets.load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    return ds.to_pandas()


def add_hierarchical_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add level-0 (binary), level-1 (category), level-2 (type) labels."""
    unicode_set = set(cfg["labels"]["unicode_attacks"])
    nlp_set = set(cfg["labels"]["nlp_attacks"])
    label_col = cfg["dataset"]["label_col"]

    df = df.copy()

    # Level 0: everything in raw data is adversarial
    df["label_binary"] = "adversarial"

    # Level 1: category
    def _category(name):
        if name in unicode_set:
            return "unicode_attack"
        if name in nlp_set:
            return "nlp_attack"
        return "unknown"

    df["label_category"] = df[label_col].map(_category)

    # Level 2: specific type (only meaningful for unicode; NLP collapsed)
    def _type(row):
        if row["label_category"] == "unicode_attack":
            return row[label_col]
        return "nlp_attack"  # collapsed

    df["label_type"] = df.apply(_type, axis=1)

    return df


def build_prompt_hash(text: str) -> str:
    """Deterministic hash for grouping prompts."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


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


def build_benign_set(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construct benign examples from unique original prompts.

    Uses de-duplicated original_prompt values as the seed set.
    If benign.synthetic.enabled=True and per-category parquets exist,
    validated synthetic benigns are appended before any resampling.
    """
    orig_col = cfg["dataset"]["original_text_col"]
    text_col = cfg["dataset"]["text_col"]
    target = cfg["benign"]["target_count"]

    # Unique original prompts → benign seed
    originals = df[orig_col].drop_duplicates().dropna().reset_index(drop=True)
    benign = pd.DataFrame({
        text_col: originals,
        orig_col: originals,
        cfg["dataset"]["label_col"]: "benign",
        "label_binary": "benign",
        "label_category": "benign",
        "label_type": "benign",
        "benign_source": "original",
        "is_synthetic_benign": False,
    })

    # Optionally integrate synthetic benigns (per-category parquets)
    synth_cfg = cfg.get("benign", {}).get("synthetic", {})
    if synth_cfg.get("enabled", False):
        synth_dir = Path(synth_cfg.get("output_dir", "data/processed/synthetic_benign"))
        synth_files = sorted(synth_dir.glob("synthetic_benign_*.parquet"))
        if synth_files:
            df_synth = pd.concat([pd.read_parquet(f) for f in synth_files], ignore_index=True)
            # Only include validated synthetic samples
            df_synth_valid = df_synth[df_synth["synth_validated"].astype(bool)].copy()
            if len(df_synth_valid) > 0:
                # Align columns: synthetic rows may have extra synth_* cols (kept as NaN in originals)
                synth_rows = pd.DataFrame({
                    text_col: df_synth_valid["modified_sample"].values,
                    orig_col: df_synth_valid["original_sample"].values,
                    cfg["dataset"]["label_col"]: "benign",
                    "label_binary": "benign",
                    "label_category": "benign",
                    "label_type": "benign",
                    "benign_source": "synthetic_validated",
                    "is_synthetic_benign": True,
                })
                # Carry synthetic metadata columns
                for col in df_synth_valid.columns:
                    if col.startswith("synth_"):
                        synth_rows[col] = df_synth_valid[col].values
                # Deduplicate by prompt_hash before merging
                existing_hashes = set(benign[text_col].apply(
                    lambda t: hashlib.md5(str(t).strip().lower().encode()).hexdigest()[:12]
                ))
                new_hashes = df_synth_valid["prompt_hash"].tolist()
                mask = [h not in existing_hashes for h in new_hashes]
                synth_rows = synth_rows[mask].reset_index(drop=True)
                benign = pd.concat([benign, synth_rows], ignore_index=True)
                print(f"  Integrated {synth_rows.shape[0]} synthetic benign samples from {len(synth_files)} files")
        else:
            raise FileNotFoundError(
                f"benign.synthetic.enabled=True but no parquets found in {synth_dir}.\n"
                "Generate first: python -m src.cli.generate_synthetic_benign --category all"
            )

    # Resample with replacement if still below target
    if len(benign) < target:
        extra_needed = target - len(benign)
        extras = benign.sample(n=extra_needed, replace=True, random_state=42).reset_index(drop=True)
        benign = pd.concat([benign, extras], ignore_index=True)

    return benign.head(target)


def preprocess(config_path: str = None, output_dir: str = None) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns combined DataFrame."""
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Load raw
    print("Loading dataset from HuggingFace...")
    df_raw = load_raw_dataset(cfg)
    print(f"  Raw samples: {len(df_raw)}")

    # Add hierarchical labels
    df_adv = add_hierarchical_labels(df_raw, cfg)
    df_adv["benign_source"] = "adversarial"
    df_adv["is_synthetic_benign"] = False

    # Build benign set
    print("Building benign set...")
    df_benign = build_benign_set(df_raw, cfg)
    df_benign = add_hierarchical_labels_benign(df_benign)
    print(f"  Benign samples: {len(df_benign)}")

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

    # Dedup combined frame by exact text; Mindgard rows are first so they win.
    # We dedup on the text column (not prompt_hash) because Mindgard adversarial
    # variants legitimately share a prompt_hash with their benign original.
    n_before_text = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    n_dropped_text = n_before_text - len(df)
    if n_dropped_text:
        print(f"  Dropped {n_dropped_text} text duplicates after merge ({len(df)} remaining)")

    print("\nRow counts by source:")
    print(df["source"].value_counts(dropna=False).to_string())

    # Save
    path = out / "full_dataset.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} samples → {path}")

    return df


def add_hierarchical_labels_benign(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure benign rows have correct hierarchical labels."""
    df = df.copy()
    df["label_binary"] = "benign"
    df["label_category"] = "benign"
    df["label_type"] = "benign"
    if "benign_source" not in df.columns:
        df["benign_source"] = "original"
    if "is_synthetic_benign" not in df.columns:
        df["is_synthetic_benign"] = False
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the attack dataset")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    preprocess(args.config, args.output_dir)
