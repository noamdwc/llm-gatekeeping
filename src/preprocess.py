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


def build_benign_set(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construct benign examples from unique original prompts.

    Uses de-duplicated original_prompt values as the seed set.
    If benign.synthetic.enabled=True and synthetic_benign.parquet exists,
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
    })

    # Optionally integrate synthetic benigns
    synth_cfg = cfg.get("benign", {}).get("synthetic", {})
    if synth_cfg.get("enabled", False):
        synth_path = Path(synth_cfg.get("output_path", "data/processed/synthetic_benign.parquet"))
        if synth_path.exists():
            df_synth = pd.read_parquet(synth_path)
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
                print(f"  Integrated {synth_rows.shape[0]} synthetic benign samples")
        else:
            print(f"  Warning: benign.synthetic.enabled=True but {synth_path} not found")

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

    # Build benign set
    print("Building benign set...")
    df_benign = build_benign_set(df_raw, cfg)
    df_benign = add_hierarchical_labels_benign(df_benign)
    print(f"  Benign samples: {len(df_benign)}")

    # Combine and drop duplicates
    text_col = cfg["dataset"]["text_col"]
    df = pd.concat([df_adv, df_benign], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} duplicate rows ({len(df)} remaining)")

    # Add prompt hash for grouped splitting
    orig_col = cfg["dataset"]["original_text_col"]
    df["prompt_hash"] = df[orig_col].fillna("").apply(build_prompt_hash)

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
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the attack dataset")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    preprocess(args.config, args.output_dir)
