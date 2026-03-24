"""Detailed analysis of attack types in external datasets vs training data."""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path("/Users/noamc/repos/llm-gatekeeping")
TRAIN_PATH = BASE / "data/processed/splits/train.parquet"
EXT_DIR = BASE / "data/processed/research_external"

# Keywords to scan for in adversarial samples
KEYWORDS = [
    "ignore", "pretend", "DAN", "jailbreak", "bypass", "disregard",
    "override", "forget", "roleplay", "act as", "new instructions",
    "system prompt", "developer mode", "do anything now",
]


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    externals = {}
    for f in sorted(EXT_DIR.glob("research_external_*.parquet")):
        name = f.stem.replace("research_external_", "")
        externals[name] = pd.read_parquet(f)
    return train, externals


def print_separator(char="=", width=90):
    print(char * width)


def analyze_training(train):
    print_separator()
    print("TRAINING DATA OVERVIEW")
    print_separator()
    print(f"Total samples: {len(train)}")
    print()

    # Binary distribution
    print("Binary label distribution:")
    for label, count in train["label_binary"].value_counts().items():
        print(f"  {label:30s}  {count:5d}  ({100*count/len(train):.1f}%)")
    print()

    # Category distribution
    print("Category distribution:")
    for label, count in train["label_category"].value_counts().items():
        print(f"  {label:30s}  {count:5d}  ({100*count/len(train):.1f}%)")
    print()

    # Type distribution
    print("Type (attack_type) distribution:")
    type_counts = train["label_type"].value_counts()
    for label, count in type_counts.items():
        print(f"  {label:30s}  {count:5d}  ({100*count/len(train):.1f}%)")
    print()

    # attack_name distribution (original dataset label)
    if "attack_name" in train.columns:
        print("attack_name distribution (top 30):")
        for label, count in train["attack_name"].value_counts().head(30).items():
            print(f"  {str(label):40s}  {count:5d}")
        print()

    return set(train["label_binary"].dropna().unique()), \
           set(train["label_category"].dropna().unique()), \
           set(train["label_type"].dropna().unique()), \
           set(train["attack_name"].dropna().unique()) if "attack_name" in train.columns else set()


def keyword_analysis(texts):
    """Count keyword occurrences in adversarial texts."""
    results = {}
    lower_texts = texts.str.lower()
    for kw in KEYWORDS:
        count = lower_texts.str.contains(re.escape(kw), case=False, na=False).sum()
        if count > 0:
            results[kw] = count
    return dict(sorted(results.items(), key=lambda x: -x[1]))


def analyze_external(name, df, train_types, train_categories, train_attack_names):
    print_separator()
    print(f"EXTERNAL DATASET: {name.upper()}")
    print_separator()
    print(f"Columns: {list(df.columns)[:15]}{'...' if len(df.columns) > 15 else ''}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Total samples: {len(df)}")
    print()

    # --- 3a: Binary breakdown ---
    print("Binary label distribution:")
    for label, count in df["label_binary"].value_counts().items():
        print(f"  {str(label):30s}  {count:5d}  ({100*count/len(df):.1f}%)")
    print()

    # --- 3b: Unique labels ---
    print("Unique label_category values:")
    for v in sorted(df["label_category"].dropna().unique()):
        print(f"  - {v}")
    print()

    print("Unique label_type values:")
    for v in sorted(df["label_type"].dropna().unique()):
        print(f"  - {v}")
    print()

    # --- 3c: Types NOT in training data ---
    ext_types = set(df["label_type"].dropna().unique())
    ext_categories = set(df["label_category"].dropna().unique())
    novel_types = ext_types - train_types
    novel_categories = ext_categories - train_categories

    if novel_types:
        print(f"label_type values NOT in training data ({len(novel_types)}):")
        for v in sorted(novel_types):
            count = (df["label_type"] == v).sum()
            print(f"  - {v}  (n={count})")
    else:
        print("All label_type values are present in training data.")
    print()

    if novel_categories:
        print(f"label_category values NOT in training data ({len(novel_categories)}):")
        for v in sorted(novel_categories):
            count = (df["label_category"] == v).sum()
            print(f"  - {v}  (n={count})")
    else:
        print("All label_category values are present in training data.")
    print()

    # --- 3d: Text characteristics ---
    adv = df[df["label_binary"] == "adversarial"]
    ben = df[df["label_binary"] == "benign"]

    print("Text characteristics:")
    text_col = "modified_sample"
    lengths = df[text_col].str.len()
    print(f"  Overall  - mean length: {lengths.mean():.0f}, median: {lengths.median():.0f}, "
          f"min: {lengths.min()}, max: {lengths.max()}")
    if len(adv) > 0:
        adv_len = adv[text_col].str.len()
        print(f"  Adversarial - mean length: {adv_len.mean():.0f}, median: {adv_len.median():.0f}, "
              f"min: {adv_len.min()}, max: {adv_len.max()}")
    if len(ben) > 0:
        ben_len = ben[text_col].str.len()
        print(f"  Benign      - mean length: {ben_len.mean():.0f}, median: {ben_len.median():.0f}, "
              f"min: {ben_len.min()}, max: {ben_len.max()}")
    print()

    # Keyword analysis on adversarial samples
    if len(adv) > 0:
        kw_counts = keyword_analysis(adv[text_col])
        if kw_counts:
            print("Keyword occurrences in adversarial samples:")
            for kw, cnt in kw_counts.items():
                print(f"  '{kw}':  {cnt:5d}  ({100*cnt/len(adv):.1f}% of adversarial)")
        else:
            print("No keyword matches found in adversarial samples.")
        print()

    # --- 3e: Example adversarial texts per type grouping ---
    if len(adv) > 0:
        print("Example adversarial texts (first 200 chars) per label_type:")
        for lt in sorted(adv["label_type"].dropna().unique()):
            subset = adv[adv["label_type"] == lt]
            print(f"\n  [{lt}] (n={len(subset)}):")
            for _, row in subset.head(3).iterrows():
                txt = str(row[text_col])[:200].replace("\n", " ")
                print(f"    >>> {txt}")
    print()


def overlap_summary(externals, train_types):
    print_separator()
    print("OVERLAP SUMMARY: % of external adversarial samples with training-matching types")
    print_separator()
    print()

    all_ext_adv = 0
    all_ext_matched = 0

    for name, df in externals.items():
        adv = df[df["label_binary"] == "adversarial"]
        if len(adv) == 0:
            print(f"  {name}: no adversarial samples")
            continue
        matched = adv["label_type"].isin(train_types).sum()
        novel = len(adv) - matched
        pct_matched = 100 * matched / len(adv) if len(adv) > 0 else 0
        print(f"  {name:15s}:  {len(adv):5d} adversarial,  "
              f"{matched:5d} matching training types ({pct_matched:.1f}%),  "
              f"{novel:5d} novel types ({100-pct_matched:.1f}%)")
        all_ext_adv += len(adv)
        all_ext_matched += matched

    print()
    if all_ext_adv > 0:
        pct = 100 * all_ext_matched / all_ext_adv
        print(f"  TOTAL:  {all_ext_adv} adversarial across all external sets,  "
              f"{all_ext_matched} matched ({pct:.1f}%),  "
              f"{all_ext_adv - all_ext_matched} novel ({100-pct:.1f}%)")
    print()


def main():
    train, externals = load_data()

    train_binary, train_categories, train_types, train_attack_names = analyze_training(train)

    for name, df in externals.items():
        analyze_external(name, df, train_types, train_categories, train_attack_names)

    overlap_summary(externals, train_types)


if __name__ == "__main__":
    main()
