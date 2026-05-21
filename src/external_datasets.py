"""External dataset loading helpers for configured evaluation datasets."""

import pandas as pd
from datasets import load_dataset


def load_external_dataset(ds_cfg: dict) -> pd.DataFrame:
    """
    Download a HuggingFace dataset and map labels to the project binary schema.

    Returns a DataFrame with:
    - modified_sample: text column renamed from the dataset config
    - label_binary: "adversarial" or "benign"
    - label_category: mirrors label_binary for binary-only external datasets
    - label_type: mirrors label_binary for binary-only external datasets
    """
    ds = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
    df = ds.to_pandas()

    text_col = ds_cfg["text_col"]
    label_col = ds_cfg["label_col"]
    label_map = ds_cfg["label_map"]

    col_dtype = df[label_col].dtype
    if pd.api.types.is_bool_dtype(col_dtype):
        typed_map = {str(k).lower() == "true": v for k, v in label_map.items()}
    elif pd.api.types.is_integer_dtype(col_dtype):
        typed_map = {int(k): v for k, v in label_map.items()}
    else:
        typed_map = {str(k): v for k, v in label_map.items()}

    df["label_binary"] = df[label_col].map(typed_map)

    unmapped = df["label_binary"].isna()
    if unmapped.any():
        print(f"  Warning: dropping {unmapped.sum()} rows with unmapped labels")
        df = df[~unmapped].reset_index(drop=True)

    null_text = df[text_col].isna()
    if null_text.any():
        print(f"  Warning: dropping {null_text.sum()} rows with null text")
        df = df[~null_text].reset_index(drop=True)

    df = df.rename(columns={text_col: "modified_sample"})

    n_before = len(df)
    df = df.drop_duplicates(subset=["modified_sample"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Warning: dropping {n_dropped} duplicate modified_sample rows")

    df["label_category"] = df["label_binary"]
    df["label_type"] = df["label_binary"]

    return df
