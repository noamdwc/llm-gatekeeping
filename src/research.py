"""
Research mode pipeline — comprehensive run capturing all intermediate
probabilities from ML and (optionally) LLM classifiers.

Produces a wide parquet file suitable for offline analysis (calibration
studies, error analysis, threshold tuning).

Usage:
    python -m src.research --split test [--limit N] [--skip-llm] [--force-all-stages]
"""

import argparse

import pandas as pd
from tqdm import tqdm

from src.utils import ROOT, load_config
from src.ml_classifier.ml_baseline import MLBaseline


GROUND_TRUTH_COLS = [
    "modified_sample",
    "original_sample",
    "attack_name",
    "label_binary",
    "label_category",
    "label_type",
    "prompt_hash",
]


def run_ml_full(ml_model: MLBaseline, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run ML predict_full() and return the wide probability DataFrame."""
    return ml_model.predict_full(df, text_col)


def run_llm_full(
    df: pd.DataFrame,
    cfg: dict,
    text_col: str,
    force_all_stages: bool = False,
) -> pd.DataFrame:
    """Run LLM classifier on all samples and return predictions DataFrame.

    Columns: llm_pred_{binary,category,type}, llm_conf_{binary,category,type},
             llm_stages_run
    """
    import dotenv
    dotenv.load_dotenv()

    from src.llm_classifier.llm_classifier import (
        HierarchicalLLMClassifier,
        build_few_shot_examples,
    )

    data_dir = ROOT / "data" / "processed"
    df_train = pd.read_parquet(data_dir / "train.parquet")
    few_shot, _ = build_few_shot_examples(df_train, cfg)
    classifier = HierarchicalLLMClassifier(cfg, few_shot)

    results = classifier.predict_batch(
        df[text_col].tolist(),
        desc="LLM classifying",
        force_all_stages=force_all_stages,
    )

    rows = []
    for r in results:
        rows.append({
            "llm_pred_binary": r["label_binary"],
            "llm_pred_category": r["label_category"],
            "llm_pred_type": r["label_type"],
            "llm_conf_binary": r["confidence_binary"],
            "llm_conf_category": r["confidence_category"],
            "llm_conf_type": r["confidence_type"],
            "llm_stages_run": r.get("llm_stages_run"),
        })
    return pd.DataFrame(rows)


def compute_hybrid_routing(
    ml_df: pd.DataFrame,
    llm_df: pd.DataFrame | None,
    threshold: float,
) -> pd.DataFrame:
    """Compute hybrid routing decisions from ML confidence + threshold.

    If LLM results are available, escalated samples use LLM predictions.
    Otherwise, escalated samples fall back to ML predictions.

    Returns DataFrame with: hybrid_routed_to, hybrid_pred_{binary,category,type}
    """
    n = len(ml_df)
    routed_to = []
    pred_binary = []
    pred_category = []
    pred_type = []

    for i in range(n):
        ml_conf = ml_df.iloc[i]["ml_conf_binary"]
        if ml_conf >= threshold:
            routed_to.append("ml")
            pred_binary.append(ml_df.iloc[i]["ml_pred_binary"])
            pred_category.append(ml_df.iloc[i]["ml_pred_category"])
            pred_type.append(ml_df.iloc[i]["ml_pred_type"])
        else:
            routed_to.append("llm")
            if llm_df is not None:
                pred_binary.append(llm_df.iloc[i]["llm_pred_binary"])
                pred_category.append(llm_df.iloc[i]["llm_pred_category"])
                pred_type.append(llm_df.iloc[i]["llm_pred_type"])
            else:
                # Fallback to ML when LLM was skipped
                pred_binary.append(ml_df.iloc[i]["ml_pred_binary"])
                pred_category.append(ml_df.iloc[i]["ml_pred_category"])
                pred_type.append(ml_df.iloc[i]["ml_pred_type"])

    return pd.DataFrame({
        "hybrid_routed_to": routed_to,
        "hybrid_pred_binary": pred_binary,
        "hybrid_pred_category": pred_category,
        "hybrid_pred_type": pred_type,
    })


def build_research_dataframe(
    df: pd.DataFrame,
    ml_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    llm_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ground truth, ML, LLM, and hybrid results into one wide DataFrame."""
    gt = df[GROUND_TRUTH_COLS].reset_index(drop=True)
    parts = [gt, ml_df.reset_index(drop=True), hybrid_df.reset_index(drop=True)]
    if llm_df is not None:
        parts.insert(2, llm_df.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Research mode: comprehensive analysis run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", help="Which split to run on")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM (ML + hybrid only)")
    parser.add_argument("--force-all-stages", action="store_true",
                        help="Force LLM to run all 3 stages on every sample")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / "data" / "processed"
    text_col = cfg["dataset"]["text_col"]

    # Load split data
    df = pd.read_parquet(data_dir / f"{args.split}.parquet")
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=42)
    print(f"Research run: {len(df)} samples from {args.split} split")

    # ML full probabilities (fast, free)
    print("Running ML predict_full()...")
    ml = MLBaseline(cfg)
    ml.load(str(data_dir / "ml_baseline.pkl"))
    ml_df = run_ml_full(ml, df, text_col)

    # Optional LLM
    llm_df = None
    if not args.skip_llm:
        print("Running LLM classifier...")
        llm_df = run_llm_full(df, cfg, text_col, force_all_stages=args.force_all_stages)

    # Hybrid routing
    threshold = cfg["hybrid"]["ml_confidence_threshold"]
    print(f"Computing hybrid routing (threshold={threshold})...")
    hybrid_df = compute_hybrid_routing(ml_df, llm_df, threshold)

    # Build wide DataFrame
    research_df = build_research_dataframe(df, ml_df, hybrid_df, llm_df)

    # Save
    out_path = data_dir / f"research_{args.split}.parquet"
    research_df.to_parquet(out_path, index=False)
    print(f"\nResearch parquet saved → {out_path}")
    print(f"Shape: {research_df.shape}")
    print(f"Columns: {research_df.columns.tolist()}")

    # Quick sanity check on probabilities
    binary_cols = [c for c in research_df.columns if c.startswith("ml_proba_binary_")]
    if binary_cols:
        sums = research_df[binary_cols].sum(axis=1)
        print(f"\nBinary proba sum: min={sums.min():.6f}, max={sums.max():.6f}")

    # Routing summary
    routing_counts = hybrid_df["hybrid_routed_to"].value_counts()
    print(f"\nRouting: {routing_counts.to_dict()}")


if __name__ == "__main__":
    main()
