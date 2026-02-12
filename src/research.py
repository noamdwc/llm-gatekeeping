"""
Research mode pipeline — reads pre-computed ML and (optionally) LLM prediction
parquets, computes hybrid routing, and produces a wide research parquet plus
evaluation reports.

In the DVC research pipeline:
  - ml_model stage produces: predictions/ml_predictions_{split}.parquet
  - llm_classifier stage produces: predictions/llm_predictions_{split}.parquet
  - This stage merges them + computes hybrid routing + generates reports.

Usage:
    python -m src.research --split test
"""

import argparse

import pandas as pd

from src.utils import (
    load_config,
    PREDICTIONS_DIR, RESEARCH_DIR,
    REPORTS_RESEARCH_DIR,
)
from src.evaluate import (
    binary_metrics, category_metrics, type_metrics,
    calibration_metrics, generate_report,
)


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
    ml_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    llm_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ML predictions, LLM predictions, and hybrid results into one wide DataFrame.

    The ML predictions parquet already contains ground-truth columns, so we
    don't need the original split parquet.
    """
    parts = [ml_df.reset_index(drop=True), hybrid_df.reset_index(drop=True)]
    if llm_df is not None:
        # Insert LLM columns (only the llm_* columns, not duplicated GT)
        llm_cols = [c for c in llm_df.columns if c.startswith("llm_")]
        parts.insert(1, llm_df[llm_cols].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def generate_ml_report(research_df: pd.DataFrame, output_path: str):
    """Generate ML-only evaluation report from the research DataFrame."""
    binary = binary_metrics(research_df["label_binary"], research_df["ml_pred_binary"])
    cat = category_metrics(research_df["label_category"], research_df["ml_pred_category"])
    types = type_metrics(research_df["label_type"], research_df["ml_pred_type"])
    cal = calibration_metrics(
        research_df["label_binary"], research_df["ml_pred_binary"],
        research_df["ml_conf_binary"],
    )
    report = generate_report(research_df, binary, cat, types, cal)
    # Replace title
    report = report.replace("# LLM Classifier Evaluation Report", "# ML Classifier Evaluation Report")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  ML report saved → {output_path}")
    return binary


def generate_hybrid_report(research_df: pd.DataFrame, output_path: str):
    """Generate hybrid evaluation report from the research DataFrame."""
    binary = binary_metrics(research_df["label_binary"], research_df["hybrid_pred_binary"])
    cat = category_metrics(research_df["label_category"], research_df["hybrid_pred_category"])
    types = type_metrics(research_df["label_type"], research_df["hybrid_pred_type"])

    # Use ML confidence as proxy for hybrid confidence
    cal = calibration_metrics(
        research_df["label_binary"], research_df["hybrid_pred_binary"],
        research_df["ml_conf_binary"],
    )

    # Add routing stats as usage info
    routing = research_df["hybrid_routed_to"].value_counts().to_dict()
    usage = {f"routed_{k}": v for k, v in routing.items()}

    report = generate_report(research_df, binary, cat, types, cal, usage)
    report = report.replace("# LLM Classifier Evaluation Report", "# Hybrid Router Evaluation Report")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  Hybrid report saved → {output_path}")
    return binary


def generate_llm_report(research_df: pd.DataFrame, output_path: str):
    """Generate LLM-only evaluation report from the research DataFrame."""
    binary = binary_metrics(research_df["label_binary"], research_df["llm_pred_binary"])
    cat = category_metrics(research_df["label_category"], research_df["llm_pred_category"])
    types = type_metrics(research_df["label_type"], research_df["llm_pred_type"])
    cal = calibration_metrics(
        research_df["label_binary"], research_df["llm_pred_binary"],
        research_df["llm_conf_binary"],
    )
    report = generate_report(research_df, binary, cat, types, cal)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  LLM report saved → {output_path}")
    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Research stage: merge predictions, compute hybrid routing, generate reports"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", help="Which split to run on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = cfg["hybrid"]["ml_confidence_threshold"]

    # ── Read pre-computed ML predictions ─────────────────────────────────────
    ml_path = PREDICTIONS_DIR / f"ml_predictions_{args.split}.parquet"
    if not ml_path.exists():
        raise FileNotFoundError(
            f"ML predictions not found: {ml_path}\n"
            "Run the ml_model stage first (dvc repro ml_model)."
        )
    ml_df = pd.read_parquet(ml_path)
    print(f"Loaded ML predictions: {ml_path} ({len(ml_df)} samples)")

    # ── Read pre-computed LLM predictions (optional) ─────────────────────────
    llm_path = PREDICTIONS_DIR / f"llm_predictions_{args.split}.parquet"
    llm_df = None
    if llm_path.exists():
        llm_df = pd.read_parquet(llm_path)
        print(f"Loaded LLM predictions: {llm_path} ({len(llm_df)} samples)")
    else:
        print(f"No LLM predictions found at {llm_path} — using ML-only hybrid routing")

    # ── Compute hybrid routing ───────────────────────────────────────────────
    print(f"Computing hybrid routing (threshold={threshold})...")
    hybrid_df = compute_hybrid_routing(ml_df, llm_df, threshold)

    # ── Build wide research DataFrame ────────────────────────────────────────
    research_df = build_research_dataframe(ml_df, hybrid_df, llm_df)

    # ── Save research parquet ────────────────────────────────────────────────
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESEARCH_DIR / f"research_{args.split}.parquet"
    research_df.to_parquet(out_path, index=False)
    print(f"\nResearch parquet saved → {out_path}")
    print(f"Shape: {research_df.shape}")
    print(f"Columns: {research_df.columns.tolist()}")

    # ── Generate evaluation reports ──────────────────────────────────────────
    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating reports...")
    generate_ml_report(research_df, str(REPORTS_RESEARCH_DIR / "eval_report_ml.md"))
    generate_hybrid_report(research_df, str(REPORTS_RESEARCH_DIR / "eval_report_hybrid.md"))
    if llm_df is not None:
        generate_llm_report(research_df, str(REPORTS_RESEARCH_DIR / "eval_report_llm.md"))

    # ── Quick sanity check ───────────────────────────────────────────────────
    binary_cols = [c for c in research_df.columns if c.startswith("ml_proba_binary_")]
    if binary_cols:
        sums = research_df[binary_cols].sum(axis=1)
        print(f"\nBinary proba sum: min={sums.min():.6f}, max={sums.max():.6f}")

    routing_counts = hybrid_df["hybrid_routed_to"].value_counts()
    print(f"\nRouting: {routing_counts.to_dict()}")


if __name__ == "__main__":
    main()
