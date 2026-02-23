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

import numpy as np
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


ADVERSARIAL_LABEL_ALIASES = {
    "adversarial",
    "adversary",
    "adv",
    "attack",
    "attacker",
    "malicious",
    "jailbreak",
    "prompt_injection",
    "prompt injection",
    "injection",
}


def _normalize_binary_label(label) -> str:
    return str(label).strip().lower().replace("-", "_")


def _is_adversarial_label(label) -> bool:
    norm = _normalize_binary_label(label)
    return norm in ADVERSARIAL_LABEL_ALIASES or norm.startswith("adv")


def compute_hybrid_routing(
    ml_df: pd.DataFrame,
    llm_df: pd.DataFrame | None,
    threshold: float,
) -> pd.DataFrame:
    """Compute hybrid routing decisions from ML prediction + confidence threshold.

    If LLM results are available, escalated samples use LLM predictions.
    Otherwise, escalated samples fall back to ML predictions.
    Rows are matched between ml_df and llm_df via the ``sample_id`` column.

    Specialist policy:
      - ML benign (or non-adversarial) predictions always escalate to LLM.
      - ML adversarial predictions route to ML only when confidence >= threshold.

    Returns DataFrame with: sample_id, hybrid_routed_to, hybrid_pred_{binary,category,type}
    """
    conf_col = "ml_conf_binary_cal" if "ml_conf_binary_cal" in ml_df.columns else "ml_conf_binary"
    ml_conf = ml_df[conf_col].values
    ml_pred_binary = ml_df["ml_pred_binary"].values

    ml_adv_mask = np.array([_is_adversarial_label(v) for v in ml_pred_binary], dtype=bool)
    confident = ml_adv_mask & (ml_conf >= threshold)

    # Start with ML predictions for every row (default / fallback)
    result = pd.DataFrame({
        "sample_id": ml_df["sample_id"].values,
        "hybrid_routed_to": np.where(confident, "ml", "llm"),
        "hybrid_pred_binary": ml_df["ml_pred_binary"].values,
        "hybrid_pred_category": ml_df["ml_pred_category"].values,
        "hybrid_pred_type": ml_df["ml_pred_type"].values,
    })

    if llm_df is not None:
        escalated = ~confident
        llm_indexed = llm_df.set_index("sample_id")

        # Among escalated rows, find which have matching LLM predictions
        esc_ids = result.loc[escalated, "sample_id"]
        has_llm = esc_ids.isin(llm_indexed.index)

        # Override escalated rows that have LLM predictions
        override_idx = has_llm[has_llm].index
        if len(override_idx) > 0:
            matched_ids = result.loc[override_idx, "sample_id"]
            llm_rows = llm_indexed.loc[matched_ids]
            result.loc[override_idx, "hybrid_pred_binary"] = llm_rows["llm_pred_binary"].values
            # Override category if LLM provides it (llm_pred_category exists when LLM derives category)
            if "llm_pred_category" in llm_indexed.columns:
                result.loc[override_idx, "hybrid_pred_category"] = llm_rows["llm_pred_category"].values
            # LLM does not provide type-level predictions; hybrid_pred_type stays as ML's prediction

        # Escalated rows without LLM fall back to ML (predictions already set);
        # correct routing label from "llm" → "ml"
        no_llm_idx = has_llm[~has_llm].index
        result.loc[no_llm_idx, "hybrid_routed_to"] = "ml"
    else:
        # No LLM at all — everything routes to ML
        result["hybrid_routed_to"] = "ml"

    return result


def build_research_dataframe(
    ml_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    llm_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ML predictions, LLM predictions, and hybrid results into one wide DataFrame.

    All DataFrames are joined on ``sample_id`` so row order doesn't matter.
    The ML predictions parquet already contains ground-truth columns, so we
    don't need the original split parquet.
    """
    result = ml_df.merge(hybrid_df, on="sample_id", validate="one_to_one")
    if llm_df is not None:
        llm_cols = ["sample_id"] + [c for c in llm_df.columns if c.startswith("llm_")]
        result = result.merge(llm_df[llm_cols], on="sample_id", how="left", validate="one_to_one")
    return result


def generate_ml_report(research_df: pd.DataFrame, output_path: str):
    """Generate ML-only evaluation report — evaluated on ML domain only (benign + unicode)."""
    # ML is a unicode specialist; NLP attacks are intentionally deferred to LLM
    df = research_df[research_df["label_category"] != "nlp_attack"].copy()
    n_excluded = len(research_df) - len(df)

    binary = binary_metrics(df["label_binary"], df["ml_pred_binary"])
    cat = category_metrics(df["label_category"], df["ml_pred_category"])
    types = type_metrics(df["label_type"], df["ml_pred_type"])
    cal = calibration_metrics(
        df["label_binary"], df["ml_pred_binary"],
        df["ml_conf_binary"],
    )
    usage = {
        "eval_scope": "benign_plus_unicode_only",
        "nlp_rows_excluded": n_excluded,
    }
    report = generate_report(df, binary, cat, types, cal, usage=usage,
                             title="ML Classifier Evaluation Report")
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

    report = generate_report(research_df, binary, cat, types, cal, usage,
                             title="Hybrid Router Evaluation Report")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  Hybrid report saved → {output_path}")
    return binary


def generate_llm_report(research_df: pd.DataFrame, output_path: str):
    """Generate LLM-only evaluation report from the research DataFrame."""
    # Only evaluate rows that have LLM predictions (left merge may leave NaN)
    df = research_df.dropna(subset=["llm_pred_binary"])
    if df.empty:
        print(f"  Skipping LLM report — no LLM predictions available (0/{len(research_df)} samples)")
        return None
    judge_decisions = df.get("judge_computed_decision") if "judge_computed_decision" in df.columns else None
    binary = binary_metrics(df["label_binary"], df["llm_pred_binary"], judge_decisions=judge_decisions)
    # Category metrics only if LLM provides category predictions
    if "llm_pred_category" in df.columns:
        cat = category_metrics(df["label_category"], df["llm_pred_category"])
        # LLM does not predict type-level labels; skip type metrics
        types = {"type_accuracy": 0.0, "type_f1_macro": 0.0}
    else:
        cat = {"category_accuracy": 0.0, "category_f1_macro": 0.0}
        types = {"type_accuracy": 0.0, "type_f1_macro": 0.0}
    cal = calibration_metrics(
        df["label_binary"], df["llm_pred_binary"],
        df["llm_conf_binary"],
    )
    report = generate_report(df, binary, cat, types, cal)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  LLM report saved → {output_path} ({len(df)}/{len(research_df)} samples with LLM predictions)")
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
