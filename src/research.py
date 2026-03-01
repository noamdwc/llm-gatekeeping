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


def _format_llm_required_error(
    message: str,
    llm_required_path: str | None = None,
    llm_generation_hint: str | None = None,
) -> str:
    lines = [message]
    if llm_required_path:
        lines.append(f"Expected LLM predictions artifact: {llm_required_path}")
    if llm_generation_hint:
        lines.append(f"Generate it via: {llm_generation_hint}")
    return "\n".join(lines)


def compute_routing_diagnostics(
    df: pd.DataFrame,
    ml_pred_col: str = "ml_pred_binary",
    route_col: str = "hybrid_routed_to",
) -> dict:
    """Compute additive routing diagnostics for hybrid reports."""
    total = int(len(df))
    routed_ml = int((df[route_col] == "ml").sum()) if total else 0
    routed_llm = int((df[route_col] == "llm").sum()) if total else 0

    ml_is_adv = df[ml_pred_col].map(_is_adversarial_label) if total else pd.Series(dtype=bool)
    ben_mask = ~ml_is_adv if total else pd.Series(dtype=bool)
    adv_mask = ml_is_adv if total else pd.Series(dtype=bool)

    ben_total = int(ben_mask.sum()) if total else 0
    ben_to_ml = int(((df[route_col] == "ml") & ben_mask).sum()) if total else 0
    ben_to_llm = int(((df[route_col] == "llm") & ben_mask).sum()) if total else 0

    adv_total = int(adv_mask.sum()) if total else 0
    adv_to_ml = int(((df[route_col] == "ml") & adv_mask).sum()) if total else 0
    adv_to_llm = int(((df[route_col] == "llm") & adv_mask).sum()) if total else 0

    return {
        "total_samples": total,
        "routed_ml": routed_ml,
        "routed_llm": routed_llm,
        "routed_ml_rate": (routed_ml / total) if total else 0.0,
        "routed_llm_rate": (routed_llm / total) if total else 0.0,
        "ml_pred_benign_total": ben_total,
        "ml_pred_benign_routed_ml": ben_to_ml,
        "ml_pred_benign_routed_llm": ben_to_llm,
        "ml_pred_benign_escalation_rate": (ben_to_llm / ben_total) if ben_total else 0.0,
        "ml_pred_adversarial_total": adv_total,
        "ml_pred_adversarial_routed_ml": adv_to_ml,
        "ml_pred_adversarial_routed_llm": adv_to_llm,
        "ml_pred_adversarial_escalation_rate": (adv_to_llm / adv_total) if adv_total else 0.0,
    }


def render_routing_diagnostics_markdown(diag: dict) -> str:
    """Render routing diagnostics as an additive markdown section."""
    lines = [
        "## Routing Diagnostics",
        "",
        f"- total_samples: {diag['total_samples']}",
        f"- routed_ml: {diag['routed_ml']} ({diag['routed_ml_rate']:.4f})",
        f"- routed_llm: {diag['routed_llm']} ({diag['routed_llm_rate']:.4f})",
        "",
        "| ml_pred_label | routed_ml | routed_llm | escalation_rate |",
        "|---------------|-----------|------------|-----------------|",
        (
            f"| benign | {diag['ml_pred_benign_routed_ml']} | "
            f"{diag['ml_pred_benign_routed_llm']} | "
            f"{diag['ml_pred_benign_escalation_rate']:.4f} |"
        ),
        (
            f"| adversarial | {diag['ml_pred_adversarial_routed_ml']} | "
            f"{diag['ml_pred_adversarial_routed_llm']} | "
            f"{diag['ml_pred_adversarial_escalation_rate']:.4f} |"
        ),
        "",
    ]
    return "\n".join(lines)


def compute_hybrid_routing(
    ml_df: pd.DataFrame,
    llm_df: pd.DataFrame | None,
    threshold: float,
    require_llm_for_escalations: bool = False,
    llm_required_path: str | None = None,
    llm_generation_hint: str | None = None,
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
    print(f"  [research routing] confidence source={conf_col}")
    ml_conf = ml_df[conf_col].values
    ml_pred_binary = ml_df["ml_pred_binary"].values

    ml_adv_mask = np.array([_is_adversarial_label(v) for v in ml_pred_binary], dtype=bool)
    confident = ml_adv_mask & (ml_conf >= threshold)
    n_ml_fastpath = int(confident.sum())
    n_llm_candidates = int((~confident).sum())
    print(
        "  [research routing] "
        f"threshold={threshold} | ml_confident_adv_fastpath={n_ml_fastpath} | "
        f"llm_escalation_candidates={n_llm_candidates}"
    )

    if require_llm_for_escalations and llm_df is None:
        raise RuntimeError(
            _format_llm_required_error(
                "Hybrid routing requires LLM predictions but llm_df is missing.",
                llm_required_path=llm_required_path,
                llm_generation_hint=llm_generation_hint,
            )
        )
    if require_llm_for_escalations and llm_df is not None and llm_df.empty:
        raise RuntimeError(
            _format_llm_required_error(
                "Hybrid routing requires non-empty LLM predictions but llm_df is empty.",
                llm_required_path=llm_required_path,
                llm_generation_hint=llm_generation_hint,
            )
        )

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
        if require_llm_for_escalations and len(no_llm_idx) > 0:
            raise RuntimeError(
                _format_llm_required_error(
                    (
                        "Hybrid routing requires LLM coverage for all escalated samples, "
                        f"but {len(no_llm_idx)} escalated sample(s) are missing from llm_df."
                    ),
                    llm_required_path=llm_required_path,
                    llm_generation_hint=llm_generation_hint,
                )
            )
        result.loc[no_llm_idx, "hybrid_routed_to"] = "ml"
    else:
        if require_llm_for_escalations:
            raise RuntimeError(
                _format_llm_required_error(
                    "Hybrid routing requires LLM predictions but llm_df is missing.",
                    llm_required_path=llm_required_path,
                    llm_generation_hint=llm_generation_hint,
                )
            )
        # No LLM at all — everything routes to ML
        result["hybrid_routed_to"] = "ml"
        print("  [research routing] llm_df missing -> ML fallback for all rows")

    route_diag = compute_routing_diagnostics(result.assign(ml_pred_binary=ml_df["ml_pred_binary"].values))
    print(f"  [research routing] final routing counts={result['hybrid_routed_to'].value_counts().to_dict()}")
    print(
        "  [research routing] diagnostics | "
        f"ml_pred_benign_to_llm_rate={route_diag['ml_pred_benign_escalation_rate']:.4f} | "
        f"ml_pred_adv_to_llm_rate={route_diag['ml_pred_adversarial_escalation_rate']:.4f}"
    )

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
    routing_diag = compute_routing_diagnostics(research_df)
    usage = {
        "routed_ml": routing_diag["routed_ml"],
        "routed_llm": routing_diag["routed_llm"],
        "ml_pred_benign_routed_ml": routing_diag["ml_pred_benign_routed_ml"],
        "ml_pred_benign_routed_llm": routing_diag["ml_pred_benign_routed_llm"],
        "ml_pred_adversarial_routed_ml": routing_diag["ml_pred_adversarial_routed_ml"],
        "ml_pred_adversarial_routed_llm": routing_diag["ml_pred_adversarial_routed_llm"],
    }

    report = generate_report(research_df, binary, cat, types, cal, usage,
                             title="Hybrid Router Evaluation Report")
    report = f"{report}\n{render_routing_diagnostics_markdown(routing_diag)}"
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
