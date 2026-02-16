"""
Research mode for external datasets — comprehensive run capturing all
intermediate ML probabilities and hybrid routing decisions.

Designed to be called once per dataset by DVC foreach stages:
    python -m src.cli.research_external --dataset deepset

LLM execution is controlled primarily via the ``SKIP_LLM`` environment
variable.  By default ``SKIP_LLM`` is ``"1"`` (skip LLM), so a plain
``dvc repro`` will not compute LLM predictions.  To enable LLM predictions,
set ``SKIP_LLM=0`` in the environment.  The CLI flags ``--skip-llm`` and
``--no-skip-llm`` can override the environment variable explicitly.

Produces per-dataset:
  - Wide parquet file (data/processed/research_external/{ds_key}.parquet)
  - Detailed markdown report (reports/research_external/{ds_key}.md)
"""

import argparse
import os

import dotenv
import numpy as np
import pandas as pd

from src.utils import (
    load_config, build_sample_id, MODELS_DIR, SPLITS_DIR,
    RESEARCH_EXTERNAL_DIR, REPORTS_EXTERNAL_DIR,
)
from src.evaluate import binary_metrics, calibration_metrics
from src.eval_external import load_external_dataset
from src.research import compute_hybrid_routing
from src.llm_classifier.llm_classifier import (
        HierarchicalLLMClassifier,
        build_few_shot_examples,
    )

dotenv.load_dotenv()

# Ground truth columns available in external datasets (subset of research.py's)
EXTERNAL_GT_COLS = [
    "modified_sample",
    "label_binary",
    "label_category",
    "label_type",
]


def run_ml_full(ml_model, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run ML predict_full() and return the wide probability DataFrame."""
    result = ml_model.predict_full(df, text_col)
    result.insert(0, "sample_id", df[text_col].reset_index(drop=True).apply(build_sample_id))
    return result


def run_llm_full(
    df: pd.DataFrame,
    cfg: dict,
    text_col: str,
    force_all_stages: bool = False,
) -> pd.DataFrame:
    """Run LLM classifier on all samples and return predictions DataFrame."""
    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
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
    result = pd.DataFrame(rows)
    result.insert(0, "sample_id", df[text_col].reset_index(drop=True).apply(build_sample_id))
    return result


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------

def build_external_research_df(
    df: pd.DataFrame,
    ml_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    llm_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ground truth, ML, LLM, and hybrid results into one wide DataFrame.

    All DataFrames are joined on ``sample_id`` so row order doesn't matter.
    """
    gt = df[EXTERNAL_GT_COLS].reset_index(drop=True)
    gt.insert(0, "sample_id", gt["modified_sample"].apply(build_sample_id))
    result = gt.merge(ml_df, on="sample_id", validate="one_to_one")
    result = result.merge(hybrid_df, on="sample_id", validate="one_to_one")
    if llm_df is not None:
        result = result.merge(llm_df, on="sample_id", how="left", validate="one_to_one")
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text for display in tables."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def generate_research_report(
    ds_key: str,
    hf_name: str,
    research_df: pd.DataFrame,
    binary: dict,
    calibration: dict,
    threshold: float,
) -> str:
    """Generate a detailed Markdown research report for an external dataset."""
    n = len(research_df)
    n_adv = (research_df["label_binary"] == "adversarial").sum()
    n_ben = (research_df["label_binary"] == "benign").sum()

    lines = [
        f"# Research Report — {ds_key}\n",
        f"- **Dataset**: `{hf_name}`",
        f"- **Total samples**: {n}",
        f"- **Adversarial**: {n_adv} ({n_adv / n * 100:.1f}%)",
        f"- **Benign**: {n_ben} ({n_ben / n * 100:.1f}%)",
        f"- **ML confidence threshold**: {threshold}",
        "",
    ]

    # --- Binary metrics ---
    lines.append("## Binary Detection Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in binary.items():
        if k.startswith("support"):
            lines.append(f"| {k} | {v} |")
        else:
            lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    # --- ML confidence distribution ---
    lines.append("## ML Confidence Distribution\n")
    conf = research_df["ml_conf_binary"]
    lines.append(f"- **Overall**: mean={conf.mean():.4f}, median={conf.median():.4f}, "
                 f"std={conf.std():.4f}, min={conf.min():.4f}, max={conf.max():.4f}")

    for label in ["adversarial", "benign"]:
        mask = research_df["label_binary"] == label
        c = conf[mask]
        if len(c) > 0:
            lines.append(f"- **True {label}**: mean={c.mean():.4f}, "
                         f"median={c.median():.4f}, std={c.std():.4f}")
    lines.append("")

    # Confidence by correctness
    correct = research_df["ml_pred_binary"] == research_df["label_binary"]
    c_correct = conf[correct]
    c_wrong = conf[~correct]
    lines.append("### By Prediction Correctness\n")
    if len(c_correct) > 0:
        lines.append(f"- **Correct** ({len(c_correct)} samples): "
                     f"mean={c_correct.mean():.4f}, median={c_correct.median():.4f}")
    if len(c_wrong) > 0:
        lines.append(f"- **Wrong** ({len(c_wrong)} samples): "
                     f"mean={c_wrong.mean():.4f}, median={c_wrong.median():.4f}")
    lines.append("")

    # --- Calibration ---
    lines.append("## Calibration\n")
    lines.append("| Bin | Count | Avg Confidence | Accuracy |")
    lines.append("|-----|-------|----------------|----------|")
    for b in calibration.get("calibration_buckets", []):
        lines.append(
            f"| {b['bin']} | {b['count']} | {b['avg_confidence']:.3f} | {b['accuracy']:.3f} |"
        )
    lines.append("")

    # --- Hybrid routing ---
    lines.append("## Hybrid Routing Analysis\n")
    routing = research_df["hybrid_routed_to"].value_counts()
    for route, count in routing.items():
        pct = count / n * 100
        route_mask = research_df["hybrid_routed_to"] == route
        route_correct = (
            research_df.loc[route_mask, "hybrid_pred_binary"]
            == research_df.loc[route_mask, "label_binary"]
        ).mean()
        lines.append(f"- **{route}**: {count} samples ({pct:.1f}%), "
                     f"accuracy={route_correct:.4f}")
    lines.append("")

    # --- Error analysis ---
    lines.append("## Error Analysis\n")
    errors = research_df[~correct].copy()
    n_errors = len(errors)
    lines.append(f"Total misclassified: {n_errors} / {n} ({n_errors / n * 100:.1f}%)\n")

    if n_errors > 0:
        # False negatives (adversarial predicted as benign)
        fn_mask = (errors["label_binary"] == "adversarial") & (errors["ml_pred_binary"] == "benign")
        fn = errors[fn_mask]
        lines.append(f"### False Negatives (adversarial -> benign): {len(fn)}\n")
        if len(fn) > 0:
            lines.append("| Text | Confidence |")
            lines.append("|------|------------|")
            for _, row in fn.head(15).iterrows():
                text = _truncate(str(row["modified_sample"]))
                text = text.replace("|", "\\|")
                lines.append(f"| {text} | {row['ml_conf_binary']:.4f} |")
            if len(fn) > 15:
                lines.append(f"| ... ({len(fn) - 15} more) | |")
            lines.append("")

        # False positives (benign predicted as adversarial)
        fp_mask = (errors["label_binary"] == "benign") & (errors["ml_pred_binary"] == "adversarial")
        fp = errors[fp_mask]
        lines.append(f"### False Positives (benign -> adversarial): {len(fp)}\n")
        if len(fp) > 0:
            lines.append("| Text | Confidence |")
            lines.append("|------|------------|")
            for _, row in fp.head(15).iterrows():
                text = _truncate(str(row["modified_sample"]))
                text = text.replace("|", "\\|")
                lines.append(f"| {text} | {row['ml_conf_binary']:.4f} |")
            if len(fp) > 15:
                lines.append(f"| ... ({len(fp) - 15} more) | |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predictions DataFrame (compact: text + labels + model output)
# ---------------------------------------------------------------------------

def build_predictions_df(
    research_df: pd.DataFrame,
    has_llm: bool = False,
) -> pd.DataFrame:
    """Extract a compact predictions DataFrame from the wide research DataFrame."""
    cols = [
        "modified_sample",
        "label_binary",
        "ml_pred_binary",
        "ml_conf_binary",
        "ml_pred_category",
        "ml_conf_category",
        "ml_pred_type",
        "ml_conf_type",
        "hybrid_routed_to",
        "hybrid_pred_binary",
        "hybrid_pred_category",
        "hybrid_pred_type",
    ]
    if has_llm:
        cols.extend([
            "llm_pred_binary",
            "llm_conf_binary",
            "llm_pred_category",
            "llm_conf_category",
            "llm_pred_type",
            "llm_conf_type",
        ])
    cols = [c for c in cols if c in research_df.columns]
    return research_df[cols].copy()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_research_single(
    ds_key: str,
    ds_cfg: dict,
    cfg: dict,
    skip_llm: bool = True,
    force_all_stages: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run research-mode pipeline on a single external dataset."""
    from src.ml_classifier.ml_baseline import MLBaseline

    print(f"\n{'=' * 60}")
    print(f"Research (external): {ds_key} ({ds_cfg['name']})")
    print(f"{'=' * 60}")

    df = load_external_dataset(ds_cfg)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    n_adv = (df["label_binary"] == "adversarial").sum()
    n_ben = (df["label_binary"] == "benign").sum()
    print(f"  Loaded {len(df)} samples ({n_adv} adversarial, {n_ben} benign)")

    print("  Running ML predict_full()...")
    ml = MLBaseline(cfg)
    ml.load(str(MODELS_DIR / "ml_baseline.pkl"))
    ml_df = run_ml_full(ml, df, "modified_sample")

    llm_df = None
    if not skip_llm:
        print("  Running LLM classifier...")
        llm_df = run_llm_full(df, cfg, "modified_sample", force_all_stages=force_all_stages)

    threshold = cfg["hybrid"]["ml_confidence_threshold"]
    print(f"  Computing hybrid routing (threshold={threshold})...")
    hybrid_df = compute_hybrid_routing(ml_df, llm_df, threshold)

    research_df = build_external_research_df(df, ml_df, hybrid_df, llm_df)

    RESEARCH_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = RESEARCH_EXTERNAL_DIR / f"research_external_{ds_key}.parquet"
    research_df.to_parquet(parquet_path, index=False)
    print(f"  Research parquet saved -> {parquet_path} (shape: {research_df.shape})")

    binary = binary_metrics(
        research_df["label_binary"],
        research_df["ml_pred_binary"],
    )
    cal = calibration_metrics(
        research_df["label_binary"],
        research_df["ml_pred_binary"],
        research_df["ml_conf_binary"],
    )

    report = generate_research_report(
        ds_key, ds_cfg["name"], research_df, binary, cal, threshold,
    )
    REPORTS_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_EXTERNAL_DIR / f"research_external_{ds_key}.md"
    report_path.write_text(report)
    print(f"  Report saved -> {report_path}")

    print(f"\n  --- {ds_key} Summary ---")
    print(f"  Accuracy:           {binary['accuracy']:.4f}")
    print(f"  Adversarial F1:     {binary['adversarial_f1']:.4f}")
    print(f"  Benign F1:          {binary['benign_f1']:.4f}")
    print(f"  False-negative rate: {binary['false_negative_rate']:.4f}")

    routing = hybrid_df["hybrid_routed_to"].value_counts().to_dict()
    print(f"  Routing: {routing}")

    return research_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_skip_llm(cli_flag: bool | None) -> bool:
    """Resolve the skip_llm tri-state: CLI flag > SKIP_LLM env var > default (True).

    ``cli_flag`` is ``True`` (``--skip-llm``), ``False`` (``--no-skip-llm``),
    or ``None`` (neither flag given → fall back to the ``SKIP_LLM`` env var,
    which itself defaults to ``"1"`` = skip).
    """
    if cli_flag is not None:
        return cli_flag
    return os.environ.get("SKIP_LLM", "1") == "1"


def main():
    parser = argparse.ArgumentParser(
        description="Research mode: comprehensive analysis on external datasets"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="External dataset key from config (e.g. 'deepset', 'jackhhao')",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per dataset")
    parser.add_argument(
        "--skip-llm", action="store_true", dest="skip_llm",
        help="Force skip LLM (overrides SKIP_LLM env var)",
    )
    parser.add_argument(
        "--no-skip-llm", action="store_false", dest="skip_llm",
        help="Force run LLM (overrides SKIP_LLM env var)",
    )
    parser.set_defaults(skip_llm=None)
    parser.add_argument("--force-all-stages", action="store_true",
                        default=False, help="Force LLM to run all 3 stages on every sample")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ext_datasets = cfg.get("external_datasets", {})

    if not ext_datasets:
        print("No external_datasets defined in config.")
        return

    if args.dataset not in ext_datasets:
        print(f"Unknown dataset key: {args.dataset!r}")
        print(f"Available: {list(ext_datasets.keys())}")
        return

    skip_llm = resolve_skip_llm(args.skip_llm)

    run_research_single(
        args.dataset, ext_datasets[args.dataset], cfg,
        skip_llm=skip_llm,
        force_all_stages=args.force_all_stages,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

