"""
Evaluate the pipeline on external HuggingFace datasets (binary-only).

Loads an external dataset, maps its labels to binary (adversarial/benign),
and runs the ML baseline (or hybrid router) against it.

Usage:
    python -m src.eval_external --dataset deepset --mode ml
    python -m src.eval_external --dataset jackhhao --mode ml
    python -m src.eval_external --dataset all --mode ml
    python -m src.eval_external --dataset deepset --mode hybrid --limit 100
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.evaluate import binary_metrics, calibration_metrics
from src.research import compute_routing_diagnostics, render_routing_diagnostics_markdown
from src.utils import load_config, MODELS_DIR, SPLITS_DIR, REPORTS_EXTERNAL_DIR


# ---------------------------------------------------------------------------
# Dataset loading & label mapping
# ---------------------------------------------------------------------------

def load_external_dataset(ds_cfg: dict) -> pd.DataFrame:
    """
    Download a HuggingFace dataset and map labels to binary format.

    Returns a DataFrame with columns:
        modified_sample  — text (renamed from the dataset's text column)
        label_binary     — "adversarial" or "benign"
        label_category   — same as label_binary (no sub-labels available)
        label_type       — same as label_binary (no sub-labels available)
    """
    ds = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
    df = ds.to_pandas()

    text_col = ds_cfg["text_col"]
    label_col = ds_cfg["label_col"]
    label_map = ds_cfg["label_map"]

    # Convert label_map keys to match the dtype in the dataframe.
    # YAML may parse int keys as int, but the column may be int, str, or bool.
    col_dtype = df[label_col].dtype
    if pd.api.types.is_bool_dtype(col_dtype):
        typed_map = {str(k).lower() == "true": v for k, v in label_map.items()}
    elif pd.api.types.is_integer_dtype(col_dtype):
        typed_map = {int(k): v for k, v in label_map.items()}
    else:
        typed_map = {str(k): v for k, v in label_map.items()}

    df["label_binary"] = df[label_col].map(typed_map)

    # Drop rows that didn't map (unexpected labels)
    unmapped = df["label_binary"].isna()
    if unmapped.any():
        print(f"  Warning: dropping {unmapped.sum()} rows with unmapped labels")
        df = df[~unmapped].reset_index(drop=True)

    # Drop rows with null text
    null_text = df[text_col].isna()
    if null_text.any():
        print(f"  Warning: dropping {null_text.sum()} rows with null text")
        df = df[~null_text].reset_index(drop=True)

    # Rename text column to what the ML pipeline expects
    df = df.rename(columns={text_col: "modified_sample"})

    # Drop duplicate texts
    n_before = len(df)
    df = df.drop_duplicates(subset=["modified_sample"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Warning: dropping {n_dropped} duplicate modified_sample rows")

    # Fill hierarchy columns with binary value (external data has no sub-labels)
    df["label_category"] = df["label_binary"]
    df["label_type"] = df["label_binary"]

    return df


# ---------------------------------------------------------------------------
# Report generation (binary-only)
# ---------------------------------------------------------------------------

def generate_binary_report(
    dataset_name: str,
    hf_name: str,
    mode: str,
    n_samples: int,
    binary: dict,
    calibration: dict,
    router_stats: dict | None = None,
) -> str:
    """Generate a Markdown report for binary-only external evaluation."""
    lines = [
        f"# External Evaluation Report — {dataset_name}\n",
        f"- **Dataset**: `{hf_name}`",
        f"- **Mode**: {mode}",
        f"- **Samples**: {n_samples}",
        "",
        "## Binary Detection (Adversarial vs Benign)\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in binary.items():
        if k.startswith("support"):
            lines.append(f"| {k} | {v} |")
        else:
            lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    # Calibration
    lines.append("## Calibration\n")
    lines.append("| Bin | Count | Avg Confidence | Accuracy |")
    lines.append("|-----|-------|----------------|----------|")
    for b in calibration.get("calibration_buckets", []):
        lines.append(
            f"| {b['bin']} | {b['count']} | {b['avg_confidence']:.3f} | {b['accuracy']:.3f} |"
        )
    lines.append("")

    if router_stats:
        lines.append("## Router Stats\n")
        for k, v in router_stats.items():
            if k == "routing_diagnostics":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
        if "routing_diagnostics" in router_stats:
            lines.append(render_routing_diagnostics_markdown(router_stats["routing_diagnostics"]))
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def evaluate_ml(df: pd.DataFrame, cfg: dict) -> tuple[dict, dict, pd.DataFrame]:
    """Run ML-only evaluation on an external DataFrame. Returns (binary, calibration, ml_preds)."""
    from src.ml_classifier.ml_baseline import MLBaseline

    model_path = MODELS_DIR / "ml_baseline.pkl"

    ml = MLBaseline(cfg)
    ml.load(str(model_path))

    ml_preds = ml.predict(df, "modified_sample")

    pred_binary = ml_preds["pred_label_binary"]
    conf_binary = ml_preds["confidence_label_binary"]

    binary = binary_metrics(df["label_binary"], pred_binary)
    cal = calibration_metrics(df["label_binary"], pred_binary, conf_binary)

    return binary, cal, ml_preds


def evaluate_hybrid(
    df: pd.DataFrame, cfg: dict, limit: int | None = None,
) -> tuple[dict, dict, dict]:
    """Run hybrid router evaluation on an external DataFrame."""
    import dotenv
    dotenv.load_dotenv()

    from src.ml_classifier.ml_baseline import MLBaseline
    from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier, build_few_shot_examples
    from src.hybrid_router import HybridRouter

    # Load ML model
    ml = MLBaseline(cfg)
    ml.load(str(MODELS_DIR / "ml_baseline.pkl"))

    # Load LLM classifier with few-shot examples from training data
    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    few_shot, _ = build_few_shot_examples(df_train, cfg)
    llm = HierarchicalLLMClassifier(cfg, few_shot)

    router = HybridRouter(ml, llm, cfg)

    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    results = router.predict_batch(df, "modified_sample", desc="Hybrid (external)")

    preds = pd.DataFrame(results)
    pred_binary = preds["label_binary"]
    conf_binary = preds.get("confidence_binary", pd.Series([0.5] * len(preds)))

    binary = binary_metrics(df["label_binary"], pred_binary)
    cal = calibration_metrics(df["label_binary"], pred_binary, conf_binary)

    routing_diag = compute_routing_diagnostics(
        preds,
        ml_pred_col="ml_pred_binary",
        route_col="routed_to",
        ml_category_col="ml_pred_category",
        ml_type_col="ml_pred_type",
        unicode_types=cfg.get("labels", {}).get("unicode_attacks", []),
    )
    router_stats = {**router.stats.to_dict(), "routing_diagnostics": routing_diag}
    return binary, cal, router_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_single_dataset(
    ds_key: str, ds_cfg: dict, mode: str, cfg: dict, limit: int | None = None,
):
    """Evaluate a single external dataset and save the report."""
    print(f"\n{'=' * 60}")
    print(f"External eval: {ds_key} ({ds_cfg['name']})")
    print(f"{'=' * 60}")

    df = load_external_dataset(ds_cfg)
    print(f"  Loaded {len(df)} samples "
          f"({(df['label_binary'] == 'adversarial').sum()} adversarial, "
          f"{(df['label_binary'] == 'benign').sum()} benign)")

    router_stats = None
    if mode == "ml":
        binary, cal, _ = evaluate_ml(df, cfg)
    elif mode == "hybrid":
        binary, cal, router_stats = evaluate_hybrid(df, cfg, limit=limit)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'ml' or 'hybrid')")

    # Print summary
    print(f"\n--- {ds_key} Binary Results ({mode}) ---")
    print(f"  Accuracy:           {binary['accuracy']:.4f}")
    print(f"  Adversarial F1:     {binary['adversarial_f1']:.4f}")
    print(f"  Benign F1:          {binary['benign_f1']:.4f}")
    print(f"  False-positive rate: {binary['false_positive_rate']:.4f}")
    print(f"  False-negative rate: {binary['false_negative_rate']:.4f}")

    # Save report
    report = generate_binary_report(
        dataset_name=ds_key,
        hf_name=ds_cfg["name"],
        mode=mode,
        n_samples=len(df),
        binary=binary,
        calibration=cal,
        router_stats=router_stats,
    )
    REPORTS_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_EXTERNAL_DIR / f"eval_external_{ds_key}.md"
    report_path.write_text(report)
    print(f"  Report saved -> {report_path}")

    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline on external HuggingFace datasets (binary-only)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="External dataset key from config (e.g. 'deepset', 'jackhhao', or 'all')",
    )
    parser.add_argument(
        "--mode", default="ml", choices=["ml", "hybrid"],
        help="Evaluation mode: ml (fast, no API) or hybrid (costs API tokens)",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (hybrid mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ext_datasets = cfg.get("external_datasets", {})

    if not ext_datasets:
        print("No external_datasets defined in config.")
        return

    if args.dataset == "all":
        keys = list(ext_datasets.keys())
    else:
        if args.dataset not in ext_datasets:
            print(f"Unknown dataset key: {args.dataset!r}")
            print(f"Available: {list(ext_datasets.keys())}")
            return
        keys = [args.dataset]

    all_results = {}
    for ds_key in keys:
        binary = run_single_dataset(
            ds_key, ext_datasets[ds_key], args.mode, cfg, limit=args.limit,
        )
        all_results[ds_key] = binary

    # Print comparison table if multiple datasets
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Cross-dataset comparison")
        print(f"{'=' * 60}")
        header = f"{'Dataset':<15} {'Accuracy':>10} {'Adv F1':>10} {'FPR':>10} {'FNR':>10}"
        print(header)
        print("-" * len(header))
        for name, b in all_results.items():
            print(
                f"{name:<15} {b['accuracy']:>10.4f} {b['adversarial_f1']:>10.4f} "
                f"{b['false_positive_rate']:>10.4f} {b['false_negative_rate']:>10.4f}"
            )


if __name__ == "__main__":
    main()
