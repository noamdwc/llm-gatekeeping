"""
Research mode for external datasets — comprehensive run capturing all
intermediate ML probabilities and hybrid routing decisions.

Designed to be called once per dataset by DVC foreach stages:
    python -m src.cli.research_external --dataset deepset

Modes:
  - ``llm``: generate LLM predictions artifact for one external dataset.
  - ``hybrid``: require external LLM artifact + compute strict hybrid outputs.
  - ``ml``: ML-only research output (explicitly non-hybrid).

Produces per-dataset:
  - LLM predictions parquet (llm mode)
  - Wide parquet file (data/processed/research_external/{ds_key}.parquet)
  - Detailed markdown report (reports/research_external/{ds_key}.md)
"""

import argparse
import os
import time
from pathlib import Path

import dotenv
import pandas as pd

from src.utils import (
    load_config, build_sample_id, MODELS_DIR, SPLITS_DIR,
    RESEARCH_EXTERNAL_DIR, REPORTS_EXTERNAL_DIR, PREDICTIONS_EXTERNAL_DIR,
)
from src.evaluate import binary_metrics, calibration_metrics
from src.eval_external import load_external_dataset
from src.research import (
    compute_hybrid_routing,
    compute_routing_diagnostics,
    render_routing_diagnostics_markdown,
)
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

LLM_OUTPUT_COLUMNS = [
    "sample_id",
    "llm_pred_binary",
    "llm_pred_raw",
    "llm_pred_category",
    "llm_conf_binary",
    "llm_evidence",
    "llm_stages_run",
    "clf_label",
    "clf_category",
    "clf_confidence",
    "clf_evidence",
    "clf_nlp_attack_type",
    "judge_independent_label",
    "judge_category",
    "judge_independent_confidence",
    "judge_independent_evidence",
    "judge_computed_decision",
]


def default_llm_predictions_path(ds_key: str) -> Path:
    return PREDICTIONS_EXTERNAL_DIR / f"llm_predictions_external_{ds_key}.parquet"


def load_llm_predictions_required(
    ds_key: str,
    llm_predictions_path: Path,
) -> pd.DataFrame:
    """Load external LLM predictions artifact and fail loudly if missing/empty."""
    if not llm_predictions_path.exists():
        raise RuntimeError(
            "Hybrid external evaluation requires precomputed LLM predictions but the artifact is missing.\n"
            f"Missing file: {llm_predictions_path}\n"
            f"Generate it via DVC: dvc repro research_external_llm@{ds_key}\n"
            "Or run CLI directly: "
            f"python -m src.cli.research_external --dataset {ds_key} --mode llm "
            f"--llm-predictions-path {llm_predictions_path}"
        )

    llm_df = pd.read_parquet(llm_predictions_path)
    if llm_df.empty:
        raise RuntimeError(
            "Hybrid external evaluation requires non-empty LLM predictions, but the artifact is empty.\n"
            f"Artifact: {llm_predictions_path}\n"
            f"Regenerate via: dvc repro research_external_llm@{ds_key}"
        )

    return llm_df


def run_ml_full(ml_model, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run ML predict_full() and return the wide probability DataFrame."""
    result = ml_model.predict_full(df, text_col)
    result.insert(0, "sample_id", df[text_col].reset_index(drop=True).apply(build_sample_id))
    return result


def _llm_result_to_row(result: dict, sample_id: str) -> dict:
    """Convert raw classifier output to persisted LLM prediction row."""
    return {
        "sample_id": sample_id,
        "llm_pred_binary": result["label_binary"],
        "llm_pred_raw": result["label"],
        "llm_pred_category": result["label_category"],
        "llm_conf_binary": result["confidence"],
        "llm_evidence": result.get("evidence", ""),
        "llm_stages_run": result.get("llm_stages_run"),
        "clf_label": result.get("clf_label"),
        "clf_category": result.get("clf_category"),
        "clf_confidence": result.get("clf_confidence"),
        "clf_evidence": result.get("clf_evidence", ""),
        "clf_nlp_attack_type": result.get("clf_nlp_attack_type", "none"),
        "judge_independent_label": result.get("judge_independent_label"),
        "judge_category": result.get("judge_category"),
        "judge_independent_confidence": result.get("judge_independent_confidence"),
        "judge_independent_evidence": result.get("judge_independent_evidence"),
        "judge_computed_decision": result.get("judge_computed_decision"),
    }


def _normalize_existing_llm_df(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize existing artifacts to current expected schema."""
    normalized = llm_df.copy()
    for col in LLM_OUTPUT_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = None
    return normalized[LLM_OUTPUT_COLUMNS].drop_duplicates("sample_id", keep="last")


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically to avoid corrupted checkpoints on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def run_llm_full(
    df: pd.DataFrame,
    cfg: dict,
    text_col: str,
    force_all_stages: bool = False,
    llm_predictions_path: Path | None = None,
    resume: bool = True,
    checkpoint_every: int = 0,
    max_concurrency: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run LLM classifier and return predictions + runtime metadata."""
    t0 = time.time()
    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    few_shot, _ = build_few_shot_examples(df_train, cfg)
    classifier = HierarchicalLLMClassifier(cfg, few_shot)

    sample_ids = df[text_col].reset_index(drop=True).apply(build_sample_id).tolist()
    existing_df = pd.DataFrame(columns=LLM_OUTPUT_COLUMNS)
    if resume and llm_predictions_path is not None and llm_predictions_path.exists():
        existing_df = _normalize_existing_llm_df(pd.read_parquet(llm_predictions_path))
    sample_id_set = set(sample_ids)
    existing_df = existing_df[existing_df["sample_id"].isin(sample_id_set)].reset_index(drop=True)
    existing_ids = set(existing_df["sample_id"].tolist())

    pending_indices = [idx for idx, sid in enumerate(sample_ids) if sid not in existing_ids]
    pending_ids = [sample_ids[idx] for idx in pending_indices]
    pending_texts = [df[text_col].iloc[idx] for idx in pending_indices]
    pending_rows: dict[str, dict] = {}

    def on_result(idx: int, result: dict) -> None:
        pending_rows[pending_ids[idx]] = _llm_result_to_row(result, pending_ids[idx])
        if (
            checkpoint_every > 0
            and llm_predictions_path is not None
            and len(pending_rows) % checkpoint_every == 0
        ):
            checkpoint_df = pd.DataFrame(list(pending_rows.values()))
            merged = pd.concat([existing_df, checkpoint_df], ignore_index=True)
            merged = _normalize_existing_llm_df(merged)
            _write_parquet_atomic(merged, llm_predictions_path)

    results = classifier.predict_batch(
        pending_texts,
        desc="LLM classifying",
        force_all_stages=force_all_stages,
        max_workers=max_concurrency,
        on_result=on_result if checkpoint_every > 0 and llm_predictions_path is not None else None,
    )

    if not pending_rows and results:
        pending_rows = {
            pending_ids[idx]: _llm_result_to_row(result, pending_ids[idx])
            for idx, result in enumerate(results)
        }

    new_df = pd.DataFrame(list(pending_rows.values())) if pending_rows else pd.DataFrame(columns=LLM_OUTPUT_COLUMNS)
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df = _normalize_existing_llm_df(merged_df)
    result = merged_df.set_index("sample_id").reindex(sample_ids).reset_index()
    result = _normalize_existing_llm_df(result)

    meta = {
        "n_total": len(sample_ids),
        "n_resumed": len(existing_ids),
        "n_new": len(pending_indices),
        "elapsed_s": time.time() - t0,
        "usage": classifier.usage.to_dict(),
    }
    return result, meta


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
    pred_col: str = "ml_pred_binary",
    mode: str = "hybrid",
) -> str:
    """Generate a detailed Markdown research report for an external dataset."""
    n = len(research_df)
    n_adv = (research_df["label_binary"] == "adversarial").sum()
    n_ben = (research_df["label_binary"] == "benign").sum()

    lines = [
        f"# Research Report — {ds_key}\n",
        f"- **Dataset**: `{hf_name}`",
        f"- **Mode**: {mode}",
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
    correct = research_df[pred_col] == research_df["label_binary"]
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

    # --- LLM uncertain rate (when LLM results are present) ---
    if "llm_pred_raw" in research_df.columns:
        uncertain_mask = research_df["llm_pred_raw"] == "uncertain"
        uncertain_rate = uncertain_mask.mean()
        n_uncertain = uncertain_mask.sum()
        lines.append("## LLM Uncertain Rate\n")
        lines.append(f"- **Uncertain predictions**: {n_uncertain} / {n} ({uncertain_rate * 100:.1f}%)")
        if n_uncertain > 0:
            adv_uncertain = (research_df.loc[uncertain_mask, "label_binary"] == "adversarial").sum()
            ben_uncertain = (research_df.loc[uncertain_mask, "label_binary"] == "benign").sum()
            lines.append(f"  - True adversarial marked uncertain: {adv_uncertain}")
            lines.append(f"  - True benign marked uncertain: {ben_uncertain}")
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

    # --- Routing diagnostics ---
    route_diag = compute_routing_diagnostics(research_df)
    lines.append(render_routing_diagnostics_markdown(route_diag))

    # --- Error analysis ---
    lines.append("## Error Analysis\n")
    errors = research_df[~correct].copy()
    n_errors = len(errors)
    lines.append(f"Total misclassified: {n_errors} / {n} ({n_errors / n * 100:.1f}%)\n")

    if n_errors > 0:
        # False negatives (adversarial predicted as benign)
        fn_mask = (errors["label_binary"] == "adversarial") & (errors[pred_col] == "benign")
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
        fp_mask = (errors["label_binary"] == "benign") & (errors[pred_col] == "adversarial")
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
            "llm_pred_raw",
            "llm_pred_category",
            "llm_conf_binary",
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
    mode: str = "hybrid",
    llm_predictions_path: str | None = None,
    skip_llm: bool = True,
    force_all_stages: bool = False,
    llm_max_concurrency: int | None = None,
    llm_checkpoint_every: int | None = None,
    llm_resume: bool | None = None,
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

    llm_path = Path(llm_predictions_path) if llm_predictions_path else default_llm_predictions_path(ds_key)

    if mode == "llm":
        llm_cfg = cfg.get("llm", {})
        effective_resume = llm_cfg.get("resume", True) if llm_resume is None else llm_resume
        effective_checkpoint = (
            int(llm_cfg.get("checkpoint_every", 200))
            if llm_checkpoint_every is None else int(llm_checkpoint_every)
        )
        effective_concurrency = (
            int(llm_cfg.get("max_concurrency", 8))
            if llm_max_concurrency is None else int(llm_max_concurrency)
        )
        print("  Running LLM classifier...")
        print(
            "  LLM settings: "
            f"max_concurrency={effective_concurrency}, "
            f"resume={effective_resume}, checkpoint_every={effective_checkpoint}"
        )
        llm_df, llm_meta = run_llm_full(
            df,
            cfg,
            "modified_sample",
            force_all_stages=force_all_stages,
            llm_predictions_path=llm_path,
            resume=effective_resume,
            checkpoint_every=effective_checkpoint,
            max_concurrency=effective_concurrency,
        )
        _write_parquet_atomic(llm_df, llm_path)
        print(f"  LLM predictions saved -> {llm_path} (shape: {llm_df.shape})")
        elapsed = llm_meta["elapsed_s"]
        n_total = llm_meta["n_total"]
        n_new = llm_meta["n_new"]
        throughput = (n_new / elapsed) if elapsed > 0 else 0.0
        print(
            "  LLM summary: "
            f"total={n_total}, resumed={llm_meta['n_resumed']}, newly_classified={n_new}, "
            f"elapsed={elapsed:.1f}s, throughput={throughput:.2f} samples/s"
        )
        print(f"  LLM usage stats: {llm_meta['usage']}")
        return llm_df

    if mode == "hybrid":
        llm_df = load_llm_predictions_required(ds_key, llm_path)
        print(f"  Loaded LLM predictions: {llm_path} ({len(llm_df)} samples)")
    elif mode == "ml":
        llm_df = None
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected one of: hybrid, ml, llm.")

    if not skip_llm and mode == "ml":
        print("  [note] --no-skip-llm ignored in mode=ml.")

    threshold = cfg["hybrid"]["ml_confidence_threshold"]
    llm_threshold = cfg["hybrid"]["llm_confidence_threshold"]
    unicode_types = cfg.get("labels", {}).get("unicode_attacks", [])
    print(f"  Computing hybrid routing (threshold={threshold})...")
    hybrid_df = compute_hybrid_routing(
        ml_df,
        llm_df,
        threshold,
        llm_conf_threshold=llm_threshold,
        unicode_types=unicode_types,
        require_llm_for_escalations=(mode == "hybrid"),
        llm_required_path=str(llm_path) if mode == "hybrid" else None,
        llm_generation_hint=(
            f"dvc repro research_external_llm@{ds_key} research_external@{ds_key}"
            if mode == "hybrid" else None
        ),
    )

    research_df = build_external_research_df(df, ml_df, hybrid_df, llm_df)

    RESEARCH_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = RESEARCH_EXTERNAL_DIR / f"research_external_{ds_key}.parquet"
    research_df.to_parquet(parquet_path, index=False)
    print(f"  Research parquet saved -> {parquet_path} (shape: {research_df.shape})")

    pred_col = "hybrid_pred_binary" if mode == "hybrid" else "ml_pred_binary"
    binary = binary_metrics(research_df["label_binary"], research_df[pred_col])
    cal = calibration_metrics(
        research_df["label_binary"],
        research_df[pred_col],
        research_df["ml_conf_binary"],
    )

    report = generate_research_report(
        ds_key, ds_cfg["name"], research_df, binary, cal, threshold,
        pred_col=pred_col, mode=mode,
    )
    REPORTS_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_EXTERNAL_DIR / f"research_external_{ds_key}.md"
    report_path.write_text(report)
    print(f"  Report saved -> {report_path}")

    print(f"\n  --- {ds_key} Summary ---")
    print(f"  Accuracy:           {binary['accuracy']:.4f}")
    print(f"  Adversarial F1:     {binary['adversarial_f1']:.4f}")
    print(f"  Benign F1:          {binary['benign_f1']:.4f}")
    print(f"  False-positive rate: {binary['false_positive_rate']:.4f}")
    print(f"  False-negative rate: {binary['false_negative_rate']:.4f}")

    routing = hybrid_df["hybrid_routed_to"].value_counts().to_dict()
    print(f"  Routing: {routing}")
    route_diag = compute_routing_diagnostics(research_df)
    print(
        "  Routing diagnostics: "
        f"benign->llm_rate={route_diag['ml_pred_benign_escalation_rate']:.4f}, "
        f"adv->llm_rate={route_diag['ml_pred_adversarial_escalation_rate']:.4f}"
    )

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
        "--mode",
        choices=["hybrid", "ml", "llm"],
        default="hybrid",
        help="Execution mode: llm (artifact generation), hybrid (strict), or ml",
    )
    parser.add_argument(
        "--llm-predictions-path",
        default=None,
        help="Path to external LLM predictions artifact parquet",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", dest="skip_llm",
        help="Legacy flag (deprecated): force skip LLM (overrides SKIP_LLM env var)",
    )
    parser.add_argument(
        "--no-skip-llm", action="store_false", dest="skip_llm",
        help="Legacy flag (deprecated): force run LLM (overrides SKIP_LLM env var)",
    )
    parser.set_defaults(skip_llm=None)
    parser.add_argument("--force-all-stages", action="store_true",
                        default=False, help="Force LLM to run all 3 stages on every sample")
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=None,
        help="Max parallel workers for LLM classification (defaults to config llm.max_concurrency)",
    )
    parser.add_argument(
        "--llm-checkpoint-every",
        type=int,
        default=None,
        help="Checkpoint every N newly classified samples in mode=llm (defaults to config llm.checkpoint_every)",
    )
    parser.add_argument(
        "--no-llm-resume",
        action="store_false",
        dest="llm_resume",
        default=None,
        help="Disable resume from existing llm_predictions parquet in mode=llm",
    )
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
        mode=args.mode,
        llm_predictions_path=args.llm_predictions_path,
        skip_llm=skip_llm,
        force_all_stages=args.force_all_stages,
        llm_max_concurrency=args.llm_max_concurrency,
        llm_checkpoint_every=args.llm_checkpoint_every,
        llm_resume=args.llm_resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
