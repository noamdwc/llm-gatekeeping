"""
Generate markdown evaluation reports from precomputed research artifacts.

Usage:
    python -m src.cli.eval_new --split test --config configs/default.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import binary_metrics, calibration_metrics
from src.research import generate_ml_report, generate_hybrid_report, generate_llm_report
from src.cli.research_external import generate_research_report
from src.utils import (
    load_config,
    RESEARCH_DIR,
    RESEARCH_EXTERNAL_DIR,
    REPORTS_RESEARCH_DIR,
    REPORTS_EXTERNAL_DIR,
)


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _binary_metrics_or_none(df: pd.DataFrame, pred_col: str) -> dict | None:
    if pred_col not in df.columns:
        return None
    if df.empty:
        return None
    return binary_metrics(df["label_binary"], df[pred_col])


def _external_pred_col(df: pd.DataFrame) -> str:
    """Prefer hybrid predictions for external reports when available."""
    return "hybrid_pred_binary" if "hybrid_pred_binary" in df.columns else "ml_pred_binary"


def _main_metric_rows(df: pd.DataFrame) -> list[dict]:
    rows = []

    # ML summary follows the ML report scope (exclude NLP attacks).
    if "label_category" in df.columns:
        ml_df = df[df["label_category"] != "nlp_attack"].copy()
    else:
        ml_df = df.copy()
    ml_metrics = _binary_metrics_or_none(ml_df, "ml_pred_binary")
    rows.append({
        "model": "ML (unicode scope)",
        "rows": int(len(ml_df)),
        "metrics": ml_metrics,
    })

    hybrid_metrics = _binary_metrics_or_none(df, "hybrid_pred_binary")
    rows.append({
        "model": "Hybrid",
        "rows": int(len(df)),
        "metrics": hybrid_metrics,
    })

    if "llm_pred_binary" in df.columns:
        llm_df = df.dropna(subset=["llm_pred_binary"]).copy()
        llm_metrics = _binary_metrics_or_none(llm_df, "llm_pred_binary")
        rows.append({
            "model": "LLM",
            "rows": int(len(llm_df)),
            "metrics": llm_metrics,
        })
    else:
        rows.append({
            "model": "LLM",
            "rows": 0,
            "metrics": None,
        })

    return rows


def _load_external_research_frames(cfg: dict) -> dict[str, pd.DataFrame]:
    ext_cfg = cfg.get("external_datasets", {})
    if not ext_cfg:
        return {}

    frames = {}
    for ds_key in ext_cfg.keys():
        research_path = RESEARCH_EXTERNAL_DIR / f"research_external_{ds_key}.parquet"
        if not research_path.exists():
            raise FileNotFoundError(
                f"Missing external research parquet: {research_path}\n"
                f"Run upstream stage first: `dvc repro research_external@{ds_key}`."
            )
        frames[ds_key] = pd.read_parquet(research_path)
    return frames


def _render_summary_markdown(
    split: str,
    main_df: pd.DataFrame,
    external_frames: dict[str, pd.DataFrame],
) -> str:
    lines = [
        "# Summary Report",
        "",
        f"- split: `{split}`",
        "",
        "## Main Dataset",
        "",
        "| Model | Rows | Accuracy | Adv F1 | Benign F1 | FPR | FNR |",
        "|-------|------|----------|--------|-----------|-----|-----|",
    ]

    for row in _main_metric_rows(main_df):
        m = row["metrics"]
        lines.append(
            "| "
            f"{row['model']} | {row['rows']} | "
            f"{_fmt_metric(m['accuracy'] if m else None)} | "
            f"{_fmt_metric(m['adversarial_f1'] if m else None)} | "
            f"{_fmt_metric(m['benign_f1'] if m else None)} | "
            f"{_fmt_metric(m['false_positive_rate'] if m else None)} | "
            f"{_fmt_metric(m['false_negative_rate'] if m else None)} |"
        )

    if "hybrid_routed_to" in main_df.columns:
        routing = main_df["hybrid_routed_to"].value_counts().to_dict()
        routing_parts = [f"{k}={v}" for k, v in sorted(routing.items())]
        lines.extend(["", f"- routing: {', '.join(routing_parts)}"])

    if external_frames:
        combined_df = pd.concat(list(external_frames.values()), ignore_index=True)
        combined = _binary_metrics_or_none(combined_df, _external_pred_col(combined_df))
        adv_pct = (
            (combined_df["label_binary"] == "adversarial").mean() if len(combined_df) > 0 else 0.0
        )

        lines.extend([
            "",
            "## External Combined (Unseen) Progress",
            "",
            "| Total Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR | Support Adv | Support Benign |",
            "|------------|-------|----------|--------|-----------|-----|-----|-------------|----------------|",
            "| "
            f"{len(combined_df)} | {adv_pct:.4f} | "
            f"{_fmt_metric(combined['accuracy'] if combined else None)} | "
            f"{_fmt_metric(combined['adversarial_f1'] if combined else None)} | "
            f"{_fmt_metric(combined['benign_f1'] if combined else None)} | "
            f"{_fmt_metric(combined['false_positive_rate'] if combined else None)} | "
            f"{_fmt_metric(combined['false_negative_rate'] if combined else None)} | "
            f"{(combined['support_adversarial'] if combined else 'N/A')} | "
            f"{(combined['support_benign'] if combined else 'N/A')} |",
            "",
            "## External Dataset Breakdown",
            "",
            "| Dataset | Rows | Adv % | Accuracy | Adv F1 | Benign F1 | FPR | FNR |",
            "|---------|------|-------|----------|--------|-----------|-----|-----|",
        ])

        for ds_key in sorted(external_frames.keys()):
            df = external_frames[ds_key]
            m = _binary_metrics_or_none(df, _external_pred_col(df))
            adv_pct = (df["label_binary"] == "adversarial").mean() if len(df) > 0 else 0.0
            lines.append(
                "| "
                f"{ds_key} | {len(df)} | {adv_pct:.4f} | "
                f"{_fmt_metric(m['accuracy'] if m else None)} | "
                f"{_fmt_metric(m['adversarial_f1'] if m else None)} | "
                f"{_fmt_metric(m['benign_f1'] if m else None)} | "
                f"{_fmt_metric(m['false_positive_rate'] if m else None)} | "
                f"{_fmt_metric(m['false_negative_rate'] if m else None)} |"
            )
    else:
        lines.extend([
            "",
            "## External Combined (Unseen) Progress",
            "",
            "No external datasets configured.",
        ])

    return "\n".join(lines) + "\n"


def generate_summary_report(split: str, cfg: dict):
    main_path = RESEARCH_DIR / f"research_{split}.parquet"
    if not main_path.exists():
        raise FileNotFoundError(
            f"Missing research parquet: {main_path}\n"
            "Run upstream stage first: `dvc repro research`."
        )

    main_df = pd.read_parquet(main_path)
    external_frames = _load_external_research_frames(cfg)

    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_RESEARCH_DIR / "summary_report.md"
    summary_md = _render_summary_markdown(split, main_df, external_frames)
    summary_path.write_text(summary_md)
    print(f"  Summary report saved -> {summary_path}")


def _write_llm_placeholder(path: Path, split: str):
    text = (
        "# LLM Evaluation Report\n\n"
        f"No LLM predictions were found for split `{split}` in the research parquet.\n"
        "Run `dvc unfreeze llm_classifier` and then `dvc repro llm_classifier research eval_new` "
        "to generate this report with LLM metrics.\n"
    )
    path.write_text(text)


def generate_main_reports(split: str):
    research_path = RESEARCH_DIR / f"research_{split}.parquet"
    if not research_path.exists():
        raise FileNotFoundError(
            f"Missing research parquet: {research_path}\n"
            "Run upstream stages first: `dvc repro research`."
        )

    df = pd.read_parquet(research_path)
    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    ml_path = REPORTS_RESEARCH_DIR / "eval_report_ml.md"
    hybrid_path = REPORTS_RESEARCH_DIR / "eval_report_hybrid.md"
    llm_path = REPORTS_RESEARCH_DIR / "eval_report_llm.md"

    generate_ml_report(df, str(ml_path))
    generate_hybrid_report(df, str(hybrid_path))

    llm_metrics = generate_llm_report(df, str(llm_path))
    if llm_metrics is None:
        _write_llm_placeholder(llm_path, split)
        print(f"  LLM report placeholder saved -> {llm_path}")


def generate_external_reports(cfg: dict, dataset: str | None = None):
    ext_cfg = cfg.get("external_datasets", {})
    if not ext_cfg:
        print("No external_datasets configured; skipping external reports.")
        return

    if dataset is not None:
        if dataset not in ext_cfg:
            raise ValueError(
                f"Unknown external dataset key: {dataset!r}. "
                f"Available: {list(ext_cfg.keys())}"
            )
        ext_cfg = {dataset: ext_cfg[dataset]}

    threshold = cfg["hybrid"]["ml_confidence_threshold"]
    REPORTS_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    for ds_key, ds_meta in ext_cfg.items():
        research_path = RESEARCH_EXTERNAL_DIR / f"research_external_{ds_key}.parquet"
        if not research_path.exists():
            raise FileNotFoundError(
                f"Missing external research parquet: {research_path}\n"
                f"Run upstream stage first: `dvc repro research_external@{ds_key}`."
            )

        research_df = pd.read_parquet(research_path)
        pred_col = _external_pred_col(research_df)
        binary = binary_metrics(research_df["label_binary"], research_df[pred_col])
        cal = calibration_metrics(
            research_df["label_binary"],
            research_df[pred_col],
            research_df["ml_conf_binary"],
        )
        report = generate_research_report(
            ds_key,
            ds_meta["name"],
            research_df,
            binary,
            cal,
            threshold,
            pred_col=pred_col,
            mode="hybrid" if pred_col == "hybrid_pred_binary" else "ml",
        )
        report_path = REPORTS_EXTERNAL_DIR / f"research_external_{ds_key}.md"
        report_path.write_text(report)
        print(f"  External report saved -> {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate post-router-patch evaluation markdown reports")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default=None, help="External dataset key (for external-only mode)")
    parser.add_argument("--only-main", action="store_true", help="Generate only main split reports")
    parser.add_argument("--only-external", action="store_true", help="Generate only external dataset reports")
    args = parser.parse_args()

    if args.only_main and args.only_external:
        raise ValueError("Choose at most one of --only-main or --only-external.")

    cfg = load_config(args.config)
    if args.only_main:
        generate_main_reports(args.split)
        generate_summary_report(args.split, cfg)
        return

    if args.only_external:
        generate_external_reports(cfg, dataset=args.dataset)
        return

    generate_main_reports(args.split)
    generate_external_reports(cfg, dataset=args.dataset)
    generate_summary_report(args.split, cfg)


if __name__ == "__main__":
    main()
