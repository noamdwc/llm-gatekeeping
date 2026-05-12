"""Generate the canonical escalation-model final-verdict report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluate import binary_metrics
from src.utils import PREDICTIONS_DIR, PREDICTIONS_EXTERNAL_DIR, REPORTS_DIR, RESEARCH_DIR, load_config


DEFAULT_INTERNAL_SPLITS = ["test", "unseen_test", "safeguard_test"]
DEFAULT_OUTPUT = REPORTS_DIR / "pipeline_final_verdict_report.md"


@dataclass(frozen=True)
class DatasetResult:
    name: str
    kind: str
    frame: pd.DataFrame


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, got {value!r}"
        )
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty")
    return name, Path(path)


def default_internal_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"llm_predictions_{split}_colab_local_judged.parquet"


def default_external_path(dataset: str) -> Path:
    return (
        PREDICTIONS_EXTERNAL_DIR
        / f"llm_predictions_external_{dataset}_colab_local_judged.parquet"
    )


def default_external_score_path(dataset: str) -> Path:
    return RESEARCH_DIR / f"escalating_model_eval_external_{dataset}.parquet"


def apply_final_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Add final-verdict columns to a judged cheap-classifier prediction frame."""
    required = {"label_binary", "llm_pred_binary", "llm_pred_category", "llm_conf_binary"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required final-verdict columns: {missing}")

    out = df.copy()
    judge_ran_raw = out.get("judge_ran", pd.Series(False, index=out.index))
    judge_ran = judge_ran_raw.where(judge_ran_raw.notna(), False).astype(bool)
    has_judge_binary = out.get(
        "judge_final_pred_binary",
        pd.Series(pd.NA, index=out.index),
    ).notna()
    use_judge = judge_ran & has_judge_binary

    out["final_pred_binary"] = out["llm_pred_binary"]
    out.loc[use_judge, "final_pred_binary"] = out.loc[use_judge, "judge_final_pred_binary"]

    out["final_pred_category"] = out["llm_pred_category"]
    if "judge_final_category" in out.columns:
        has_judge_category = out["judge_final_category"].notna()
        out.loc[use_judge & has_judge_category, "final_pred_category"] = out.loc[
            use_judge & has_judge_category,
            "judge_final_category",
        ]

    out["final_confidence"] = out["llm_conf_binary"]
    if "judge_final_confidence" in out.columns:
        has_judge_conf = out["judge_final_confidence"].notna()
        out.loc[use_judge & has_judge_conf, "final_confidence"] = out.loc[
            use_judge & has_judge_conf,
            "judge_final_confidence",
        ]

    out["final_source"] = "cheap_classifier"
    out.loc[use_judge, "final_source"] = "judge"
    return out


def _metrics_row(result: DatasetResult) -> dict[str, object]:
    frame = result.frame
    metrics = binary_metrics(frame["label_binary"], frame["final_pred_binary"])
    judge_calls = int((frame["final_source"] == "judge").sum())
    return {
        "name": result.name,
        "kind": result.kind,
        "rows": len(frame),
        "judge_calls": judge_calls,
        "judge_rate": judge_calls / len(frame) if len(frame) else 0.0,
        "accuracy": metrics["accuracy"],
        "adv_recall": metrics["adversarial_recall"],
        "benign_recall": metrics["benign_recall"],
        "adv_precision": metrics["adversarial_precision"],
        "n_adv": metrics["support_adversarial"],
        "n_benign": metrics["support_benign"],
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _append_summary(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        lines.extend([f"## {title}", "", "No canonical judged artifacts found.", ""])
        return

    total_rows = int(sum(int(row["rows"]) for row in rows))
    total_judge = int(sum(int(row["judge_calls"]) for row in rows))
    combined = pd.concat([row["_frame"] for row in rows], ignore_index=True)
    metrics = binary_metrics(combined["label_binary"], combined["final_pred_binary"])
    lines.extend(
        [
            f"## {title}",
            "",
            f"- Rows: **{total_rows}** "
            f"({metrics['support_adversarial']} adv, {metrics['support_benign']} benign)",
            f"- Judge calls: **{total_judge}** ({_fmt_pct(total_judge / total_rows if total_rows else 0.0)})",
            f"- Binary accuracy: **{_fmt_pct(metrics['accuracy'])}**",
            f"- Adversarial recall: **{_fmt_pct(metrics['adversarial_recall'])}**",
            f"- Benign recall: **{_fmt_pct(metrics['benign_recall'])}**",
            f"- Adversarial precision: **{_fmt_pct(metrics['adversarial_precision'])}**",
            "",
            "| name | rows | judge_calls | judge_rate | accuracy | adv_recall | benign_recall | adv_precision | n_adv | n_benign |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['rows']} | {row['judge_calls']} | "
            f"{_fmt_pct(float(row['judge_rate']))} | "
            f"{_fmt_pct(float(row['accuracy']))} | "
            f"{_fmt_pct(float(row['adv_recall']))} | "
            f"{_fmt_pct(float(row['benign_recall']))} | "
            f"{_fmt_pct(float(row['adv_precision']))} | "
            f"{row['n_adv']} | {row['n_benign']} |"
        )
    lines.append("")


def render_report(
    results: list[DatasetResult],
    *,
    threshold: float,
    calibration_method: str,
    model_path: str,
) -> str:
    rows = []
    for result in results:
        row = _metrics_row(result)
        row["_frame"] = result.frame
        rows.append(row)

    internal_rows = [row for row in rows if row["kind"] == "internal"]
    external_rows = [row for row in rows if row["kind"] == "external"]
    all_rows = rows

    lines = [
        "# Pipeline Final-Verdict Report",
        "",
        "This is the canonical evaluation report for the escalation-model path. "
        "For each row, the final verdict is the cheap Colab/local LLM classifier "
        "output unless `hybrid.escalating_model` routes the row to the judge and "
        "a judge final label is available.",
        "",
        "## Historical Correction",
        "",
        "This branch was originally described as risk-model work, but the "
        "implemented gate is the escalation model. The risk model remains the "
        "separate abstain-resolution path; the escalation model owns judge-call "
        "thresholding and final-verdict evaluation.",
        "",
        "## Run Metadata",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Escalation gate threshold | `{threshold:g}` |",
        "| Escalation-score column | `calibrated_escalation_score` when present, else `escalation_score` |",
        f"| Calibration method | `{calibration_method}` |",
        f"| Escalating model artifact | `{model_path}` |",
        "| Selected operating point | `0.5`, frozen for this POC canonical path |",
        "",
        "External escalation is canonical for datasets with judged artifacts under "
        "`data/processed/predictions_external/*_colab_local_judged.parquet`. "
        "External datasets without that artifact are excluded from this report "
        "rather than mixed in as research-only numbers.",
        "",
    ]

    _append_summary(lines, "Overall", all_rows)
    _append_summary(lines, "Internal Splits", internal_rows)
    _append_summary(lines, "External Datasets", external_rows)

    if all_rows:
        total_rows = sum(int(row["rows"]) for row in all_rows)
        total_judge = sum(int(row["judge_calls"]) for row in all_rows)
        lines.extend(
            [
                "## Judge Workload Summary",
                "",
                f"- Total rows scored: {total_rows}",
                f"- Rows escalated to judge: {total_judge} ({_fmt_pct(total_judge / total_rows if total_rows else 0.0)})",
                f"- Reduction vs. judge-everything: **{_fmt_pct(1.0 - (total_judge / total_rows if total_rows else 0.0))}** fewer judge calls",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _attach_labels_from_scores(df: pd.DataFrame, score_path: Path | None) -> pd.DataFrame:
    if "label_binary" in df.columns or score_path is None:
        return df
    if not score_path.exists():
        raise FileNotFoundError(
            f"{df.shape[0]} judged rows are missing label_binary and score file is missing: {score_path}"
        )
    scores = pd.read_parquet(score_path)
    if "sample_id" not in scores.columns or "label_binary" not in scores.columns:
        raise ValueError(f"{score_path} must contain sample_id and label_binary")
    label_cols = ["sample_id", "label_binary"]
    for optional_col in ("label_category", "attack_name"):
        if optional_col in scores.columns and optional_col not in df.columns:
            label_cols.append(optional_col)
    return df.merge(scores[label_cols], on="sample_id", how="left", validate="many_to_one")


def _load_result(
    name: str,
    kind: str,
    path: Path,
    score_path: Path | None = None,
) -> DatasetResult:
    if not path.exists():
        raise FileNotFoundError(f"Missing judged final-verdict input: {path}")
    df = _attach_labels_from_scores(pd.read_parquet(path), score_path)
    return DatasetResult(name=name, kind=kind, frame=apply_final_verdict(df))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final-verdict escalation report.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--internal",
        action="append",
        type=_parse_named_path,
        help="Internal judged artifact as NAME=PATH. Defaults to test/unseen_test/safeguard_test.",
    )
    parser.add_argument(
        "--external",
        action="append",
        type=_parse_named_path,
        help="External judged artifact as NAME=PATH. Defaults to all configured external datasets; missing judged artifacts are errors.",
    )
    return parser


def _default_external_inputs(cfg: dict) -> list[tuple[str, Path]]:
    return [
        (f"external_{key}", default_external_path(key))
        for key in cfg.get("external_datasets", {})
    ]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    escalating_cfg = cfg.get("hybrid", {}).get("escalating_model", {})

    internal_inputs = args.internal or [
        (split, default_internal_path(split)) for split in DEFAULT_INTERNAL_SPLITS
    ]
    external_inputs = args.external
    if external_inputs is None:
        external_inputs = _default_external_inputs(cfg)

    results = [
        _load_result(name, "internal", path)
        for name, path in internal_inputs
    ]
    results.extend(
        _load_result(
            name,
            "external",
            path,
            default_external_score_path(name.removeprefix("external_")),
        )
        for name, path in external_inputs
    )

    report = render_report(
        results,
        threshold=float(escalating_cfg.get("judge_threshold", 0.5)),
        calibration_method=str(escalating_cfg.get("calibration_method", "sigmoid")),
        model_path=str(escalating_cfg.get("model_path", "data/processed/models/escalating_model.pkl")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote final-verdict report to {args.output}")


if __name__ == "__main__":
    main()
