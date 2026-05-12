"""
Lightweight split inference + report generation.

Two modes:

- ``--mode ml``: run the ML baseline and produce a markdown report. Fast,
  no API tokens.
- ``--mode escalation``: canonical lightweight inference path. Scores an
  already-existing cheap classifier + DeBERTa prediction pair with the
  trained escalating model, runs the judge on rows whose score crosses
  ``hybrid.escalating_model.judge_threshold``, and emits the
  final-verdict report. Assumes ``escalating_model.pkl`` and the
  cheap-classifier/DeBERTa prediction parquets exist (produced by
  ``dvc repro``); does not retrain the escalating model.

Usage:
  python -m src.cli.infer_split --mode ml --split test
  python -m src.cli.infer_split --mode escalation --split test
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_config, SPLITS_DIR, MODELS_DIR, REPORTS_DIR, REPORTS_RESEARCH_DIR
from src.ml_classifier.ml_baseline import MLBaseline
from src.evaluate import (
    binary_metrics,
    category_metrics,
    type_metrics,
    calibration_metrics,
    generate_report,
)
from src.escalating_model import EscalatingModel
from src.cli import score_escalation, judge_colab_local_predictions, final_verdict_report
from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier


def _compute_metrics_bundle(df_eval: pd.DataFrame, preds_eval: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Compute metric bundle for an already aligned eval subset."""
    df_eval = df_eval.reset_index(drop=True)
    preds_eval = preds_eval.reset_index(drop=True)
    conf_col = (
        "confidence_label_binary_cal"
        if "confidence_label_binary_cal" in preds_eval.columns
        else "confidence_label_binary"
    )

    binary = binary_metrics(df_eval["label_binary"], preds_eval["pred_label_binary"])
    cat = category_metrics(df_eval["label_category"], preds_eval["pred_label_category"])
    types = type_metrics(df_eval["label_type"], preds_eval["pred_label_type"])
    cal = calibration_metrics(df_eval["label_binary"], preds_eval["pred_label_binary"], preds_eval[conf_col])
    return binary, cat, types, cal


def compute_scope_breakdown(df: pd.DataFrame, preds: pd.DataFrame) -> dict[str, dict]:
    """Compute full-scope, ML-scope (no NLP), and NLP-only binary summaries."""
    all_mask = pd.Series(True, index=df.index)
    if "label_category" in df.columns:
        ml_scope_mask = df["label_category"] != "nlp_attack"
        nlp_only_mask = df["label_category"] == "nlp_attack"
    else:
        ml_scope_mask = all_mask
        nlp_only_mask = pd.Series(False, index=df.index)

    scopes = {
        "full": all_mask,
        "ml_scope_no_nlp": ml_scope_mask,
        "nlp_only": nlp_only_mask,
    }
    breakdown = {}

    for scope_name, mask in scopes.items():
        df_scope = df.loc[mask]
        preds_scope = preds.loc[mask]
        if len(df_scope) == 0:
            breakdown[scope_name] = {"rows": 0, "binary": None}
            continue

        binary, _, _, _ = _compute_metrics_bundle(df_scope, preds_scope)
        breakdown[scope_name] = {
            "rows": int(len(df_scope)),
            "binary": binary,
        }

    return breakdown


def format_scope_breakdown_markdown(scope_breakdown: dict[str, dict]) -> str:
    """Render scope breakdown as a markdown section to append to report."""
    lines = [
        "## Scope Breakdown",
        "",
        "| Scope | Rows | Accuracy | False-positive rate | False-negative rate |",
        "|-------|------|----------|---------------------|---------------------|",
    ]
    for scope in ["full", "ml_scope_no_nlp", "nlp_only"]:
        entry = scope_breakdown[scope]
        rows = entry["rows"]
        if entry["binary"] is None:
            lines.append(f"| {scope} | {rows} | N/A | N/A | N/A |")
            continue
        acc = entry["binary"]["accuracy"]
        fpr = entry["binary"]["false_positive_rate"]
        fnr = entry["binary"]["false_negative_rate"]
        lines.append(f"| {scope} | {rows} | {acc:.4f} | {fpr:.4f} | {fnr:.4f} |")

    lines.extend([
        "",
        "- `ml_scope_no_nlp` excludes `label_category == nlp_attack` rows.",
        "- `nlp_only` is out-of-scope for the unicode/character-specialist ML model.",
    ])
    return "\n".join(lines)


def infer_ml_predictions(
    split: str,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    cfg = load_config(config_path)
    text_col = cfg["dataset"]["text_col"]

    df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")

    ml = MLBaseline(cfg)
    ml.load(str(MODELS_DIR / "ml_baseline.pkl"))
    preds = ml.predict(df, text_col)

    binary, cat, types, cal = _compute_metrics_bundle(df, preds)

    return df, preds, binary, cat, types, cal


def infer_ml_split(split: str, config_path: str | None = None) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """Backward-compatible helper: return the original 5-value tuple."""
    df, _, binary, cat, types, cal = infer_ml_predictions(split, config_path)
    return df, binary, cat, types, cal


def run_escalation_split(
    split: str,
    config_path: str | None = None,
    *,
    output: Path | None = None,
) -> Path:
    """Score → judge → final-verdict for a single split. Returns the report path."""
    cfg = load_config(config_path)
    escalating_cfg = cfg.get("hybrid", {}).get("escalating_model", {})
    model_path = Path(escalating_cfg.get("model_path", MODELS_DIR / "escalating_model.pkl"))
    threshold = float(escalating_cfg.get("judge_threshold", 0.5))

    classifier_path = score_escalation.default_classifier_path(split)
    deberta_path = score_escalation.default_deberta_path(split)
    score_output = score_escalation.default_output_path(split)
    judged_output = judge_colab_local_predictions.default_output_path(split)

    model = EscalatingModel.load(model_path)
    classifier_df = pd.read_parquet(classifier_path)
    deberta_df = pd.read_parquet(deberta_path)
    scored, _ = score_escalation.score_split(
        split,
        classifier_df=classifier_df,
        deberta_df=deberta_df,
        model=model,
    )
    score_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(score_output, index=False)

    classifier = HierarchicalLLMClassifier(cfg)
    classifier_predictions = pd.read_parquet(
        judge_colab_local_predictions.default_input_path(split)
    )
    judge_colab_local_predictions._validate_input(
        classifier_predictions,
        judge_colab_local_predictions.default_input_path(split),
    )
    escalation_scores = judge_colab_local_predictions._load_escalation_scores(score_output)
    judged = judge_colab_local_predictions.apply_judge_to_predictions(
        classifier_predictions,
        classifier,
        escalation_scores=escalation_scores,
        escalation_threshold=threshold,
    )
    judged_output.parent.mkdir(parents=True, exist_ok=True)
    judged.to_parquet(judged_output, index=False)

    result = final_verdict_report.DatasetResult(
        name=split,
        kind="internal",
        frame=final_verdict_report.apply_final_verdict(judged),
    )
    report = final_verdict_report.render_report(
        [result],
        threshold=threshold,
        calibration_method=str(escalating_cfg.get("calibration_method", "sigmoid")),
        model_path=str(model_path),
    )
    output_path = output or (REPORTS_DIR / f"inference_escalation_{split}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Infer on a split parquet and generate report")
    parser.add_argument("--mode", choices=["ml", "escalation"], default="ml")
    parser.add_argument("--split", default="test", choices=["test", "val", "unseen_val", "unseen_test", "safeguard_test"])
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output report path (markdown)")
    args = parser.parse_args()

    if args.mode == "escalation":
        out = run_escalation_split(
            args.split,
            args.config,
            output=Path(args.output) if args.output else None,
        )
        print(f"Final-verdict report saved -> {out}")
        return

    df, preds, binary, cat, types, cal = infer_ml_predictions(args.split, args.config)
    scope_breakdown = compute_scope_breakdown(df, preds)

    report = generate_report(df, binary, cat, types, cal)
    report = report.replace("# LLM Classifier Evaluation Report", f"# ML Inference Report — {args.split}")
    report = f"{report}\n\n{format_scope_breakdown_markdown(scope_breakdown)}\n"

    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else (REPORTS_RESEARCH_DIR / f"inference_ml_{args.split}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)

    print(f"Loaded {len(df)} samples from {args.split} split")
    print(f"Report saved -> {out}")
    print("Scope breakdown:")
    for scope in ["full", "ml_scope_no_nlp", "nlp_only"]:
        entry = scope_breakdown[scope]
        if entry["binary"] is None:
            print(
                f"  {scope}: rows={entry['rows']} | "
                "accuracy=N/A | false_positive_rate=N/A | false_negative_rate=N/A"
            )
            continue
        b = entry["binary"]
        print(
            f"  {scope}: rows={entry['rows']} | "
            f"accuracy={b['accuracy']:.4f} | "
            f"false_positive_rate={b['false_positive_rate']:.4f} | "
            f"false_negative_rate={b['false_negative_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
