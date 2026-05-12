"""Run the LLM judge on classifier-only Colab local prediction artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

import dotenv
import pandas as pd
from tqdm import tqdm

from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier
from src.utils import PREDICTIONS_DIR, load_config

dotenv.load_dotenv()

INPUT_SUFFIX = "colab_local_classifier"
OUTPUT_SUFFIX = "colab_local_judged"

JUDGE_COLUMNS = [
    "judge_ran",
    "judge_independent_label",
    "judge_category",
    "judge_independent_confidence",
    "judge_independent_evidence",
    "judge_final_label",
    "judge_final_pred_binary",
    "judge_final_category",
    "judge_final_confidence",
    "judge_computed_decision",
    "judge_benign_task_override",
    "judge_override_reason",
    "judge_provider_name",
    "judge_model_name",
    "judge_raw_response_text",
    "judge_parse_success",
    "judge_token_logprobs",
]

REQUIRED_COLUMNS = [
    "sample_id",
    "modified_sample",
    "llm_pred_binary",
    "llm_pred_raw",
    "llm_pred_category",
    "llm_conf_binary",
    "llm_stages_run",
    "llm_provider_name",
    "llm_model_name",
    "llm_raw_response_text",
    "llm_parse_success",
    "clf_label",
    "clf_category",
    "clf_confidence",
    "clf_evidence",
    "clf_nlp_attack_type",
    "clf_provider_name",
    "clf_model_name",
    "clf_raw_response_text",
    "clf_parse_success",
    "clf_token_logprobs",
]


def default_input_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"llm_predictions_{split}_{INPUT_SUFFIX}.parquet"


def default_output_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"llm_predictions_{split}_{OUTPUT_SUFFIX}.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run judge on classifier-only Colab local LLM predictions."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--threshold-only",
        action="store_true",
        help="Only run judge below the configured threshold. Default is research mode: judge every row.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max parallel judge workers. Defaults to config llm.max_concurrency.",
    )
    parser.add_argument(
        "--target-rpm",
        type=float,
        default=None,
        help="Target judge requests per minute. Defaults to config llm.target_rpm.",
    )
    parser.add_argument(
        "--cooldown-on-429",
        type=float,
        default=None,
        help="Global cooldown seconds after a 429. Defaults to config llm.cooldown_on_429.",
    )
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = default_input_path(args.split)
    if args.output is None:
        args.output = default_output_path(args.split)
    return args


def _validate_input(df: pd.DataFrame, input_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")
    if any(column.startswith("judge_") for column in df.columns):
        raise ValueError(f"{input_path} already contains judge columns")


def _classifier_output(row: pd.Series) -> dict[str, Any]:
    return {
        "label": row["clf_label"],
        "confidence": row["clf_confidence"],
        "evidence": row.get("clf_evidence", ""),
        "nlp_attack_type": row.get("clf_nlp_attack_type", "none"),
    }


def _should_run_judge(
    row: pd.Series,
    threshold: float,
    force_all_stages: bool,
) -> bool:
    if force_all_stages:
        return True
    confidence = HierarchicalLLMClassifier._normalize_confidence(
        row.get("clf_confidence"),
        default=0.0,
    )
    return confidence < threshold


def _json_dumps(value: Any) -> str:
    return json.dumps(value)


def _apply_judge_result(row: pd.Series, judge_result: dict[str, Any]) -> dict[str, Any]:
    out = row.to_dict()
    computed_decision = judge_result.get("computed_decision")

    judge_label = judge_result.get("independent_label", "")
    judge_binary = "benign" if judge_label == "benign" else "adversarial"
    judge_category = HierarchicalLLMClassifier._derive_category(
        judge_binary,
        judge_result.get("nlp_attack_type", "none"),
    )
    final_label = judge_result.get("final_label") or judge_label
    if final_label not in ("benign", "adversarial", "uncertain"):
        final_label = "adversarial"
    final_binary = "benign" if final_label == "benign" else "adversarial"
    final_category = HierarchicalLLMClassifier._derive_category(
        final_binary,
        judge_result.get("nlp_attack_type", "none"),
    )

    out.update(
        {
            "judge_ran": True,
            "judge_independent_label": judge_result.get("independent_label"),
            "judge_category": judge_category,
            "judge_independent_confidence": HierarchicalLLMClassifier._normalize_confidence(
                judge_result.get("independent_confidence")
            ),
            "judge_independent_evidence": judge_result.get("independent_evidence", ""),
            "judge_final_label": final_label,
            "judge_final_pred_binary": final_binary,
            "judge_final_category": final_category,
            "judge_final_confidence": HierarchicalLLMClassifier._normalize_confidence(
                judge_result.get("final_confidence")
            ),
            "judge_computed_decision": computed_decision,
            "judge_benign_task_override": judge_result.get("judge_benign_task_override", False),
            "judge_override_reason": judge_result.get("judge_override_reason"),
            "judge_provider_name": judge_result.get("_provider_name"),
            "judge_model_name": judge_result.get("_model_name"),
            "judge_raw_response_text": judge_result.get("_raw_response_text"),
            "judge_parse_success": judge_result.get("_parse_success", False),
            "judge_token_logprobs": _json_dumps(judge_result.get("_token_logprobs")),
        }
    )
    return out


def apply_judge_to_predictions(
    predictions: pd.DataFrame,
    classifier: HierarchicalLLMClassifier,
    *,
    threshold_only: bool = False,
    max_workers: int | None = None,
) -> pd.DataFrame:
    threshold = float(classifier.cfg.get("llm", {}).get("judge_confidence_threshold", 0.8))
    workers = max_workers if max_workers is not None else int(
        classifier.cfg.get("llm", {}).get("max_concurrency", 1)
    )
    workers = max(1, int(workers))

    def process_row(idx: int, row: pd.Series) -> tuple[int, dict[str, Any]]:
        if _should_run_judge(row, threshold, force_all_stages=not threshold_only):
            judge_result = classifier.judge(str(row["modified_sample"]), _classifier_output(row))
            return idx, _apply_judge_result(row, judge_result)

        out = row.to_dict()
        for column in JUDGE_COLUMNS:
            out[column] = None
        return idx, out

    indexed_rows = list(enumerate((row for _, row in predictions.iterrows())))
    rows: list[dict[str, Any] | None] = [None] * len(indexed_rows)
    if workers == 1:
        for idx, row in tqdm(indexed_rows, total=len(indexed_rows), desc="Judging Colab predictions"):
            out_idx, out_row = process_row(idx, row)
            rows[out_idx] = out_row
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_row, idx, row): idx
                for idx, row in indexed_rows
            }
            with tqdm(total=len(futures), desc="Judging Colab predictions") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    out_idx, out_row = future.result()
                    rows[out_idx] = out_row
                    pbar.update(1)

    if any(row is None for row in rows):
        raise RuntimeError("Judge run completed with missing rows.")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.max_concurrency is not None:
        cfg["llm"]["max_concurrency"] = args.max_concurrency
    if args.target_rpm is not None:
        cfg["llm"]["target_rpm"] = args.target_rpm
    elif os.environ.get("LLM_PROVIDER", "nim").lower() == "nim":
        cfg["llm"]["target_rpm"] = 0
    if args.cooldown_on_429 is not None:
        cfg["llm"]["cooldown_on_429"] = args.cooldown_on_429

    classifier = HierarchicalLLMClassifier(cfg)

    predictions = pd.read_parquet(args.input)
    _validate_input(predictions, args.input)

    out = apply_judge_to_predictions(
        predictions,
        classifier,
        threshold_only=args.threshold_only,
        max_workers=args.max_concurrency,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    n_judged = int(out["judge_ran"].fillna(False).sum())
    print(f"Saved {len(out)} rows to {args.output}")
    print(f"Judge ran on {n_judged} rows")
    print(f"Usage: {classifier.usage.to_dict()}")


if __name__ == "__main__":
    main()
