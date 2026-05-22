# ---------------------------------------------------------------------------
# Checkpointing / resume helpers
# ---------------------------------------------------------------------------

import json
import pandas as pd
from pathlib import Path
from src.utils import PREDICTIONS_DIR

def checkpoint_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"llm_checkpoint_{split}.parquet"


def build_research_row(r: dict) -> dict:
    """Convert a single predict() result dict to a research-parquet row."""
    return {
        "llm_pred_binary": r["label_binary"],
        "llm_pred_raw": r["label"],
        "llm_pred_category": r["label_category"],
        "llm_conf_binary": r["confidence"],
        "llm_evidence": r.get("evidence", ""),
        "llm_stages_run": r.get("llm_stages_run"),
        "llm_provider_name": r.get("llm_provider_name"),
        "llm_model_name": r.get("llm_model_name"),
        "llm_raw_response_text": (
            r.get("judge_raw_response_text")
            if r.get("llm_stages_run") == 2
            else r.get("clf_raw_response_text")
        ),
        "llm_parse_success": (
            r.get("judge_parse_success")
            if r.get("llm_stages_run") == 2
            else r.get("clf_parse_success")
        ),
        "clf_label": r.get("clf_label"),
        "clf_category": r.get("clf_category"),
        "clf_confidence": r.get("clf_confidence"),
        "clf_evidence": r.get("clf_evidence", ""),
        "clf_nlp_attack_type": r.get("clf_nlp_attack_type", "none"),
        "clf_provider_name": r.get("clf_provider_name"),
        "clf_model_name": r.get("clf_model_name"),
        "clf_raw_response_text": r.get("clf_raw_response_text"),
        "clf_parse_success": r.get("clf_parse_success"),
        "clf_token_logprobs": json.dumps(r.get("clf_token_logprobs")),
        "judge_independent_label": r.get("judge_independent_label"),
        "judge_category": r.get("judge_category"),
        "judge_independent_confidence": r.get("judge_independent_confidence"),
        "judge_independent_evidence": r.get("judge_independent_evidence"),
        "judge_computed_decision": r.get("judge_computed_decision"),
        "judge_benign_task_override": r.get("judge_benign_task_override"),
        "judge_override_reason": r.get("judge_override_reason"),
        "judge_provider_name": r.get("judge_provider_name"),
        "judge_model_name": r.get("judge_model_name"),
        "judge_raw_response_text": r.get("judge_raw_response_text"),
        "judge_parse_success": r.get("judge_parse_success"),
        "judge_token_logprobs": json.dumps(r.get("judge_token_logprobs")),
    }


def load_checkpoint(split: str) -> set[str]:
    """Return set of sample_ids already completed in a prior checkpoint."""
    cp = checkpoint_path(split)
    if cp.exists():
        df = pd.read_parquet(cp, columns=["sample_id"])
        return set(df["sample_id"].tolist())
    return set()


def append_checkpoint(split: str, rows: list[dict]):
    """Append rows to the checkpoint parquet (create if missing)."""
    if not rows:
        return
    cp = checkpoint_path(split)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if cp.exists():
        df_existing = pd.read_parquet(cp)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(cp, index=False)


def finalize_checkpoint(split: str, out_path: str):
    """Move checkpoint to final output path and clean up."""
    cp = checkpoint_path(split)
    if cp.exists():
        import shutil

        shutil.move(str(cp), out_path)
