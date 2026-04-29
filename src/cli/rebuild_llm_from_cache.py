"""Rebuild LLM prediction artifacts from exact local cache hits only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm_cache import get_cache_path
from src.llm_classifier.constants import NLP_TYPES
from src.llm_classifier.llm_classifier import (
    HierarchicalLLMClassifier,
    build_few_shot_examples,
)
from src.llm_classifier.prompts import build_classifier_messages, build_judge_messages
from src.llm_classifier.utils import decide_accept_or_override
from src.utils import PREDICTIONS_DIR, ROOT, SPLITS_DIR, build_sample_id, load_config

OLD_TRAIN_PATH = ROOT / ".dvc/cache/files/md5/a7/2dd2ca5052759088e4f88c52fb6e06"
OLD_SMOKE_PATHS = [
    ROOT / ".dvc/cache/files/md5/26/6f7b14ce982b4fb2439aa3a7b24e6b",
    ROOT / ".dvc/cache/files/md5/47/bcb4c4d8c041c03abf1090a4bf6179",
]

TARGET_SPLITS = ["val", "test", "unseen_val", "unseen_test", "safeguard_test"]

BASE_LLM_COLUMNS = [
    "llm_pred_binary",
    "llm_pred_raw",
    "llm_pred_category",
    "llm_conf_binary",
    "llm_evidence",
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
    "judge_independent_label",
    "judge_category",
    "judge_independent_confidence",
    "judge_independent_evidence",
    "judge_computed_decision",
    "judge_benign_task_override",
    "judge_override_reason",
    "judge_provider_name",
    "judge_model_name",
    "judge_raw_response_text",
    "judge_parse_success",
    "judge_token_logprobs",
]

PROVENANCE_COLUMNS = [
    "sample_id",
    "modified_sample",
    "original_sample",
    "attack_name",
    "label_binary",
    "label_category",
    "label_type",
    "prompt_hash",
    "benign_source",
    "is_synthetic_benign",
    "source",
    "current_split",
    "request_type",
    "cache_key",
    "cache_file",
    "recovered_from_cache",
    "old_fewshot_source",
    "fewshot_sample_ids_hash",
    "model",
    "temperature",
    "max_tokens",
    "logprobs",
    "top_logprobs",
    "raw_response_path",
    "recovery_timestamp",
]

RECOVERED_LLM_COLUMNS = PROVENANCE_COLUMNS + BASE_LLM_COLUMNS

STRING_COLUMNS = {
    "sample_id",
    "prompt_hash",
    "modified_sample",
    "original_sample",
    "attack_name",
    "label_binary",
    "label_category",
    "label_type",
    "benign_source",
    "source",
    "current_split",
    "request_type",
    "cache_key",
    "cache_file",
    "old_fewshot_source",
    "fewshot_sample_ids_hash",
    "model",
    "raw_response_path",
    "recovery_timestamp",
    "llm_pred_binary",
    "llm_pred_raw",
    "llm_pred_category",
    "llm_evidence",
    "llm_provider_name",
    "llm_model_name",
    "llm_raw_response_text",
    "clf_label",
    "clf_category",
    "clf_evidence",
    "clf_nlp_attack_type",
    "clf_provider_name",
    "clf_model_name",
    "clf_raw_response_text",
    "clf_token_logprobs",
    "judge_independent_label",
    "judge_category",
    "judge_independent_evidence",
    "judge_computed_decision",
    "judge_override_reason",
    "judge_provider_name",
    "judge_model_name",
    "judge_raw_response_text",
    "judge_token_logprobs",
}
FLOAT_COLUMNS = {"llm_conf_binary", "clf_confidence", "judge_independent_confidence", "temperature"}
INT_COLUMNS = {"llm_stages_run", "max_tokens", "top_logprobs"}
BOOL_COLUMNS = {"recovered_from_cache", "logprobs", "llm_parse_success", "clf_parse_success", "judge_parse_success"}
BOOL_COLUMNS.add("is_synthetic_benign")


@dataclass(frozen=True)
class ValidatedPayload:
    parsed: dict[str, Any] | None
    raw_response_text: str | None
    token_logprobs: list[dict[str, Any]] | None
    drop_reason: str | None


@dataclass(frozen=True)
class JudgeRecovery:
    payload: ValidatedPayload | None
    request_kwargs: dict[str, Any] | None
    cache_path: Path | None
    cache_key: str | None
    drop_reason: str | None


def cache_key_from_path(path: Path) -> str:
    return path.stem


def load_old_train(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_smoke_df(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise RuntimeError("No old smoke split files found for real cache-key validation")
    return pd.concat(frames, ignore_index=True)


def load_current_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ["train", *TARGET_SPLITS]:
        path = splits_dir / f"{split}.parquet"
        if path.exists():
            frames[split] = pd.read_parquet(path)
    return frames


def _fewshot_messages_from_pairs(pairs: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for benign_text, attack_text, attack_type in pairs:
        messages.append({"role": "user", "content": f"INPUT_PROMPT:\n{benign_text}"})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "label": "benign",
                        "confidence": 95,
                        "nlp_attack_type": "none",
                        "evidence": "",
                        "reason": "No active attempt to override instructions, exfiltrate data, or hijack tools.",
                    }
                ),
            }
        )
        if attack_type in NLP_TYPES:
            evidence = ""
            reason = f"Perturbed tokens characteristic of {attack_type} adversarial attack."
        else:
            evidence = attack_text[:80]
            reason = f"Contains {attack_type} obfuscation; active adversarial prompt detected."
        messages.append({"role": "user", "content": f"INPUT_PROMPT:\n{attack_text}"})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "label": "adversarial",
                        "confidence": 84,
                        "nlp_attack_type": attack_type if attack_type in NLP_TYPES else "none",
                        "evidence": evidence,
                        "reason": reason,
                    }
                ),
            }
        )
    return messages


def build_few_shot_from_old_train(df_train: pd.DataFrame, cfg: dict) -> tuple[list[dict[str, str]], set[str], str]:
    pairs, _ = build_few_shot_examples(df_train, cfg)
    fewshot_texts = {str(text) for pair in pairs for text in (pair[0], pair[1])}
    sample_ids = {build_sample_id(text) for text in fewshot_texts}
    digest = hashlib.sha256("\n".join(sorted(sample_ids)).encode("utf-8")).hexdigest()
    return _fewshot_messages_from_pairs(pairs), sample_ids, digest


def build_classifier_request(text: str, fewshot_messages: list[dict[str, str]], cfg: dict) -> dict[str, Any]:
    llm_cfg = cfg["llm"]
    return {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": build_classifier_messages(text, fewshot_messages),
        "temperature": llm_cfg.get("temperature", 0),
        "max_tokens": llm_cfg.get("max_tokens_classifier", 200),
        "response_format": {"type": "json_object"},
        "logprobs": True,
        "top_logprobs": llm_cfg.get("top_logprobs", 5),
    }


def build_judge_request(text: str, classifier_json: dict[str, Any], cfg: dict) -> dict[str, Any]:
    llm_cfg = cfg["llm"]
    return {
        "model": llm_cfg.get("model_quality", "meta/llama-3.1-70b-instruct"),
        "messages": build_judge_messages(text, classifier_json),
        "temperature": llm_cfg.get("temperature", 0),
        "max_tokens": llm_cfg.get("max_tokens_judge", 500),
        "response_format": {"type": "json_object"},
        "logprobs": True,
        "top_logprobs": llm_cfg.get("top_logprobs", 5),
    }


def request_cache_path(request_kwargs: dict[str, Any], cache_dir: Path) -> Path:
    return cache_dir / get_cache_path("nim", request_kwargs).name


def validate_real_cache_smoke(
    *,
    smoke_df: pd.DataFrame,
    fewshot_messages: list[dict[str, str]],
    cfg: dict,
    cache_dir: Path,
    smoke_count: int = 5,
) -> dict[str, int]:
    text_col = cfg["dataset"]["text_col"]
    checked = smoke_df.head(smoke_count).copy()
    hits = checked[text_col].map(
        lambda text: request_cache_path(build_classifier_request(str(text), fewshot_messages, cfg), cache_dir).exists()
    )
    n_hits = int(hits.sum())
    if len(checked) == 0 or n_hits != len(checked):
        raise RuntimeError(
            f"real cache smoke validation failed: checked={len(checked)} hits={n_hits}. "
            "Stopping instead of guessed recovery."
        )
    return {"checked": len(checked), "hits": n_hits}


def compute_fewshot_leakage(fewshot_sample_ids: set[str], current_splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    text_col = "modified_sample"
    leaking: set[str] = set()
    counts: dict[str, int] = {}
    for split in ["val", "test"]:
        frame = current_splits.get(split, pd.DataFrame())
        ids = set(frame[text_col].map(build_sample_id)) if text_col in frame else set()
        overlap = fewshot_sample_ids & ids
        counts[f"{split}_overlap_count"] = len(overlap)
        leaking |= overlap
    return {"leaking_sample_ids": leaking, **counts}


def route_current_splits(candidates: pd.DataFrame, current_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    membership_parts = []
    for split, frame in current_splits.items():
        if "modified_sample" not in frame:
            continue
        part = frame[["modified_sample"]].copy()
        part["sample_id"] = part["modified_sample"].map(build_sample_id)
        part["current_split"] = split
        membership_parts.append(part[["sample_id", "current_split"]])

    if membership_parts:
        membership = pd.concat(membership_parts, ignore_index=True).drop_duplicates()
        grouped = membership.groupby("sample_id")["current_split"].agg(lambda values: sorted(set(values))).reset_index()
    else:
        grouped = pd.DataFrame(columns=["sample_id", "current_split"])

    routed = candidates.merge(grouped, on="sample_id", how="left")
    routed["split_count"] = routed["current_split"].map(lambda value: len(value) if isinstance(value, list) else 0)
    routed["drop_reason"] = pd.NA
    routed.loc[routed["split_count"] == 0, "drop_reason"] = "unmatched_current_split"
    routed.loc[
        routed["current_split"].map(lambda value: isinstance(value, list) and "train" in value),
        "drop_reason",
    ] = "current_split_train_skipped"
    routed.loc[routed["split_count"] > 1, "drop_reason"] = "ambiguous_current_split"
    routed["current_split"] = routed["current_split"].map(
        lambda value: value[0] if isinstance(value, list) and len(value) == 1 else pd.NA
    )
    return routed


def validate_cached_payload(payload: dict[str, Any], *, request_type: str) -> ValidatedPayload:
    if not isinstance(payload, dict) or "choices" not in payload:
        return ValidatedPayload(None, None, None, "invalid_cached_payload")
    try:
        raw_response_text = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError):
        return ValidatedPayload(None, None, None, "invalid_cached_payload")
    try:
        parsed = json.loads(raw_response_text)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except (TypeError, json.JSONDecodeError):
        return ValidatedPayload(None, raw_response_text, None, "invalid_json_response")
    if not isinstance(parsed, dict):
        return ValidatedPayload(None, raw_response_text, None, "invalid_json_response")

    token_logprobs = HierarchicalLLMClassifier._extract_completion_logprobs(payload)
    if not token_logprobs:
        return ValidatedPayload(parsed, raw_response_text, None, "missing_logprobs")
    if any(not item.get("top_logprobs") for item in token_logprobs):
        return ValidatedPayload(parsed, raw_response_text, token_logprobs, "missing_top_logprobs")
    if request_type not in {"classifier", "judge"}:
        return ValidatedPayload(parsed, raw_response_text, token_logprobs, "unsupported_request_type")
    return ValidatedPayload(parsed, raw_response_text, token_logprobs, None)


def _read_cached_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError):
        return None, "invalid_cached_payload"


def _normalize_confidence(value: Any, default: float = 0.5) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf > 1.0:
        conf /= 100.0
    return max(0.0, min(1.0, conf))


def _derive_category(label: str, nlp_attack_type: str) -> str:
    if label == "benign":
        return "benign"
    if nlp_attack_type and nlp_attack_type != "none":
        return "nlp_attack"
    return "unicode_attack"


def _classifier_result(payload: ValidatedPayload, cfg: dict) -> dict[str, Any]:
    parsed = dict(payload.parsed or {})
    label = parsed.get("label") if parsed.get("label") in {"benign", "adversarial", "uncertain"} else "adversarial"
    nlp_type = parsed.get("nlp_attack_type", "none")
    if nlp_type not in set(NLP_TYPES) | {"none"}:
        nlp_type = "none"
    return {
        "label": label,
        "confidence": _normalize_confidence(parsed.get("confidence", 50)),
        "nlp_attack_type": nlp_type,
        "evidence": parsed.get("evidence", "") if label == "adversarial" else "",
        "_token_logprobs": payload.token_logprobs,
        "_provider_name": "nim",
        "_model_name": "meta/llama-3.1-8b-instruct",
        "_raw_response_text": payload.raw_response_text,
        "_parse_success": True,
    }


def _judge_result(payload: ValidatedPayload) -> dict[str, Any]:
    parsed = dict(payload.parsed or {})
    parsed["_token_logprobs"] = payload.token_logprobs
    parsed["_provider_name"] = "nim"
    parsed["_model_name"] = "meta/llama-3.1-70b-instruct"
    parsed["_raw_response_text"] = payload.raw_response_text
    parsed["_parse_success"] = True
    return parsed


def recover_judge_for_row(
    row: pd.Series,
    classifier_payload: ValidatedPayload,
    cfg: dict,
    cache_dir: Path,
) -> JudgeRecovery:
    if classifier_payload.drop_reason is not None:
        return JudgeRecovery(None, None, None, None, "invalid_classifier_for_judge")
    if not classifier_payload.parsed:
        return JudgeRecovery(None, None, None, None, "missing_classifier_for_judge")

    request_kwargs = build_judge_request(str(row["modified_sample"]), classifier_payload.parsed, cfg)
    cache_path = request_cache_path(request_kwargs, cache_dir)
    cache_key = cache_key_from_path(cache_path)
    if not cache_path.exists():
        return JudgeRecovery(None, request_kwargs, cache_path, cache_key, "judge_cache_miss")
    payload, read_error = _read_cached_payload(cache_path)
    if read_error is not None or payload is None:
        return JudgeRecovery(None, request_kwargs, cache_path, cache_key, read_error)
    validated = validate_cached_payload(payload, request_type="judge")
    if validated.drop_reason is not None:
        return JudgeRecovery(validated, request_kwargs, cache_path, cache_key, validated.drop_reason)
    return JudgeRecovery(validated, request_kwargs, cache_path, cache_key, None)


def _combine_prediction(
    *,
    clf_payload: ValidatedPayload,
    judge_payload: ValidatedPayload | None,
    cfg: dict,
) -> dict[str, Any]:
    clf = _classifier_result(clf_payload, cfg)
    judge = _judge_result(judge_payload) if judge_payload is not None else None
    label = clf["label"]
    confidence = clf["confidence"]
    evidence = clf.get("evidence", "")
    stages_run = 1

    if judge is not None:
        stages_run = 2
        decision = decide_accept_or_override(judge, clf)
        judge["computed_decision"] = decision
        if decision == "override_candidate":
            raw_label = judge.get("independent_label", clf["label"])
            label = raw_label if raw_label in {"benign", "adversarial", "uncertain"} else "adversarial"
            confidence = _normalize_confidence(judge.get("final_confidence", clf["confidence"] * 100))
            evidence = judge.get("independent_evidence", "")

    label_binary = "benign" if label == "benign" else "adversarial"
    clf_category = _derive_category(clf["label"], clf["nlp_attack_type"])
    judge_category = None
    if judge is not None:
        judge_label = judge.get("independent_label", "")
        judge_binary = "benign" if judge_label == "benign" else "adversarial"
        judge_category = _derive_category(judge_binary, judge.get("nlp_attack_type", "none"))
    label_category = judge_category if judge is not None and judge.get("computed_decision") == "override_candidate" else clf_category

    return {
        "llm_pred_binary": label_binary,
        "llm_pred_raw": label,
        "llm_pred_category": label_category,
        "llm_conf_binary": confidence,
        "llm_evidence": evidence,
        "llm_stages_run": stages_run,
        "llm_provider_name": "nim",
        "llm_model_name": "meta/llama-3.1-70b-instruct" if judge is not None else "meta/llama-3.1-8b-instruct",
        "llm_raw_response_text": judge.get("_raw_response_text") if judge is not None else clf.get("_raw_response_text"),
        "llm_parse_success": True,
        "clf_label": clf["label"],
        "clf_category": clf_category,
        "clf_confidence": clf["confidence"],
        "clf_evidence": clf.get("evidence", ""),
        "clf_nlp_attack_type": clf["nlp_attack_type"],
        "clf_provider_name": clf["_provider_name"],
        "clf_model_name": clf["_model_name"],
        "clf_raw_response_text": clf["_raw_response_text"],
        "clf_parse_success": True,
        "clf_token_logprobs": json.dumps(clf.get("_token_logprobs")),
        "judge_independent_label": judge.get("independent_label") if judge is not None else None,
        "judge_category": judge_category,
        "judge_independent_confidence": _normalize_confidence(judge.get("final_confidence")) if judge is not None else None,
        "judge_independent_evidence": judge.get("independent_evidence", "") if judge is not None else None,
        "judge_computed_decision": judge.get("computed_decision") if judge is not None else None,
        "judge_benign_task_override": False if judge is not None else None,
        "judge_override_reason": None,
        "judge_provider_name": judge.get("_provider_name") if judge is not None else None,
        "judge_model_name": judge.get("_model_name") if judge is not None else None,
        "judge_raw_response_text": judge.get("_raw_response_text") if judge is not None else None,
        "judge_parse_success": True if judge is not None else None,
        "judge_token_logprobs": json.dumps(judge.get("_token_logprobs")) if judge is not None else json.dumps(None),
    }


def _candidate_rows(current_splits: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    text_col = cfg["dataset"]["text_col"]
    frames = []
    for split, frame in current_splits.items():
        if split == "train" or text_col not in frame:
            continue
        work = frame.copy()
        work["source_split"] = split
        work["sample_id"] = work[text_col].map(build_sample_id)
        frames.append(work)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def recover_outputs(
    *,
    current_splits: dict[str, pd.DataFrame],
    fewshot_messages: list[dict[str, str]],
    fewshot_sample_ids: set[str],
    fewshot_sample_ids_hash: str,
    cfg: dict,
    cache_dir: Path,
    old_fewshot_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _candidate_rows(current_splits, cfg)
    if candidates.empty:
        return empty_output_frame(), pd.DataFrame(columns=["drop_reason"])

    leakage = compute_fewshot_leakage(fewshot_sample_ids, current_splits)
    routed = route_current_splits(candidates, current_splits)
    if leakage["leaking_sample_ids"]:
        routed["drop_reason"] = routed["drop_reason"].fillna("fewshot_in_val_or_test")

    valid = routed[routed["drop_reason"].isna()].copy()
    drops = routed[routed["drop_reason"].notna()].copy()
    if valid.empty:
        return empty_output_frame(), drops

    rows: list[dict[str, Any]] = []
    extra_drops: list[dict[str, Any]] = []
    timestamp = datetime.now(UTC).isoformat()
    for _, row in valid.iterrows():
        request_kwargs = build_classifier_request(str(row[cfg["dataset"]["text_col"]]), fewshot_messages, cfg)
        cache_path = request_cache_path(request_kwargs, cache_dir)
        cache_key = cache_key_from_path(cache_path)
        if not cache_path.exists():
            extra_drops.append({**row.to_dict(), "drop_reason": "classifier_cache_miss"})
            continue
        payload, read_error = _read_cached_payload(cache_path)
        if read_error is not None or payload is None:
            extra_drops.append({**row.to_dict(), "drop_reason": read_error})
            continue
        clf_payload = validate_cached_payload(payload, request_type="classifier")
        if clf_payload.drop_reason is not None:
            extra_drops.append({**row.to_dict(), "drop_reason": clf_payload.drop_reason})
            continue

        judge_payload = None
        request_type = "classifier"
        if _classifier_result(clf_payload, cfg)["confidence"] < cfg["llm"].get("judge_confidence_threshold", 0.8):
            judge = recover_judge_for_row(row, clf_payload, cfg, cache_dir)
            if judge.drop_reason == "judge_cache_miss":
                extra_drops.append({**row.to_dict(), "drop_reason": "judge_cache_miss"})
                continue
            if judge.drop_reason is not None:
                extra_drops.append({**row.to_dict(), "drop_reason": judge.drop_reason})
                continue
            judge_payload = judge.payload
            request_type = "classifier+judge"

        prediction = _combine_prediction(clf_payload=clf_payload, judge_payload=judge_payload, cfg=cfg)
        fingerprint_payload = {
            "classifier": clf_payload.raw_response_text,
            "judge": judge_payload.raw_response_text if judge_payload is not None else None,
        }
        rows.append(
            {
                **{col: row.get(col) for col in row.index},
                "payload_fingerprint": hashlib.sha256(
                    json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "current_split": row["current_split"],
                "request_type": request_type,
                "cache_key": cache_key,
                "cache_file": str(cache_path),
                "recovered_from_cache": True,
                "old_fewshot_source": old_fewshot_source,
                "fewshot_sample_ids_hash": fewshot_sample_ids_hash,
                "model": request_kwargs["model"],
                "temperature": request_kwargs["temperature"],
                "max_tokens": request_kwargs["max_tokens"],
                "logprobs": request_kwargs["logprobs"],
                "top_logprobs": request_kwargs["top_logprobs"],
                "raw_response_path": str(cache_path),
                "recovery_timestamp": timestamp,
                **prediction,
            }
        )

    recovered, duplicate_drops = resolve_duplicate_payloads(pd.DataFrame(rows))
    all_drops = pd.concat([drops, pd.DataFrame(extra_drops), duplicate_drops], ignore_index=True, sort=False)
    return _normalize_output_frame(recovered), all_drops


def resolve_duplicate_payloads(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "sample_id" not in df or "payload_fingerprint" not in df:
        return df.reset_index(drop=True), pd.DataFrame(columns=list(df.columns) + ["drop_reason"])

    fingerprints = df.groupby("sample_id")["payload_fingerprint"].nunique()
    conflicting_ids = set(fingerprints[fingerprints > 1].index)
    dropped = df[df["sample_id"].isin(conflicting_ids)].copy()
    if not dropped.empty:
        dropped["drop_reason"] = "conflicting_cache_payloads"
    kept = df[~df["sample_id"].isin(conflicting_ids)].drop_duplicates("sample_id", keep="last").reset_index(drop=True)
    return kept, dropped.reset_index(drop=True)


def empty_output_frame() -> pd.DataFrame:
    return _normalize_output_frame(pd.DataFrame(columns=RECOVERED_LLM_COLUMNS))


def _normalize_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RECOVERED_LLM_COLUMNS:
        if col not in out:
            out[col] = pd.NA
    out = out[RECOVERED_LLM_COLUMNS]
    for col in STRING_COLUMNS:
        out[col] = out[col].astype("string")
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Float64")
    for col in INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in BOOL_COLUMNS:
        out[col] = out[col].astype("boolean")
    return out


def _write_outputs(recovered: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in TARGET_SPLITS:
        split_df = recovered[recovered["current_split"] == split].copy()
        if split_df.empty:
            split_df = empty_output_frame()
        split_df.to_parquet(output_dir / f"llm_predictions_{split}.parquet", index=False)


def _audit_summary(recovered: pd.DataFrame, drops: pd.DataFrame, smoke: dict[str, int], write: bool) -> dict[str, Any]:
    drop_counts = drops["drop_reason"].value_counts(dropna=False).to_dict() if "drop_reason" in drops else {}
    emitted = recovered["current_split"].value_counts(dropna=False).to_dict() if not recovered.empty else {}
    classifier_hits = int(len(recovered))
    judge_hits = int((recovered["llm_stages_run"] == 2).sum()) if "llm_stages_run" in recovered else 0
    cache_counts = {
        "classifier_cache_hit": classifier_hits,
        "classifier_cache_miss": int(drop_counts.get("classifier_cache_miss", 0)),
        "judge_cache_hit": judge_hits,
        "judge_cache_miss": int(drop_counts.get("judge_cache_miss", 0)),
    }
    return {
        "write": write,
        "smoke_checked": smoke["checked"],
        "smoke_hits": smoke["hits"],
        **cache_counts,
        "emitted_total": int(len(recovered)),
        "emitted_by_split": {str(k): int(v) for k, v in emitted.items()},
        "drop_counts": {str(k): int(v) for k, v in drop_counts.items()},
    }


def run(
    *,
    config_path: str | None,
    splits_dir: Path,
    old_train_path: Path,
    cache_dir: Path,
    output_dir: Path,
    write: bool,
    audit_out: Path | None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    old_train = load_old_train(old_train_path)
    fewshot_messages, fewshot_ids, fewshot_hash = build_few_shot_from_old_train(old_train, cfg)
    current_splits = load_current_splits(splits_dir)
    smoke = validate_real_cache_smoke(
        smoke_df=load_smoke_df(OLD_SMOKE_PATHS),
        fewshot_messages=fewshot_messages,
        cfg=cfg,
        cache_dir=cache_dir,
    )
    recovered, drops = recover_outputs(
        current_splits=current_splits,
        fewshot_messages=fewshot_messages,
        fewshot_sample_ids=fewshot_ids,
        fewshot_sample_ids_hash=fewshot_hash,
        cfg=cfg,
        cache_dir=cache_dir,
        old_fewshot_source=str(old_train_path),
    )
    summary = _audit_summary(recovered, drops, smoke, write)
    if write:
        _write_outputs(recovered, output_dir)
    if audit_out is not None:
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        audit_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild LLM predictions from exact local cache hits")
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--old-train-path", type=Path, default=OLD_TRAIN_PATH)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "llm")
    parser.add_argument("--output-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--audit-out", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="Write prediction parquets; default is dry-run")
    args = parser.parse_args()
    run(
        config_path=args.config,
        splits_dir=args.splits_dir,
        old_train_path=args.old_train_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        write=args.write,
        audit_out=args.audit_out,
    )


if __name__ == "__main__":
    main()
