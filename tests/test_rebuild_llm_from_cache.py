import json
from pathlib import Path

import pandas as pd
import pytest

from src.llm_cache import get_cache_path
from src.llm_classifier.prompts import build_classifier_messages, build_judge_messages
from src.cli import rebuild_llm_from_cache as rebuild
from src.utils import build_sample_id


def _payload(content: str = '{"label":"benign","confidence":95,"nlp_attack_type":"none","evidence":"","reason":"ok"}'):
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {
                    "content": [
                        {
                            "token": "{",
                            "logprob": -0.1,
                            "top_logprobs": [{"token": "{", "logprob": -0.1}],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _write_cache(cache_dir: Path, provider: str, request_kwargs: dict, payload: dict | None = None) -> Path:
    path = get_cache_path(provider, request_kwargs)
    out = cache_dir / path.name
    out.write_text(json.dumps(payload or _payload()), encoding="utf-8")
    return out


def _cfg() -> dict:
    return {
        "dataset": {"text_col": "modified_sample", "label_col": "attack_name"},
        "llm": {
            "model": "meta/llama-3.1-8b-instruct",
            "model_quality": "meta/llama-3.1-70b-instruct",
            "temperature": 0,
            "max_tokens_classifier": 200,
            "max_tokens_judge": 500,
            "judge_confidence_threshold": 0.8,
            "top_logprobs": 5,
        },
        "labels": {"unicode_attacks": ["Diacritcs"], "nlp_attacks": ["BAE"]},
    }


def _split_row(text: str, prompt_hash: str = "h1", source: str = "mindgard") -> dict:
    return {
        "modified_sample": text,
        "original_sample": text,
        "attack_name": "benign",
        "label_binary": "benign",
        "label_category": "benign",
        "label_type": "benign",
        "prompt_hash": prompt_hash,
        "source": source,
    }


def test_real_cache_smoke_validation_aborts_when_reconstructed_keys_do_not_hit(tmp_path):
    cfg = _cfg()
    smoke_df = pd.DataFrame([_split_row("smoke")])
    fewshot_messages = []
    request = rebuild.build_classifier_request("different", fewshot_messages, cfg)
    _write_cache(tmp_path, "nim", request)

    with pytest.raises(RuntimeError, match="real cache smoke validation failed"):
        rebuild.validate_real_cache_smoke(
            smoke_df=smoke_df,
            fewshot_messages=fewshot_messages,
            cfg=cfg,
            cache_dir=tmp_path,
            smoke_count=1,
        )


def test_real_cache_smoke_validation_accepts_existing_reconstructed_key(tmp_path):
    cfg = _cfg()
    smoke_df = pd.DataFrame([_split_row("smoke")])
    request = rebuild.build_classifier_request("smoke", [], cfg)
    _write_cache(tmp_path, "nim", request)

    result = rebuild.validate_real_cache_smoke(
        smoke_df=smoke_df,
        fewshot_messages=[],
        cfg=cfg,
        cache_dir=tmp_path,
        smoke_count=1,
    )

    assert result["checked"] == 1
    assert result["hits"] == 1


def test_real_local_cache_smoke_diagnostic_hits_existing_files():
    if not rebuild.OLD_TRAIN_PATH.exists() or not all(path.exists() for path in rebuild.OLD_SMOKE_PATHS):
        pytest.skip("old DVC cache split files are not available")
    cache_dir = Path(".cache/llm")
    if not cache_dir.exists() or not any(cache_dir.glob("*.json")):
        pytest.skip("local LLM cache files are not available")
    cfg = rebuild.load_config(None)
    old_train = rebuild.load_old_train(rebuild.OLD_TRAIN_PATH)
    fewshot_messages, _, _ = rebuild.build_few_shot_from_old_train(old_train, cfg)

    result = rebuild.validate_real_cache_smoke(
        smoke_df=rebuild.load_smoke_df(rebuild.OLD_SMOKE_PATHS),
        fewshot_messages=fewshot_messages,
        cfg=cfg,
        cache_dir=cache_dir,
        smoke_count=2,
    )

    assert result == {"checked": 2, "hits": 2}


def test_fewshot_leakage_uses_sample_id_not_prompt_hash():
    fewshot_ids = {build_sample_id("leaking text")}
    current = {
        "val": pd.DataFrame([_split_row("leaking text", prompt_hash="different")]),
        "test": pd.DataFrame([_split_row("safe text", prompt_hash="same")]),
    }

    leakage = rebuild.compute_fewshot_leakage(fewshot_ids, current)

    assert leakage["leaking_sample_ids"] == fewshot_ids
    assert leakage["val_overlap_count"] == 1
    assert leakage["test_overlap_count"] == 0


def test_route_current_splits_marks_train_unmatched_and_ambiguous():
    frames = {
        "train": pd.DataFrame([_split_row("train")]),
        "val": pd.DataFrame([_split_row("ambiguous"), _split_row("val")]),
        "test": pd.DataFrame([_split_row("ambiguous")]),
    }
    candidates = pd.DataFrame(
        {
            "sample_id": [
                build_sample_id("train"),
                build_sample_id("missing"),
                build_sample_id("ambiguous"),
                build_sample_id("val"),
            ]
        }
    )

    routed = rebuild.route_current_splits(candidates, frames)

    assert routed.set_index("sample_id").loc[build_sample_id("train"), "drop_reason"] == "current_split_train_skipped"
    assert routed.set_index("sample_id").loc[build_sample_id("missing"), "drop_reason"] == "unmatched_current_split"
    assert routed.set_index("sample_id").loc[build_sample_id("ambiguous"), "drop_reason"] == "ambiguous_current_split"
    assert routed.set_index("sample_id").loc[build_sample_id("val"), "current_split"] == "val"


def test_validate_cached_payload_rejects_missing_top_logprobs():
    bad = _payload()
    bad["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = []

    result = rebuild.validate_cached_payload(bad, request_type="classifier")

    assert result.drop_reason == "missing_top_logprobs"


def test_judge_recovery_requires_valid_classifier_for_judge(tmp_path):
    cfg = _cfg()
    row = pd.Series({"modified_sample": "needs judge", "sample_id": build_sample_id("needs judge")})
    bad_classifier = rebuild.ValidatedPayload(
        parsed={},
        raw_response_text="{not-json",
        token_logprobs=[{"token": "{", "logprob": -0.1, "top_logprobs": [{"token": "{", "logprob": -0.1}]}],
        drop_reason="invalid_json_response",
    )

    result = rebuild.recover_judge_for_row(row, bad_classifier, cfg, tmp_path)

    assert result.drop_reason == "invalid_classifier_for_judge"


def test_duplicate_payloads_keep_identical_and_drop_conflicts():
    df = pd.DataFrame(
        [
            {"sample_id": "a", "payload_fingerprint": "same", "value": 1},
            {"sample_id": "a", "payload_fingerprint": "same", "value": 1},
            {"sample_id": "b", "payload_fingerprint": "one", "value": 1},
            {"sample_id": "b", "payload_fingerprint": "two", "value": 2},
        ]
    )

    kept, dropped = rebuild.resolve_duplicate_payloads(df)

    assert kept["sample_id"].tolist() == ["a"]
    assert dropped["drop_reason"].tolist() == ["conflicting_cache_payloads", "conflicting_cache_payloads"]


def test_dry_run_writes_nothing_without_audit_path(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, "load_config", lambda _: _cfg())
    monkeypatch.setattr(rebuild, "load_old_train", lambda _: pd.DataFrame([_split_row("old benign"), _split_row("old attack")]))
    monkeypatch.setattr(rebuild, "build_few_shot_from_old_train", lambda _df, _cfg: ([], set(), "hash"))
    monkeypatch.setattr(rebuild, "load_current_splits", lambda _dir: {"val": pd.DataFrame([_split_row("val")])})
    monkeypatch.setattr(rebuild, "load_smoke_df", lambda _path: pd.DataFrame([_split_row("val")]))
    monkeypatch.setattr(rebuild, "validate_real_cache_smoke", lambda **_kwargs: {"checked": 1, "hits": 1})
    monkeypatch.setattr(rebuild, "recover_outputs", lambda **_kwargs: (pd.DataFrame(columns=rebuild.RECOVERED_LLM_COLUMNS), pd.DataFrame()))

    result = rebuild.run(
        config_path=None,
        splits_dir=tmp_path,
        old_train_path=tmp_path / "old.parquet",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        write=False,
        audit_out=None,
    )

    assert result["write"] is False
    assert not (tmp_path / "out").exists()


def test_write_mode_writes_empty_parquets_with_stable_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, "load_config", lambda _: _cfg())
    monkeypatch.setattr(rebuild, "load_old_train", lambda _: pd.DataFrame([_split_row("old benign"), _split_row("old attack")]))
    monkeypatch.setattr(rebuild, "build_few_shot_from_old_train", lambda _df, _cfg: ([], set(), "hash"))
    monkeypatch.setattr(rebuild, "load_current_splits", lambda _dir: {"val": pd.DataFrame([_split_row("val")])})
    monkeypatch.setattr(rebuild, "load_smoke_df", lambda _path: pd.DataFrame([_split_row("val")]))
    monkeypatch.setattr(rebuild, "validate_real_cache_smoke", lambda **_kwargs: {"checked": 1, "hits": 1})
    monkeypatch.setattr(rebuild, "recover_outputs", lambda **_kwargs: (pd.DataFrame(columns=rebuild.RECOVERED_LLM_COLUMNS), pd.DataFrame()))

    rebuild.run(
        config_path=None,
        splits_dir=tmp_path,
        old_train_path=tmp_path / "old.parquet",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        write=True,
        audit_out=None,
    )

    out = pd.read_parquet(tmp_path / "out" / "llm_predictions_val.parquet")
    assert list(out.columns) == rebuild.RECOVERED_LLM_COLUMNS
    assert out.empty
    assert "modified_sample" in out.columns
    assert "classifier_cache_hit" not in out.columns


def test_build_judge_request_uses_classifier_json():
    cfg = _cfg()
    classifier_json = {"label": "benign", "confidence": 50, "nlp_attack_type": "none", "evidence": "", "reason": "ok"}

    request = rebuild.build_judge_request("prompt", classifier_json, cfg)

    assert request["messages"] == build_judge_messages("prompt", classifier_json)
    assert request["model"] == "meta/llama-3.1-70b-instruct"
    assert request["max_tokens"] == 500
