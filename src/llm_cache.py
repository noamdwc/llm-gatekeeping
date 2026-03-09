"""Project-local cache for chat completion responses."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils import ROOT

LLM_CACHE_DIR = ROOT / ".cache" / "llm"

_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class CachedChatResult:
    payload: dict[str, Any]
    cache_hit: bool


def _get_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    return value


def get_cache_path(provider_name: str, request_kwargs: dict[str, Any]) -> Path:
    key_payload = {
        "provider": provider_name,
        "request": _normalize(request_kwargs),
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return LLM_CACHE_DIR / f"{cache_key}.json"


def extract_message_content(payload: dict[str, Any]) -> str | None:
    try:
        return payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError):
        return None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def serialize_chat_completion(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped

    usage = getattr(response, "usage", None)
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    logprobs = getattr(choice, "logprobs", None)
    content_items = getattr(logprobs, "content", None) or []

    return {
        "choices": [
            {
                "message": {
                    "content": _json_scalar(getattr(message, "content", None)),
                },
                "logprobs": {
                    "content": [
                        {
                            "token": _json_scalar(getattr(item, "token", None)),
                            "logprob": _json_scalar(getattr(item, "logprob", None)),
                            "top_logprobs": [
                                {
                                    "token": _json_scalar(getattr(alt, "token", None)),
                                    "logprob": _json_scalar(getattr(alt, "logprob", None)),
                                }
                                for alt in (getattr(item, "top_logprobs", None) or [])
                            ],
                        }
                        for item in content_items
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": _json_scalar(getattr(usage, "prompt_tokens", None)),
            "completion_tokens": _json_scalar(getattr(usage, "completion_tokens", None)),
            "total_tokens": _json_scalar(getattr(usage, "total_tokens", None)),
        } if usage is not None else None,
    }


def get_or_create_chat_completion(
    provider_name: str,
    request_kwargs: dict[str, Any],
    create_fn: Callable[[], Any],
) -> CachedChatResult:
    cache_path = get_cache_path(provider_name, request_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(str(cache_path))

    with lock:
        if cache_path.exists():
            return CachedChatResult(payload=json.loads(cache_path.read_text(encoding="utf-8")), cache_hit=True)

        response = create_fn()
        payload = serialize_chat_completion(response)
        tmp_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        tmp_path.replace(cache_path)
        return CachedChatResult(payload=payload, cache_hit=False)
