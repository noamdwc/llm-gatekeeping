"""Tests for project-local LLM chat caching."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import src.llm_cache as llm_cache_module


def _make_response(content: str = '{"label": "benign", "confidence": 90}'):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.logprobs = None
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


class TestLlmCache:
    def test_same_request_reuses_cached_response(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_cache_module, "LLM_CACHE_DIR", tmp_path)
        create_fn = MagicMock(return_value=_make_response())
        request_kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]}

        first = llm_cache_module.get_or_create_chat_completion("openai", request_kwargs, create_fn)
        second = llm_cache_module.get_or_create_chat_completion("openai", request_kwargs, create_fn)

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert create_fn.call_count == 1

    def test_different_provider_uses_different_cache_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_cache_module, "LLM_CACHE_DIR", tmp_path)
        create_fn = MagicMock(return_value=_make_response())
        request_kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]}

        llm_cache_module.get_or_create_chat_completion("openai", request_kwargs, create_fn)
        llm_cache_module.get_or_create_chat_completion("nim", request_kwargs, create_fn)

        assert create_fn.call_count == 2

    def test_different_request_options_use_different_cache_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_cache_module, "LLM_CACHE_DIR", tmp_path)
        create_fn = MagicMock(return_value=_make_response())
        base = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]}

        llm_cache_module.get_or_create_chat_completion("openai", {**base, "temperature": 0}, create_fn)
        llm_cache_module.get_or_create_chat_completion("openai", {**base, "temperature": 1}, create_fn)

        assert create_fn.call_count == 2

    def test_concurrent_requests_do_not_corrupt_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_cache_module, "LLM_CACHE_DIR", tmp_path)
        request_kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]}
        call_count = 0
        count_lock = threading.Lock()

        def create():
            nonlocal call_count
            with count_lock:
                call_count += 1
            time.sleep(0.05)
            return _make_response()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: llm_cache_module.get_or_create_chat_completion("openai", request_kwargs, create),
                    range(4),
                )
            )

        assert call_count == 1
        assert sum(result.cache_hit for result in results) == 3
        assert len(list(tmp_path.glob("*.json"))) == 1
