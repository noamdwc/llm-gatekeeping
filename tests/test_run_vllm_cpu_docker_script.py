"""Structural tests for the local vLLM CPU Docker helper."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_vllm_cpu_docker.sh"


def _script_source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_vllm_cpu_docker_script_exists_and_uses_safe_local_defaults():
    source = _script_source()

    assert source.startswith("#!/usr/bin/env bash")
    assert 'MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM2-360M-Instruct}"' in source
    assert 'VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-cpu:latest-arm64}"' in source
    assert "--platform linux/arm64" in source
    assert "-p 8000:8000" in source
    assert '"$HOME/.cache/huggingface:/root/.cache/huggingface"' in source
    assert "VLLM_CPU_KVCACHE_SPACE=2" in source
    assert "--max-model-len 1024" in source
    assert "--max-num-seqs 2" in source
    assert "--tensor-parallel-size 1" in source


def test_vllm_cpu_docker_script_documents_model_override_and_local_api():
    source = _script_source()

    assert "MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct ./run_vllm_cpu_docker.sh" in source
    assert "http://127.0.0.1:8000/v1" in source
    assert "OPENAI_API_KEY=EMPTY" in source
