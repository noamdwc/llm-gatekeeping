"""Structural tests for local Docker vLLM mode in the Colab classifier notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_local_llm_classifier.ipynb"


def _all_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_notebook_can_use_local_docker_vllm_openai_server():
    source = _all_source()

    assert "VLLM_SERVER_MODE = 'colab'" in source
    assert "VLLM_SERVER_MODE == 'local_docker'" in source
    assert "run_vllm_cpu_docker.sh" in source
    assert "VLLM_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')" in source
    assert "OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'EMPTY')" in source
    assert "MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct'" in source
    assert "SKIP_REPO_SYNC = VLLM_SERVER_MODE == 'local_docker'" in source
    assert "if VLLM_SERVER_MODE == 'colab':" in source
    assert "Using existing local Docker vLLM server" in source
    assert "OpenAI(base_url=VLLM_BASE_URL, api_key=OPENAI_API_KEY)" in source
