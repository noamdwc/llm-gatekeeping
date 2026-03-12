"""Centralized LLM provider configuration.

Reads LLM_PROVIDER env var to switch between NVIDIA NIM (default) and OpenAI.

    LLM_PROVIDER=nim   → NVIDIA NIM (requires NVIDIA_API_KEY)
    LLM_PROVIDER=openai → OpenAI (requires OPENAI_API_KEY)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import dotenv
import openai

dotenv.load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str | None
    api_key: str
    supports_input_type: bool  # NIM embeddings need input_type extra_body


# NIM model name → OpenAI equivalent
_MODEL_MAP: dict[str, str] = {
    "meta/llama-3.1-8b-instruct": "gpt-4o-mini",
    "meta/llama-3.1-70b-instruct": "gpt-4o",
    "nvidia/nv-embedqa-e5-v5": "text-embedding-3-small",
}


def get_provider() -> ProviderConfig:
    """Return provider config based on LLM_PROVIDER env var (default: nim)."""
    name = os.environ.get("LLM_PROVIDER", "nim").lower()
    if name == "nim":
        print('using *nim* provider api')
        return ProviderConfig(
            name="nim",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
            supports_input_type=True,
        )
    if name == "openai":
        print('using *openai* provider api')
        return ProviderConfig(
            name="openai",
            base_url=None,
            api_key=os.environ["OPENAI_API_KEY"],
            supports_input_type=False,
        )
    raise ValueError(f"Unknown LLM_PROVIDER={name!r}. Use 'nim' or 'openai'.")


def make_client(provider: ProviderConfig | None = None) -> openai.OpenAI:
    """Create an OpenAI client configured for the given provider."""
    if provider is None:
        provider = get_provider()
    kwargs: dict = {"api_key": provider.api_key}
    if provider.base_url is not None:
        kwargs["base_url"] = provider.base_url
    return openai.OpenAI(**kwargs)


def resolve_model(name: str, provider: ProviderConfig | None = None) -> str:
    """Translate NIM model names to OpenAI equivalents when using OpenAI provider.

    Unknown names pass through unchanged so users can set custom models in YAML.
    """
    if provider is None:
        provider = get_provider()
    if provider.name == "openai":
        return _MODEL_MAP.get(name, name)
    return name
