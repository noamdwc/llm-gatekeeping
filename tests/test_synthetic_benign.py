"""Tests for src.synthetic_benign — generator class and schema validation.

All tests use mocked LLM clients to avoid real API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.synthetic_benign import SyntheticBenignGenerator, _CATEGORY_META, _build_prompt_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """Minimal config dict for generator tests."""
    return {
        "llm": {"model": "meta/llama-3.1-8b-instruct"},
        "benign": {
            "synthetic": {
                "generation_model": "meta/llama-3.1-8b-instruct",
                "batch_size": 5,
                "quotas": {"A": 3, "B": 3, "C": 3, "D": 3, "E": 3, "F": 3},
            }
        },
    }


def _make_generator(cfg, mock_prompts: list[str] | None = None):
    """Create a SyntheticBenignGenerator with a mocked OpenAI client."""
    mock_client = MagicMock()

    if mock_prompts is not None:
        # Mock the LLM to return specified prompts as JSON
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"prompts": mock_prompts})
        mock_client.chat.completions.create.return_value = mock_response

    with patch("src.synthetic_benign.openai.OpenAI", return_value=mock_client):
        gen = SyntheticBenignGenerator(cfg, client=mock_client)

    return gen


# ---------------------------------------------------------------------------
# T7 — test_synth_benign_schema
# ---------------------------------------------------------------------------
class TestSyntheticBenignSchema:
    """T7: Verify that generated records conform to the expected schema."""

    REQUIRED_FIELDS = {
        "modified_sample",
        "original_sample",
        "attack_name",
        "label_binary",
        "label_category",
        "label_type",
        "prompt_hash",
        "synth_category",
        "synth_source",
        "synth_template_id",
        "synth_model",
        "synth_validated",
        "synth_val_score",
        "synth_val_method",
    }

    def test_to_records_schema(self, cfg):
        """Generated records have all required fields."""
        gen = _make_generator(cfg)
        texts = [
            "What is machine learning?",
            "How do I write a Python loop?",
            "Translate hello to French.",
        ]
        records = gen.to_records(texts, category="A")
        assert len(records) == 3
        for r in records:
            assert self.REQUIRED_FIELDS.issubset(set(r.keys())), (
                f"Missing keys: {self.REQUIRED_FIELDS - set(r.keys())}"
            )

    def test_label_fields_are_benign(self, cfg):
        """All label fields must be 'benign'."""
        gen = _make_generator(cfg)
        records = gen.to_records(["Hello world, how are you today?"], category="C")
        r = records[0]
        assert r["label_binary"] == "benign"
        assert r["label_category"] == "benign"
        assert r["label_type"] == "benign"
        assert r["attack_name"] == "benign"

    def test_synth_category_matches_input(self, cfg):
        """synth_category must match the category argument."""
        gen = _make_generator(cfg)
        for cat in _CATEGORY_META:
            records = gen.to_records(["Some benign text for testing purposes."], category=cat)
            assert records[0]["synth_category"] == cat

    def test_prompt_hash_is_string(self, cfg):
        """prompt_hash is a non-empty string."""
        gen = _make_generator(cfg)
        records = gen.to_records(["What is the meaning of life?"], category="F")
        assert isinstance(records[0]["prompt_hash"], str)
        assert len(records[0]["prompt_hash"]) > 0

    def test_prompt_hash_deduplication(self, cfg):
        """Identical texts produce the same prompt_hash."""
        gen = _make_generator(cfg)
        text = "How does gravity work?"
        r1 = gen.to_records([text], category="A")[0]
        r2 = gen.to_records([text], category="B")[0]
        assert r1["prompt_hash"] == r2["prompt_hash"]

    def test_synth_validated_is_bool(self, cfg):
        """synth_validated field is a bool."""
        gen = _make_generator(cfg)
        records_unvalidated = gen.to_records(["Some text here for testing."], category="A", validated=False)
        records_validated = gen.to_records(["Some text here for testing."], category="A", validated=True)
        assert records_unvalidated[0]["synth_validated"] is False
        assert records_validated[0]["synth_validated"] is True

    def test_to_dataframe_returns_dataframe(self, cfg):
        """to_dataframe() returns a pd.DataFrame."""
        gen = _make_generator(cfg)
        texts = ["First benign prompt.", "Second benign prompt is here."]
        df = gen.to_dataframe(texts, category="E")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Generation logic tests
# ---------------------------------------------------------------------------
class TestSyntheticBenignGeneration:
    """Tests for generate_category() — mocked LLM calls."""

    def test_generate_category_returns_list(self, cfg):
        """generate_category() returns a list of strings."""
        prompts = [
            "What is machine learning and how does it work?",
            "Can you help me summarize this article about space?",
            "How do I write a for loop in Python?",
        ]
        gen = _make_generator(cfg, mock_prompts=prompts)
        result = gen.generate_category("A", n=3)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_generate_category_respects_limit(self, cfg):
        """generate_category() returns at most n samples."""
        prompts = [
            "What is recursion in programming?",
            "Can you explain neural networks simply?",
            "How do databases store information?",
            "What is object-oriented programming?",
            "Explain how the internet works.",
        ]
        gen = _make_generator(cfg, mock_prompts=prompts)
        result = gen.generate_category("A", n=3)
        assert len(result) <= 3

    def test_generate_category_deduplicates(self, cfg):
        """Duplicate texts in LLM response are removed."""
        prompts = [
            "What is Python?",
            "What is Python?",  # exact duplicate
            "What is machine learning?",
        ]
        gen = _make_generator(cfg, mock_prompts=prompts)
        result = gen.generate_category("A", n=5)
        assert len(result) == len(set(result))

    def test_generate_category_invalid_category(self, cfg):
        """Unknown category raises ValueError."""
        gen = _make_generator(cfg)
        with pytest.raises(ValueError, match="Unknown category"):
            gen.generate_category("Z", n=5)

    def test_generate_category_all_valid_categories(self, cfg):
        """All six categories can be generated without errors."""
        for cat in "ABCDEF":
            prompts = [f"Sample benign prompt for category {cat} testing purposes."]
            gen = _make_generator(cfg, mock_prompts=prompts)
            result = gen.generate_category(cat, n=1)
            assert isinstance(result, list)

    def test_llm_json_failure_returns_empty(self, cfg):
        """If LLM returns invalid JSON, generate_category returns empty list gracefully."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not json at all"
        mock_client.chat.completions.create.return_value = mock_response

        gen = SyntheticBenignGenerator(cfg, client=mock_client)
        result = gen.generate_category("C", n=3)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------
class TestBuildPromptHash:
    """Tests for _build_prompt_hash()."""

    def test_consistent_hash(self):
        """Same text always produces the same hash."""
        h1 = _build_prompt_hash("hello world")
        h2 = _build_prompt_hash("hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        """Hash is case-insensitive."""
        assert _build_prompt_hash("Hello World") == _build_prompt_hash("hello world")

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped before hashing."""
        assert _build_prompt_hash("  hello  ") == _build_prompt_hash("hello")

    def test_different_texts_different_hash(self):
        """Different texts produce different hashes."""
        assert _build_prompt_hash("text one") != _build_prompt_hash("text two")

    def test_returns_12_char_string(self):
        """Hash is exactly 12 characters (MD5 truncated)."""
        h = _build_prompt_hash("some text")
        assert isinstance(h, str)
        assert len(h) == 12
