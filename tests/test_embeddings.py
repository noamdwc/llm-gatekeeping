"""Tests for src.embeddings — cosine similarity and ExemplarBank."""

import pickle

import numpy as np
import pytest

from src.embeddings import ExemplarBank, cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------
class TestCosineSimilarity:
    """Tests for cosine_similarity()."""

    def test_identical_vectors(self):
        """Identical vectors have cosine similarity ~1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[1.0, 0.0, 0.0]])
        sims = cosine_similarity(a, b)
        assert abs(sims[0] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have cosine similarity ~0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[0.0, 1.0, 0.0]])
        sims = cosine_similarity(a, b)
        assert abs(sims[0]) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors have cosine similarity ~-1.0."""
        a = np.array([1.0, 0.0])
        b = np.array([[-1.0, 0.0]])
        sims = cosine_similarity(a, b)
        assert abs(sims[0] + 1.0) < 1e-6

    def test_batch_computation(self):
        """Compute similarity of one vector against multiple."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.5, 0.5, 0.0],  # partial
        ])
        sims = cosine_similarity(a, b)
        assert sims.shape == (3,)
        assert sims[0] > sims[2] > sims[1]

    def test_zero_vector_handled(self):
        """Zero vector doesn't cause division by zero (epsilon protects)."""
        a = np.zeros(3)
        b = np.array([[1.0, 0.0, 0.0]])
        sims = cosine_similarity(a, b)
        assert np.isfinite(sims[0])


# ---------------------------------------------------------------------------
# ExemplarBank
# ---------------------------------------------------------------------------
class TestExemplarBank:
    """Tests for ExemplarBank select/save/load (no API calls)."""

    @pytest.fixture
    def populated_bank(self):
        """Bank with pre-populated embeddings (no API call needed)."""
        bank = ExemplarBank()
        # Create fake embeddings for two attack types
        bank.bank["Diacritcs"] = {
            "texts": ["héllö", "wörld", "tëst"],
            "embeddings": np.array([
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        }
        bank.bank["Zero Width"] = {
            "texts": ["he\u200bllo", "wo\u200brld"],
            "embeddings": np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ]),
        }
        return bank

    def test_select_returns_top_k(self, populated_bank):
        """select() returns k items ordered by similarity."""
        query = np.array([1.0, 0.0, 0.0])
        results = populated_bank.select(query, "Diacritcs", k=2)
        assert len(results) == 2
        # Most similar to [1,0,0] should be "héllö" then "wörld"
        assert results[0]["text"] == "héllö"
        assert results[1]["text"] == "wörld"

    def test_select_correct_label(self, populated_bank):
        """Returned dicts have the correct label."""
        query = np.array([1.0, 0.0, 0.0])
        results = populated_bank.select(query, "Diacritcs", k=1)
        assert results[0]["label"] == "Diacritcs"

    def test_select_missing_type_returns_empty(self, populated_bank):
        """Missing attack type returns empty list."""
        query = np.array([1.0, 0.0, 0.0])
        results = populated_bank.select(query, "NonExistent", k=2)
        assert results == []

    def test_select_k_larger_than_pool(self, populated_bank):
        """k > available exemplars returns all available."""
        query = np.array([0.0, 1.0, 0.0])
        results = populated_bank.select(query, "Zero Width", k=100)
        assert len(results) == 2  # only 2 in bank

    def test_select_multi_type(self, populated_bank):
        """select_multi_type retrieves from multiple attack types."""
        query = np.array([1.0, 0.0, 0.0])
        results = populated_bank.select_multi_type(
            query, ["Diacritcs", "Zero Width"], k_per_type=1
        )
        assert len(results) == 2
        labels = {r["label"] for r in results}
        assert labels == {"Diacritcs", "Zero Width"}

    def test_save_load_roundtrip(self, populated_bank, tmp_path):
        """save() then load() reproduces the same bank."""
        path = str(tmp_path / "bank.pkl")
        populated_bank.save(path)

        loaded = ExemplarBank.load(path)
        assert set(loaded.bank.keys()) == set(populated_bank.bank.keys())
        for key in populated_bank.bank:
            assert loaded.bank[key]["texts"] == populated_bank.bank[key]["texts"]
            np.testing.assert_array_equal(
                loaded.bank[key]["embeddings"],
                populated_bank.bank[key]["embeddings"],
            )

    def test_repr(self, populated_bank):
        """__repr__ includes type count and exemplar count."""
        r = repr(populated_bank)
        assert "2 types" in r
        assert "5 exemplars" in r

    def test_load_preserves_embedding_model(self, populated_bank, tmp_path):
        """Embedding model name survives save/load cycle."""
        populated_bank.embedding_model = "custom-model"
        path = str(tmp_path / "bank.pkl")
        populated_bank.save(path)

        loaded = ExemplarBank.load(path)
        assert loaded.embedding_model == "custom-model"
