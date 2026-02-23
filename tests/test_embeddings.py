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


# ---------------------------------------------------------------------------
# T8 — select_pairs_by_benign() diversity after Patch 4
# ---------------------------------------------------------------------------
class TestSelectPairsByBenign:
    """T8: Tests for select_pairs_by_benign() — similarity-ranked, not type-order-biased."""

    @pytest.fixture
    def bank_multi_type(self):
        """Bank with 3 attack types and benign, each with distinct embedding directions."""
        bank = ExemplarBank()
        # "Diacritcs" exemplar points along +x axis
        bank.bank["Diacritcs"] = {
            "texts": ["héllö"],
            "embeddings": np.array([[1.0, 0.0, 0.0]]),
        }
        # "Zero Width" exemplar points along +y axis
        bank.bank["Zero Width"] = {
            "texts": ["he\u200bllo"],
            "embeddings": np.array([[0.0, 1.0, 0.0]]),
        }
        # "Homoglyphs" exemplar points along -x axis (opposite to Diacritcs)
        bank.bank["Homoglyphs"] = {
            "texts": ["hеllo"],
            "embeddings": np.array([[-1.0, 0.0, 0.0]]),
        }
        # Benign exemplars
        bank.bank["benign"] = {
            "texts": ["hello world", "good morning"],
            "embeddings": np.array([
                [0.7, 0.7, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        }
        return bank

    def test_returns_k_pairs(self, bank_multi_type):
        """select_pairs_by_benign returns exactly k pairs when k <= num types."""
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank_multi_type.select_pairs_by_benign(query, k=2)
        assert len(pairs) == 2

    def test_pairs_have_correct_structure(self, bank_multi_type):
        """Each pair is a 3-tuple of (benign_text, attack_text, attack_type)."""
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank_multi_type.select_pairs_by_benign(query, k=2)
        for benign_text, attack_text, attack_type in pairs:
            assert isinstance(benign_text, str)
            assert isinstance(attack_text, str)
            assert isinstance(attack_type, str)

    def test_selects_most_similar_attack_type_first(self, bank_multi_type):
        """Query along +x should pick Diacritcs (most similar) not Homoglyphs (least similar)."""
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank_multi_type.select_pairs_by_benign(query, k=1)
        attack_type = pairs[0][2]
        # Diacritcs at [1,0,0] is identical to query; Homoglyphs at [-1,0,0] is opposite
        assert attack_type == "Diacritcs", (
            f"Expected Diacritcs (most similar to query), got {attack_type!r}"
        )

    def test_avoids_type_iteration_order_bias(self, bank_multi_type):
        """Query along +y should pick Zero Width, not Diacritcs (which is first in ATTACK_TYPES)."""
        query = np.array([0.0, 1.0, 0.0])
        pairs = bank_multi_type.select_pairs_by_benign(query, k=1)
        attack_type = pairs[0][2]
        # Zero Width at [0,1,0] is most similar to this query
        assert attack_type == "Zero Width", (
            f"Expected Zero Width (most similar to +y query), got {attack_type!r}"
        )

    def test_k_larger_than_types_returns_all_types(self, bank_multi_type):
        """k >= number of attack types returns one pair per type (capped at available)."""
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank_multi_type.select_pairs_by_benign(query, k=10)
        # 3 attack types in the bank
        assert len(pairs) == 3
        returned_types = {p[2] for p in pairs}
        assert returned_types == {"Diacritcs", "Zero Width", "Homoglyphs"}

    def test_benign_slot_excluded_from_attack_examples(self, bank_multi_type):
        """The 'benign' bank slot must not appear as an attack example."""
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank_multi_type.select_pairs_by_benign(query, k=3)
        attack_types = {p[2] for p in pairs}
        assert "benign" not in attack_types

    def test_hard_benign_excluded_from_attack_examples(self):
        """The 'hard_benign' bank slot must not appear as an attack example."""
        bank = ExemplarBank()
        bank.bank["Diacritcs"] = {
            "texts": ["héllö"],
            "embeddings": np.array([[1.0, 0.0, 0.0]]),
        }
        bank.bank["benign"] = {
            "texts": ["hello"],
            "embeddings": np.array([[0.5, 0.5, 0.0]]),
        }
        bank.bank["hard_benign"] = {
            "texts": ["What is prompt injection?"],
            "embeddings": np.array([[0.0, 0.0, 1.0]]),
        }
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank.select_pairs_by_benign(query, k=2)
        attack_types = {p[2] for p in pairs}
        assert "hard_benign" not in attack_types
        assert "benign" not in attack_types

    def test_hard_benign_used_as_benign_in_pair(self):
        """When hard_benign slot exists, it is used as the benign half of a pair."""
        bank = ExemplarBank()
        bank.bank["Diacritcs"] = {
            "texts": ["héllö"],
            "embeddings": np.array([[1.0, 0.0, 0.0]]),
        }
        bank.bank["benign"] = {
            "texts": ["regular benign text"],
            "embeddings": np.array([[0.5, 0.5, 0.0]]),
        }
        bank.bank["hard_benign"] = {
            "texts": ["What is prompt injection?"],
            "embeddings": np.array([[0.0, 0.0, 1.0]]),
        }
        query = np.array([1.0, 0.0, 0.0])
        pairs = bank.select_pairs_by_benign(query, k=1)
        assert len(pairs) == 1
        benign_text = pairs[0][0]
        # With k=1, hard_benign gets 1 slot, regular benign gets 0 → benign_text from hard_benign
        assert benign_text == "What is prompt injection?"
