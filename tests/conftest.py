"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config():
    """Minimal config dict mirroring configs/default.yaml."""
    return {
        "dataset": {
            "name": "test-dataset",
            "split": "train",
            "text_col": "modified_sample",
            "original_text_col": "original_sample",
            "label_col": "attack_name",
        },
        "labels": {
            "unicode_attacks": [
                "Diacritcs",
                "Zero Width",
                "Homoglyphs",
            ],
            "nlp_attacks": [
                "BAE",
                "TextFooler",
            ],
            "held_out_attacks": [
                "Homoglyphs",
            ],
        },
        "splits": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
            "random_seed": 42,
        },
        "benign": {
            "use_originals": True,
            "target_count": 10,
            "paraphrase_model": "gpt-4o-mini",
            "paraphrases_per_prompt": 2,
        },
        "llm": {
            "model": "gpt-4o-mini",
            "model_quality": "gpt-4o",
            "temperature": 0,
            "max_tokens_binary": 30,
            "max_tokens_category": 30,
            "max_tokens_type": 50,
            "few_shot": {
                "unicode": 2,
                "nlp": 2,
                "embedding_model": "text-embedding-3-small",
                "bank_size_per_type": 5,
                "dynamic_k": 2,
            },
        },
        "ml": {
            "char_ngram_range": [2, 4],
            "max_features": 500,
            "C": 1.0,
        },
        "hybrid": {
            "ml_confidence_threshold": 0.85,
            "llm_confidence_threshold": 0.7,
        },
        "evaluation": {
            "calibration_bins": 10,
        },
    }


@pytest.fixture
def sample_dataframe():
    """
    Small DataFrame with adversarial + benign rows, covering unicode and nlp attacks.
    Has all the columns needed for the pipeline.
    """
    rows = [
        # Unicode attacks
        {"modified_sample": "héllö wörld", "original_sample": "hello world",
         "attack_name": "Diacritcs", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Diacritcs",
         "prompt_hash": "aaa111"},
        {"modified_sample": "he\u200bllo wo\u200brld", "original_sample": "hello world",
         "attack_name": "Zero Width", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Zero Width",
         "prompt_hash": "aaa111"},
        {"modified_sample": "hеllo wоrld", "original_sample": "hello world",
         "attack_name": "Homoglyphs", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Homoglyphs",
         "prompt_hash": "bbb222"},
        # NLP attacks
        {"modified_sample": "greetings earth", "original_sample": "hello world",
         "attack_name": "BAE", "label_binary": "adversarial",
         "label_category": "nlp_attack", "label_type": "nlp_attack",
         "prompt_hash": "ccc333"},
        {"modified_sample": "salutations globe", "original_sample": "hello world",
         "attack_name": "TextFooler", "label_binary": "adversarial",
         "label_category": "nlp_attack", "label_type": "nlp_attack",
         "prompt_hash": "ccc333"},
        # Additional entries for more prompt_hash groups
        {"modified_sample": "t̲e̲s̲t̲ input", "original_sample": "test input",
         "attack_name": "Diacritcs", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Diacritcs",
         "prompt_hash": "ddd444"},
        {"modified_sample": "te\u200bst in\u200bput", "original_sample": "test input",
         "attack_name": "Zero Width", "label_binary": "adversarial",
         "label_category": "unicode_attack", "label_type": "Zero Width",
         "prompt_hash": "ddd444"},
        {"modified_sample": "experiment entry", "original_sample": "test input",
         "attack_name": "BAE", "label_binary": "adversarial",
         "label_category": "nlp_attack", "label_type": "nlp_attack",
         "prompt_hash": "eee555"},
        # Benign
        {"modified_sample": "hello world", "original_sample": "hello world",
         "attack_name": "benign", "label_binary": "benign",
         "label_category": "benign", "label_type": "benign",
         "prompt_hash": "fff666"},
        {"modified_sample": "test input", "original_sample": "test input",
         "attack_name": "benign", "label_binary": "benign",
         "label_category": "benign", "label_type": "benign",
         "prompt_hash": "ggg777"},
        {"modified_sample": "good morning", "original_sample": "good morning",
         "attack_name": "benign", "label_binary": "benign",
         "label_category": "benign", "label_type": "benign",
         "prompt_hash": "hhh888"},
        {"modified_sample": "how are you", "original_sample": "how are you",
         "attack_name": "benign", "label_binary": "benign",
         "label_category": "benign", "label_type": "benign",
         "prompt_hash": "iii999"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_ml_model(sample_config, sample_dataframe):
    """A small fitted MLBaseline instance for testing."""
    from src.ml_baseline import MLBaseline

    model = MLBaseline(sample_config)
    model.fit(sample_dataframe, "modified_sample")
    return model
