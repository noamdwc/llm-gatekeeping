"""
Embedding-based exemplar retrieval for dynamic few-shot prompting.

Provides:
  - get_embeddings(): Batch OpenAI embedding API calls
  - ExemplarBank: Build, store, and retrieve similar examples per attack type

Usage:
    from src.embeddings import ExemplarBank, get_embeddings
    
    # Build from training data
    bank = ExemplarBank.build(df_train, cfg)
    bank.save("data/processed/exemplar_bank.pkl")
    
    # Load and use
    bank = ExemplarBank.load("data/processed/exemplar_bank.pkl")
    query_emb = get_embeddings(["some text"])[0]
    examples = bank.select(query_emb, "Diacritcs", k=2)
"""

import os
import pickle
import time
from pathlib import Path

import dotenv
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

from src.llm_classifier.constants import ATTACK_TYPES

# Load .env from project root or parent directories
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


def get_embeddings(
    texts: list[str],
    model: str = "nvidia/nv-embedqa-e5-v5",
    batch_size: int = 100,
    input_type: str | None = None,
) -> np.ndarray:
    """
    Get embeddings for a list of texts using NVIDIA NIM's embedding API.

    Args:
        texts: List of strings to embed
        model: Embedding model name
        batch_size: Number of texts per API call
        input_type: "passage" for bank texts, "query" for query texts (NIM-specific)

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    client = openai.OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
    )
    all_embeddings = []

    extra = {"extra_body": {"input_type": input_type}} if input_type else {}
    max_retries = 5

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, input=batch, **extra)
                break
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** attempt * 5, 60)
                print(f"\nRate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    # Normalize
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(b_norm, a_norm)


class ExemplarBank:
    """
    Stores embeddings for exemplar texts, organized by attack type.
    Enables fast cosine-similarity retrieval for dynamic few-shot.
    """
    
    def __init__(self):
        # Dict[attack_type -> {"texts": list[str], "embeddings": np.ndarray}]
        self.bank: dict[str, dict] = {}
        self.embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    
    @classmethod
    def build(
        cls,
        df_train: pd.DataFrame,
        cfg: dict,
        show_progress: bool = True,
    ) -> "ExemplarBank":
        """
        Build an exemplar bank from training data.
        
        Args:
            df_train: Training DataFrame with text and label columns
            cfg: Config dict with dataset and llm.few_shot settings
            show_progress: Whether to show tqdm progress bar
            
        Returns:
            Populated ExemplarBank instance
        """
        bank = cls()
        
        text_col = cfg["dataset"]["text_col"]
        label_col = cfg["dataset"]["label_col"]
        
        # Get settings from config
        few_shot_cfg = cfg.get("llm", {}).get("few_shot", {})
        bank_size = few_shot_cfg.get("bank_size_per_type", 15)
        bank.embedding_model = few_shot_cfg.get("embedding_model", "nvidia/nv-embedqa-e5-v5")
        
        # Combine unicode and nlp attack types
        all_types = cfg["labels"]["unicode_attacks"] + cfg["labels"]["nlp_attacks"]
        
        iterator = tqdm(all_types, desc="Building exemplar bank") if show_progress else all_types
        
        for attack_type in iterator:
            # Sample exemplars for this type
            pool = df_train.loc[df_train[label_col] == attack_type, text_col]
            n = min(bank_size, len(pool))
            if n == 0:
                continue

            samples = pool.sample(n=n, random_state=42).tolist()

            # Compute embeddings
            embeddings = get_embeddings(samples, model=bank.embedding_model, input_type="passage")

            bank.bank[attack_type] = {
                "texts": samples,
                "embeddings": embeddings,
            }

        # Also store benign examples for contrastive pairing
        benign_pool = df_train.loc[df_train[label_col] == "benign", text_col]
        n_benign = min(bank_size, len(benign_pool))
        if n_benign > 0:
            benign_samples = benign_pool.sample(n=n_benign, random_state=42).tolist()
            benign_embeddings = get_embeddings(benign_samples, model=bank.embedding_model, input_type="passage")
            bank.bank["benign"] = {
                "texts": benign_samples,
                "embeddings": benign_embeddings,
            }

        return bank
    
    def select(
        self,
        query_embedding: np.ndarray,
        attack_type: str,
        k: int = 2,
    ) -> list[dict]:
        """
        Select the k most similar exemplars for a given attack type.
        
        Args:
            query_embedding: Embedding vector of the query text
            attack_type: Which attack type to retrieve examples from
            k: Number of examples to retrieve
            
        Returns:
            List of {"text": str, "label": str} dicts, most similar first
        """
        if attack_type not in self.bank:
            return []
        
        data = self.bank[attack_type]
        texts = data["texts"]
        embeddings = data["embeddings"]
        
        # Compute similarities
        sims = cosine_similarity(query_embedding, embeddings)
        
        # Get top-k indices
        top_k_idx = np.argsort(sims)[::-1][:k]
        
        return [
            {"text": texts[i], "label": attack_type}
            for i in top_k_idx
        ]
    
    def select_multi_type(
        self,
        query_embedding: np.ndarray,
        attack_types: list[str],
        k_per_type: int = 1,
    ) -> list[dict]:
        """
        Select exemplars from multiple attack types.
        
        Args:
            query_embedding: Embedding vector of the query text
            attack_types: List of attack types to retrieve from
            k_per_type: Number of examples per type
            
        Returns:
            List of {"text": str, "label": str} dicts
        """
        results = []
        for attack_type in attack_types:
            results.extend(self.select(query_embedding, attack_type, k=k_per_type))
        return results

    def select_pairs_by_benign(
        self,
        query_embedding: np.ndarray,
        k: int = 2,
    ) -> list[tuple[str, str, str]]:
        """
        Select k pairs of (benign_text, attack_text, attack_type) for few-shot prompting.

        Retrieves the single most similar benign example and the k most similar attack
        examples across all attack types (one per type), then returns k pairs total.
        """
        benign_examples = self.select(query_embedding, "benign", k=1)
        benign_text = benign_examples[0]["text"] if benign_examples else ""
        attack_examples = self.select_multi_type(query_embedding, ATTACK_TYPES, k_per_type=1)
        return [
            (benign_text, a["text"], a["label"])
            for a in attack_examples[:k]
        ]


    def save(self, path: str) -> None:
        """Save the exemplar bank to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bank": self.bank,
                "embedding_model": self.embedding_model,
            }, f)
        print(f"ExemplarBank saved → {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExemplarBank":
        """Load an exemplar bank from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        bank = cls()
        bank.bank = data["bank"]
        bank.embedding_model = data.get("embedding_model", "nvidia/nv-embedqa-e5-v5")
        return bank
    
    def __repr__(self) -> str:
        types = list(self.bank.keys())
        total = sum(len(d["texts"]) for d in self.bank.values())
        return f"ExemplarBank({len(types)} types, {total} exemplars)"
