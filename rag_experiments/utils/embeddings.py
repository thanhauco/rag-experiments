"""Embedding model wrapper with caching support."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for sentence-transformers with local caching.
    
    Args:
        model_name: Name of the sentence-transformers model.
        cache_dir: Directory to store cached embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = ".cache/embeddings",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name)
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, texts: List[str]) -> str:
        """Generate a stable cache path for a list of texts."""
        # Create a hash of the texts and the model name
        content = json.dumps([self.model_name] + sorted(texts))
        text_hash = hashlib.sha256(content.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")

    def encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of strings to encode.
            use_cache: Whether to check/save to local cache.
            
        Returns:
            Numpy array of embeddings.
        """
        if not texts:
            return np.array([])

        if not use_cache or not self.cache_dir:
            return self.model.encode(texts, convert_to_numpy=True)

        cache_path = self._get_cache_path(texts)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
            
        return embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
