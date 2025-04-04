"""Dense vector index implementation using FAISS."""

from __future__ import annotations

import os
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import Index, SearchResult
from rag_experiments.utils.embeddings import EmbeddingModel


class DenseIndex(Index):
    """Dense vector index using FAISS.
    
    This index embeds text chunks into a vector space and performs
    similarity search using FAISS (Facebook AI Similarity Search).
    
    Args:
        model_name: Name of the sentence-transformers model to use.
        metric: Similarity metric ('l2' or 'inner_product').
        index_type: FAISS index type ('flat', 'ivf', 'hnsw').
        cache_dir: Directory for embedding cache.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        metric: str = "inner_product",
        index_type: str = "flat",
        cache_dir: str = ".cache/embeddings",
    ):
        self.model_name = model_name
        self.metric = metric
        self.index_type = index_type
        
        # Load embedding model wrapper
        self.embedding_model = EmbeddingModel(model_name, cache_dir=cache_dir)
        self.dimension = self.embedding_model.dimension
        
        # Initialize FAISS index
        self._init_index()
        
        # Storage for chunks (FAISS only stores vectors)
        self.chunks: list[Chunk] = []

    def _init_index(self) -> None:
        """Initialize appropriate FAISS index."""
        if self.metric == "l2":
            faiss_metric = faiss.METRIC_L2
        else:
            faiss_metric = faiss.METRIC_INNER_PRODUCT

        if self.index_type == "flat":
            if self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss_metric)
        else:
            # Fallback to flat
            self.index = faiss.IndexFlatIP(self.dimension)

    @property
    def name(self) -> str:
        return f"dense_{self.model_name}_{self.index_type}"

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.
        
        Args:
            chunks: List of chunks to index.
        """
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)
            
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for relevant chunks.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not self.chunks:
            return []

        query_embedding = self.embedding_model.encode([query])
        
        if self.metric == "inner_product":
            faiss.normalize_L2(query_embedding)
            
        scores, indices = self.index.search(
            query_embedding.astype("float32"), min(top_k, len(self.chunks))
        )
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or idx >= len(self.chunks):
                continue
                
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=i + 1,
                    metadata={"index_type": "dense"}
                )
            )
            
        return results

    def clear(self) -> None:
        """Clear the index."""
        self._init_index()
        self.chunks = []

    def save(self, path: str) -> None:
        """Save index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".faiss")
        # In a real app we'd also save chunks, but for experiments we usually rebuild

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(path + ".faiss")
        # Warning: chunks list needs to be repopulated separately
