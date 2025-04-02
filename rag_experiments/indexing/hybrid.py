"""Hybrid index implementation combining dense and sparse indices."""

from __future__ import annotations

from typing import Any

from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import Index, SearchResult
from rag_experiments.indexing.dense import DenseIndex
from rag_experiments.indexing.sparse import SparseIndex


class HybridIndex(Index):
    """Hybrid index combining dense and sparse search.
    
    This index performs both vector similarity search and lexical
    matching, combining the results using weighted fusion.
    
    Args:
        dense_weight: Weight for dense scores (0.0 to 1.0).
        sparse_weight: Weight for sparse scores (0.0 to 1.0).
        dense_index: Optional pre-configured DenseIndex.
        sparse_index: Optional pre-configured SparseIndex.
    """

    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        dense_index: DenseIndex | None = None,
        sparse_index: SparseIndex | None = None,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.dense_index = dense_index or DenseIndex()
        self.sparse_index = sparse_index or SparseIndex()

    @property
    def name(self) -> str:
        return f"hybrid_d{self.dense_weight}_s{self.sparse_weight}"

    @property
    def size(self) -> int:
        return self.dense_index.size

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to both indices."""
        self.dense_index.add(chunks)
        self.sparse_index.add(chunks)

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return []
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r.score = 1.0
            return results
            
        for r in results:
            r.score = (r.score - min_score) / (max_score - min_score)
            
        return results

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform hybrid search with score fusion.
        
        This implementation uses linear weighted combination of normalized scores.
        """
        # Fetch more results from each to allow for fusion overlap
        fetch_k = top_k * 2
        
        dense_results = self.dense_index.search(query, fetch_k)
        sparse_results = self.sparse_index.search(query, fetch_k)
        
        # Normalize scores
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)
        
        # Combine results
        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        
        for r in dense_results:
            chunk_id = r.chunk.chunk_id or r.chunk.content
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (r.score * self.dense_weight)
            chunk_map[chunk_id] = r.chunk
            
        for r in sparse_results:
            chunk_id = r.chunk.chunk_id or r.chunk.content
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (r.score * self.sparse_weight)
            chunk_map[chunk_id] = r.chunk
            
        # Sort and create final results
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        results = []
        for i, cid in enumerate(sorted_ids[:top_k]):
            results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=fused_scores[cid],
                    rank=i + 1,
                    metadata={"index_type": "hybrid", "raw_hybrid_score": fused_scores[cid]}
                )
            )
            
        return results

    def clear(self) -> None:
        """Clear both indices."""
        self.dense_index.clear()
        self.sparse_index.clear()
