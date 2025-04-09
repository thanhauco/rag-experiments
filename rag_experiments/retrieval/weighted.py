"""Weighted combination retriever."""

from __future__ import annotations

from typing import Any

from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import Index, SearchResult
from rag_experiments.retrieval.base import Retriever


class WeightedRetriever(Retriever):
    """Retriever that combines multiple indices with weighted scores.
    
    This is useful for combining fundamentally different search methods
    where scores can be normalized.
    
    Args:
        index: Primary index (usually a HybridIndex or a container).
        weights: Dictionary of index names to weights.
    """

    def __init__(self, index: Index, weights: dict[str, float] | None = None):
        super().__init__(index)
        self.weights = weights or {"dense": 0.5, "sparse": 0.5}

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results: return []
        max_s = max(r.score for r in results)
        min_s = min(r.score for r in results)
        if max_s == min_s: 
            for r in results: r.score = 1.0
            return results
        for r in results:
            r.score = (r.score - min_s) / (max_s - min_s)
        return results

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        from rag_experiments.indexing.hybrid import HybridIndex
        
        if not isinstance(self.index, HybridIndex):
            return self.index.search(query, top_k)
            
        fetch_k = top_k * 2
        dense_results = self._normalize_scores(self.index.dense_index.search(query, fetch_k))
        sparse_results = self._normalize_scores(self.index.sparse_index.search(query, fetch_k))
        
        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        
        for r in dense_results:
            cid = r.chunk.chunk_id or r.chunk.content
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (r.score * self.weights.get("dense", 0.5))
            chunk_map[cid] = r.chunk
            
        for r in sparse_results:
            cid = r.chunk.chunk_id or r.chunk.content
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (r.score * self.weights.get("sparse", 0.5))
            chunk_map[cid] = r.chunk
            
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        
        results = []
        for i, cid in enumerate(sorted_cids[:top_k]):
            results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=fused_scores[cid],
                    rank=i + 1,
                    metadata={"retrieval_method": "weighted_fusion"}
                )
            )
        return results
