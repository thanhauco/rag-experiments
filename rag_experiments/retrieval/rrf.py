"""Reciprocal Rank Fusion (RRF) retriever."""

from __future__ import annotations

from typing import Any

from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import Index, SearchResult
from rag_experiments.retrieval.base import Retriever


class RRFRetriever(Retriever):
    """Retriever using Reciprocal Rank Fusion.
    
    RRF combines multiple rankings without needing score normalization.
    It's particularly effective for combining dense and sparse results.
    
    Formula: score = sum(1 / (k + rank))
    
    Args:
        index: The index to retrieve from (must support multiple result streams or internal hybrid).
        k: Smoothing constant for RRF (default: 60).
    """

    def __init__(self, index: Index, k: int = 60):
        super().__init__(index)
        self.k = k

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform RRF retrieval.
        
        If the index is a HybridIndex, we can get its separate streams.
        Otherwise, we treat it as a single stream (which is trivial RRF).
        """
        from rag_experiments.indexing.hybrid import HybridIndex
        
        if isinstance(self.index, HybridIndex):
            # Fetch from separate streams
            fetch_k = top_k * 2
            dense_results = self.index.dense_index.search(query, fetch_k)
            sparse_results = self.index.sparse_index.search(query, fetch_k)
            
            # Apply RRF
            rrf_scores: dict[str, float] = {}
            chunk_map: dict[str, Chunk] = {}
            
            # Helper to add to RRF dict
            def add_to_rrf(results: list[SearchResult]):
                for i, r in enumerate(results):
                    # Use content as key if chunk_id is missing
                    cid = r.chunk.chunk_id or r.chunk.content
                    rank = i + 1
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))
                    chunk_map[cid] = r.chunk
            
            add_to_rrf(dense_results)
            add_to_rrf(sparse_results)
            
            # Sort by RRF score
            sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
            
            results = []
            for i, cid in enumerate(sorted_cids[:top_k]):
                results.append(
                    SearchResult(
                        chunk=chunk_map[cid],
                        score=rrf_scores[cid],
                        rank=i + 1,
                        metadata={"retrieval_method": "rrf", "rrf_k": self.k}
                    )
                )
            return results
        else:
            # Fallback for non-hybrid index: just standard search
            return self.index.search(query, top_k)
