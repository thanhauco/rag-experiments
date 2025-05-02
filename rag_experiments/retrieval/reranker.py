"""Cross-encoder reranker for improving retrieval precision."""

from __future__ import annotations

from typing import List, Callable, Any

from rag_experiments.indexing.base import SearchResult


class CrossEncoderReranker:
    """Reranker using cross-encoder model for more accurate scoring.
    
    Cross-encoders jointly encode query and document, providing more
    accurate relevance scores than bi-encoders at the cost of speed.
    
    Args:
        model_name: Cross-encoder model name.
        top_k: Number of results to return after reranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None  # Lazy load

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                # Fallback to mock scoring
                self._model = "mock"

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using cross-encoder.
        
        Args:
            query: Original search query.
            results: Initial search results to rerank.
            
        Returns:
            Reranked results with updated scores.
        """
        if not results:
            return []

        self._load_model()

        if self._model == "mock":
            # Mock reranking: just return as-is
            return results[:self.top_k]

        # Prepare query-document pairs
        pairs = [(query, r.chunk.content) for r in results]
        
        # Get cross-encoder scores
        scores = self._model.predict(pairs)
        
        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)
            result.metadata["reranked"] = True
            result.metadata["reranker"] = self.model_name

        # Sort by new scores
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for i, r in enumerate(results[:self.top_k]):
            r.rank = i + 1

        return results[:self.top_k]


class ColBERTReranker:
    """Late interaction reranker using ColBERT-style scoring.
    
    ColBERT computes token-level interactions between query and document,
    offering a balance between accuracy and efficiency.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using simulated ColBERT scoring."""
        if not results:
            return []

        # Simulate MaxSim scoring (simplified)
        query_tokens = set(query.lower().split())
        
        for result in results:
            doc_tokens = set(result.chunk.content.lower().split())
            # Jaccard-like scoring as proxy
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            result.score = intersection / union if union > 0 else 0
            result.metadata["reranker"] = "colbert_sim"

        results.sort(key=lambda r: r.score, reverse=True)
        
        for i, r in enumerate(results[:self.top_k]):
            r.rank = i + 1

        return results[:self.top_k]
