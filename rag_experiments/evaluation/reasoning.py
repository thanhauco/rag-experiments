"""Reasoning accuracy and failure analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from rag_experiments.indexing.base import SearchResult


class FailureMode(str, Enum):
    """Common failure modes in RAG systems."""
    
    NO_RELEVANT_DOCS = "no_relevant_docs"
    RELEVANT_DOCS_LOW_RANK = "relevant_docs_low_rank"
    CONTEXT_TOO_SPARSE = "context_too_sparse"
    REASONING_ERROR = "reasoning_error"
    SUCCESS = "success"


class FailureAnalyzer:
    """Analyzes and categorizes RAG failures.
    
    This helps identify if the failure happened at retrieval or reasoning stage.
    """

    @staticmethod
    def analyze(
        retrieved: list[SearchResult],
        ground_truth_ids: list[str],
        reasoning_success: bool,
        k: int = 5
    ) -> FailureMode:
        """Analyze a single failure case."""
        retrieved_ids = [r.chunk.chunk_id or r.chunk.content for r in retrieved]
        top_k_ids = retrieved_ids[:k]
        
        # Check if anything relevant was retrieved at all
        has_any_relevant = any(rid in ground_truth_ids for rid in retrieved_ids)
        if not has_any_relevant:
            return FailureMode.NO_RELEVANT_DOCS
            
        # Check if relevant documents were in top k
        has_relevant_in_top_k = any(rid in ground_truth_ids for rid in top_k_ids)
        if not has_relevant_in_top_k:
            return FailureMode.RELEVANT_DOCS_LOW_RANK
            
        # If retrieval was okay but reasoning failed
        if not reasoning_success:
            return FailureMode.REASONING_ERROR
            
        return FailureMode.SUCCESS


class ReasoningScorer:
    """Scores LLM reasoning based on retrieved context.
    
    Since we don't have a live LLM for every test, we can use heuristics
    or semantic similarity to score predicted vs actual answers.
    """

    def score(self, predicted_answer: str, reference_answer: str) -> float:
        """Score a predicted answer against a reference."""
        # Simple exact/fuzzy match for experiments
        p = predicted_answer.lower().strip()
        r = reference_answer.lower().strip()
        
        if p == r:
            return 1.0
        
        # Jaccard similarity
        p_tokens = set(p.split())
        r_tokens = set(r.split())
        if not p_tokens or not r_tokens:
            return 0.0
            
        intersection = len(p_tokens.intersection(r_tokens))
        union = len(p_tokens.union(r_tokens))
        return intersection / union
