"""Query router for directing queries to appropriate retrievers."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from rag_experiments.indexing.base import Index, SearchResult
from rag_experiments.retrieval.base import Retriever


class QueryType(str, Enum):
    """Types of queries for routing."""
    
    FACTUAL = "factual"          # Simple fact lookup
    ANALYTICAL = "analytical"    # Complex reasoning needed
    COMPARATIVE = "comparative"  # Comparing entities/concepts
    PROCEDURAL = "procedural"    # How-to questions
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """Result of query routing decision."""
    
    query_type: QueryType
    confidence: float
    selected_retriever: str
    reasoning: str


class QueryRouter:
    """Routes queries to appropriate retrieval strategies.
    
    Different query types may benefit from different retrieval approaches:
    - Factual: Dense retrieval often sufficient
    - Analytical: Hybrid with heavier semantic weight
    - Comparative: Multi-query expansion helpful
    - Procedural: Hierarchical chunking beneficial
    
    Args:
        retrievers: Dictionary mapping retriever names to instances.
        default_retriever: Fallback retriever name.
    """

    def __init__(
        self,
        retrievers: Dict[str, Retriever],
        default_retriever: str = "hybrid",
    ):
        self.retrievers = retrievers
        self.default_retriever = default_retriever
        
        # Query type patterns (simplified keyword matching)
        self.patterns = {
            QueryType.FACTUAL: ["what is", "who is", "when was", "where is", "define"],
            QueryType.ANALYTICAL: ["why", "explain", "analyze", "what causes", "how does"],
            QueryType.COMPARATIVE: ["compare", "difference", "versus", "vs", "better than"],
            QueryType.PROCEDURAL: ["how to", "steps to", "guide", "tutorial", "process"],
        }
        
        # Recommended retrievers per query type
        self.routing_table = {
            QueryType.FACTUAL: "dense",
            QueryType.ANALYTICAL: "hybrid",
            QueryType.COMPARATIVE: "query_expansion",
            QueryType.PROCEDURAL: "hybrid",
            QueryType.UNKNOWN: default_retriever,
        }

    def classify_query(self, query: str) -> tuple[QueryType, float]:
        """Classify query type based on patterns.
        
        Returns:
            Tuple of (QueryType, confidence_score).
        """
        query_lower = query.lower()
        
        for qtype, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return qtype, 0.8
        
        return QueryType.UNKNOWN, 0.5

    def route(self, query: str) -> RoutingDecision:
        """Determine optimal retriever for query.
        
        Args:
            query: The search query.
            
        Returns:
            RoutingDecision with selected retriever.
        """
        query_type, confidence = self.classify_query(query)
        
        recommended = self.routing_table.get(query_type, self.default_retriever)
        
        # Fall back if recommended retriever not available
        if recommended not in self.retrievers:
            recommended = self.default_retriever
            
        return RoutingDecision(
            query_type=query_type,
            confidence=confidence,
            selected_retriever=recommended,
            reasoning=f"Query classified as {query_type.value}, routing to {recommended}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> tuple[List[SearchResult], RoutingDecision]:
        """Route query and retrieve results.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            
        Returns:
            Tuple of (results, routing_decision).
        """
        decision = self.route(query)
        retriever = self.retrievers.get(decision.selected_retriever)
        
        if retriever is None:
            retriever = list(self.retrievers.values())[0]
            
        results = retriever.retrieve(query, top_k)
        
        # Add routing metadata to results
        for r in results:
            r.metadata["routed_via"] = decision.selected_retriever
            r.metadata["query_type"] = decision.query_type.value
            
        return results, decision
