"""Query expansion and HyDE (Hypothetical Document Embeddings) retrievers."""

from __future__ import annotations

import uuid
from typing import Callable

from rag_experiments.indexing.base import Index, SearchResult
from rag_experiments.retrieval.base import Retriever


class QueryExpansionRetriever(Retriever):
    """Retriever that uses query expansion or rewriting.
    
    Args:
        index: The index to retrieve from.
        expansion_fn: Function that takes a query and returns list of expanded queries.
        combine_results: Whether to combine results (union) or just take top from all.
    """

    def __init__(
        self,
        index: Index,
        expansion_fn: Callable[[str], list[str]] | None = None,
        combine_results: bool = True,
    ):
        super().__init__(index)
        self.expansion_fn = expansion_fn or self._default_expansion
        self.combine_results = combine_results

    def _default_expansion(self, query: str) -> list[str]:
        """Simple rule-based expansion if no LLM provided."""
        # This is a placeholder for actual LLM-based query rewriting
        return [query]

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        queries = self.expansion_fn(query)
        if not queries:
            queries = [query]
            
        all_results: list[SearchResult] = []
        seen_chunks = set()
        
        for q in queries:
            results = self.index.search(q, top_k)
            for r in results:
                # Deduplicate by content or ID
                cid = r.chunk.chunk_id or r.chunk.content
                if cid not in seen_chunks:
                    all_results.append(r)
                    seen_chunks.add(cid)
                    
        # Sort by original score and return top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]


class HyDERetriever(Retriever):
    """HyDE (Hypothetical Document Embeddings) retriever.
    
    It generates a hypothetical answer to the query and uses that to
    perform search, as the hypothetical answer's vector often matches
    the correct document better than the query vector.
    
    Args:
        index: The index to retrieve from.
        hypothetical_gen_fn: Function that generates a hypothetical answer.
    """

    def __init__(
        self,
        index: Index,
        hypothetical_gen_fn: Callable[[str], str] | None = None,
    ):
        super().__init__(index)
        self.hypothetical_gen_fn = hypothetical_gen_fn or self._mock_generator

    def _mock_generator(self, query: str) -> str:
        """Mock hypothetical answer generator."""
        # In practice, this would call an LLM
        return f"This is a hypothetical answer to the question: {query}"

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        hypothetical_doc = self.hypothetical_gen_fn(query)
        # Search using the hypothetical document instead of the query
        results = self.index.search(hypothetical_doc, top_k)
        
        for r in results:
            r.metadata["retrieval_method"] = "hyde"
            r.metadata["hypothetical_doc"] = hypothetical_doc
            
        return results
