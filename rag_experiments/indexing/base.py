"""Base classes for indexing schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_experiments.chunking.base import Chunk


@dataclass
class SearchResult:
    """A search result from an index.
    
    Attributes:
        chunk: The matching chunk.
        score: Relevance score (higher is better).
        rank: Position in the result list (1-indexed).
        metadata: Additional result metadata.
    """

    chunk: Chunk
    score: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.chunk.content[:40] + "..." if len(self.chunk.content) > 40 else self.chunk.content
        return f"SearchResult(score={self.score:.3f}, rank={self.rank}, preview='{preview}')"

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class Index(ABC):
    """Abstract base class for search indices.
    
    Implementations should provide methods to add chunks and search
    for relevant chunks given a query.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.
        
        Args:
            chunks: List of chunks to index.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for relevant chunks.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed chunks."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed chunks."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this index type."""
        pass

    def search_with_filter(
        self, query: str, top_k: int = 5, filter_fn: callable = None
    ) -> list[SearchResult]:
        """Search with optional post-filtering.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            filter_fn: Optional function to filter results (takes SearchResult, returns bool).
            
        Returns:
            Filtered list of SearchResult objects.
        """
        # Fetch more results if filtering to ensure we get enough
        fetch_k = top_k * 3 if filter_fn else top_k
        results = self.search(query, fetch_k)
        
        if filter_fn:
            results = [r for r in results if filter_fn(r)]
        
        return results[:top_k]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
