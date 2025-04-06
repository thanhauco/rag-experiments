"""Base class and common utilities for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_experiments.indexing.base import Index, SearchResult


class Retriever(ABC):
    """Abstract base class for retrieval strategies.
    
    A retriever takes a query and returns matched results from an index,
    possibly applying multi-stage processing or fusion.
    """

    def __init__(self, index: Index):
        self.index = index

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve relevant results for a query.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects.
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of this retriever."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(index={self.index.name})"
