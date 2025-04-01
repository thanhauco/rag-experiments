"""Sparse lexical index implementation using BM25."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import Index, SearchResult


class SparseIndex(Index):
    """Sparse index using BM25 for lexical matching.
    
    This index uses the BM25 algorithm to score chunks based on
    keyword overlap with the query.
    
    Args:
        language: Language for tokenization (default: 'english').
        remove_stopwords: Whether to filter out common stopwords.
    """

    def __init__(self, language: str = "english", remove_stopwords: bool = True):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        
        # Simple stopword list
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "from", "for", "with", "in", "on", "to", "of", "is", "was", "be"
        } if remove_stopwords else set()

    @property
    def name(self) -> str:
        return "sparse_bm25"

    @property
    def size(self) -> int:
        return len(self.chunks)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        tokens = re.findall(r"\w+", text.lower())
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.
        
        Args:
            chunks: List of chunks to index.
        """
        if not chunks:
            return

        self.chunks.extend(chunks)
        tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for relevant chunks.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not self.chunks or self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
                
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=i + 1,
                    metadata={"index_type": "sparse"}
                )
            )
            
        return results

    def clear(self) -> None:
        """Clear the index."""
        self.chunks = []
        self.bm25 = None
