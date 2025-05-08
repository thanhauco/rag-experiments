"""Context compressor to reduce retrieved context size while preserving relevance."""

from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass

from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import SearchResult


@dataclass
class CompressedContext:
    """Compressed context for LLM consumption."""
    
    text: str
    original_chunks: int
    compressed_chunks: int
    compression_ratio: float
    metadata: dict


class ContextCompressor:
    """Compresses retrieved context to fit LLM context windows.
    
    Long-context retrieval can exceed LLM limits or introduce noise.
    This compressor filters and summarizes to maintain quality.
    
    Args:
        max_tokens: Maximum tokens in compressed output.
        strategy: Compression strategy ('truncate', 'filter', 'summarize').
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        strategy: str = "filter",
        min_score_threshold: float = 0.3,
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.min_score_threshold = min_score_threshold

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def compress(self, results: List[SearchResult]) -> CompressedContext:
        """Compress search results into condensed context.
        
        Args:
            results: Retrieved search results.
            
        Returns:
            CompressedContext with filtered/summarized text.
        """
        if not results:
            return CompressedContext(
                text="",
                original_chunks=0,
                compressed_chunks=0,
                compression_ratio=1.0,
                metadata={}
            )

        original_count = len(results)
        
        if self.strategy == "filter":
            results = self._filter_by_score(results)
        
        # Accumulate chunks until token limit
        selected_chunks = []
        total_tokens = 0
        
        for r in results:
            chunk_tokens = self._estimate_tokens(r.chunk.content)
            if total_tokens + chunk_tokens > self.max_tokens:
                if self.strategy == "truncate":
                    # Truncate the last chunk
                    remaining = self.max_tokens - total_tokens
                    truncated = r.chunk.content[:remaining * 4]
                    selected_chunks.append(truncated)
                break
            selected_chunks.append(r.chunk.content)
            total_tokens += chunk_tokens

        compressed_text = "\n\n---\n\n".join(selected_chunks)
        
        return CompressedContext(
            text=compressed_text,
            original_chunks=original_count,
            compressed_chunks=len(selected_chunks),
            compression_ratio=len(selected_chunks) / original_count if original_count > 0 else 1.0,
            metadata={
                "strategy": self.strategy,
                "max_tokens": self.max_tokens,
                "actual_tokens": self._estimate_tokens(compressed_text)
            }
        )

    def _filter_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results below score threshold."""
        return [r for r in results if r.score >= self.min_score_threshold]


class LongContextHandler:
    """Handles very long contexts using sliding window or map-reduce."""
    
    def __init__(self, window_size: int = 1024, overlap: int = 128):
        self.window_size = window_size
        self.overlap = overlap

    def split_context(self, text: str) -> List[str]:
        """Split long context into overlapping windows."""
        if len(text) <= self.window_size * 4:
            return [text]
            
        windows = []
        step = (self.window_size - self.overlap) * 4
        
        for i in range(0, len(text), step):
            window = text[i:i + self.window_size * 4]
            windows.append(window)
            
        return windows
