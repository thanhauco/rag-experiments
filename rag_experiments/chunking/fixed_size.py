"""Fixed-size chunking strategy."""

from __future__ import annotations

import uuid
from typing import Literal

from rag_experiments.chunking.base import Chunk, Chunker


class FixedSizeChunker(Chunker):
    """Chunk text into fixed-size segments with optional overlap.
    
    This is the simplest chunking strategy. It splits text into chunks
    of a fixed number of characters (or tokens) with configurable overlap.
    
    Args:
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.
        mode: Whether to count 'characters' or 'tokens' (simplified word split).
    
    Example:
        >>> chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        >>> chunks = chunker.chunk("Your long document text here...")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        mode: Literal["characters", "tokens"] = "characters",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.mode = mode

    @property
    def name(self) -> str:
        return "fixed_size"

    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks.
        
        Args:
            text: The input text to chunk.
            doc_id: Optional document identifier.
            
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        if self.mode == "tokens":
            return self._chunk_by_tokens(text, doc_id)
        return self._chunk_by_characters(text, doc_id)

    def _chunk_by_characters(self, text: str, doc_id: str | None) -> list[Chunk]:
        """Split text by character count."""
        chunks = []
        step = self.chunk_size - self.overlap
        start_idx = 0

        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]

            # Try to break at word boundary if not at end
            if end_idx < len(text) and not text[end_idx].isspace():
                last_space = chunk_text.rfind(" ")
                if last_space > self.chunk_size // 2:
                    chunk_text = chunk_text[:last_space]
                    end_idx = start_idx + last_space

            chunk = Chunk(
                content=chunk_text.strip(),
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_id=str(uuid.uuid4()),
                metadata={
                    "chunker": self.name,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                },
            )
            if doc_id:
                chunk.metadata["doc_id"] = doc_id

            chunks.append(chunk)
            start_idx += step

        return chunks

    def _chunk_by_tokens(self, text: str, doc_id: str | None) -> list[Chunk]:
        """Split text by approximate token count (word-based)."""
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        start_word_idx = 0

        char_offset = 0
        word_positions = []
        for word in words:
            pos = text.find(word, char_offset)
            word_positions.append(pos)
            char_offset = pos + len(word)

        while start_word_idx < len(words):
            end_word_idx = min(start_word_idx + self.chunk_size, len(words))
            chunk_words = words[start_word_idx:end_word_idx]
            chunk_text = " ".join(chunk_words)

            start_char = word_positions[start_word_idx]
            end_char = (
                word_positions[end_word_idx - 1] + len(words[end_word_idx - 1])
                if end_word_idx > 0
                else len(text)
            )

            chunk = Chunk(
                content=chunk_text,
                start_idx=start_char,
                end_idx=end_char,
                chunk_id=str(uuid.uuid4()),
                metadata={
                    "chunker": self.name,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "mode": "tokens",
                },
            )
            if doc_id:
                chunk.metadata["doc_id"] = doc_id

            chunks.append(chunk)
            start_word_idx += step

        return chunks

    def __repr__(self) -> str:
        return f"FixedSizeChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
