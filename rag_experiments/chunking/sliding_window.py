"""Sliding window chunking strategy."""

from __future__ import annotations

import uuid

from rag_experiments.chunking.base import Chunk, Chunker
from rag_experiments.chunking.semantic import simple_sentence_splitter


class SlidingWindowChunker(Chunker):
    """Chunk text using a sliding window approach.
    
    This strategy creates overlapping chunks by sliding a window across
    the text. Unlike fixed-size chunking, it can snap to sentence
    boundaries for more coherent chunks.
    
    Args:
        window_size: Size of the sliding window in characters.
        stride: Number of characters to advance between windows.
        snap_to_sentences: Whether to adjust boundaries to sentence endings.
        min_chunk_size: Minimum size for a valid chunk.
    
    Example:
        >>> chunker = SlidingWindowChunker(window_size=500, stride=250)
        >>> chunks = chunker.chunk("Your long document text...")
    """

    def __init__(
        self,
        window_size: int = 500,
        stride: int = 250,
        snap_to_sentences: bool = True,
        min_chunk_size: int = 50,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride > window_size:
            raise ValueError("stride must be <= window_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative")

        self.window_size = window_size
        self.stride = stride
        self.snap_to_sentences = snap_to_sentences
        self.min_chunk_size = min_chunk_size

    @property
    def name(self) -> str:
        return "sliding_window"

    def _find_sentence_boundary(
        self, text: str, pos: int, direction: str = "backward"
    ) -> int:
        """Find the nearest sentence boundary.
        
        Args:
            text: The full text.
            pos: Current position.
            direction: 'backward' to find previous boundary, 'forward' for next.
            
        Returns:
            Position of the sentence boundary.
        """
        sentence_endings = ".!?"
        
        if direction == "backward":
            # Look backward for sentence ending
            search_start = max(0, pos - 100)
            search_text = text[search_start:pos]
            
            for i in range(len(search_text) - 1, -1, -1):
                if search_text[i] in sentence_endings:
                    # Check if followed by space (actual sentence end)
                    absolute_pos = search_start + i + 1
                    if absolute_pos < len(text) and text[absolute_pos].isspace():
                        return absolute_pos
            
            # No sentence boundary found, try word boundary
            for i in range(len(search_text) - 1, -1, -1):
                if search_text[i].isspace():
                    return search_start + i + 1
                    
            return pos
            
        else:  # forward
            # Look forward for sentence ending
            search_end = min(len(text), pos + 100)
            search_text = text[pos:search_end]
            
            for i, char in enumerate(search_text):
                if char in sentence_endings:
                    absolute_pos = pos + i + 1
                    if absolute_pos < len(text) and text[absolute_pos].isspace():
                        return absolute_pos + 1
            
            # No sentence boundary found, try word boundary
            for i, char in enumerate(search_text):
                if char.isspace():
                    return pos + i + 1
                    
            return pos

    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """Split text using sliding window.
        
        Args:
            text: The input text to chunk.
            doc_id: Optional document identifier.
            
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        if len(text) <= self.window_size:
            # Text fits in single window
            chunk = Chunk(
                content=text.strip(),
                start_idx=0,
                end_idx=len(text),
                chunk_id=str(uuid.uuid4()),
                metadata={
                    "chunker": self.name,
                    "window_size": self.window_size,
                    "stride": self.stride,
                    "window_index": 0,
                },
            )
            if doc_id:
                chunk.metadata["doc_id"] = doc_id
            return [chunk]

        chunks = []
        window_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Calculate window end
            end_pos = min(start_pos + self.window_size, len(text))
            
            # Snap to sentence boundaries if enabled
            if self.snap_to_sentences and end_pos < len(text):
                snapped_end = self._find_sentence_boundary(text, end_pos, "forward")
                # Only use snapped position if it doesn't make window too large
                if snapped_end - start_pos <= self.window_size * 1.2:
                    end_pos = snapped_end

            # Extract chunk content
            chunk_text = text[start_pos:end_pos].strip()
            
            # Skip if chunk is too small (except for last chunk)
            if len(chunk_text) >= self.min_chunk_size or start_pos + self.stride >= len(text):
                chunk = Chunk(
                    content=chunk_text,
                    start_idx=start_pos,
                    end_idx=end_pos,
                    chunk_id=str(uuid.uuid4()),
                    metadata={
                        "chunker": self.name,
                        "window_size": self.window_size,
                        "stride": self.stride,
                        "window_index": window_index,
                    },
                )
                if doc_id:
                    chunk.metadata["doc_id"] = doc_id
                    
                # Calculate overlap info
                if chunks:
                    overlap_chars = max(0, chunks[-1].end_idx - start_pos)
                    chunk.metadata["overlap_with_previous"] = overlap_chars
                    
                chunks.append(chunk)
                window_index += 1

            # Advance window
            start_pos += self.stride
            
            # Snap start to sentence boundary if enabled
            if self.snap_to_sentences and start_pos < len(text):
                snapped_start = self._find_sentence_boundary(text, start_pos, "backward")
                # Only use if it doesn't move too far back
                if start_pos - snapped_start <= self.stride // 2:
                    start_pos = snapped_start

        return chunks

    def get_overlap_percentage(self) -> float:
        """Calculate the overlap percentage between consecutive windows."""
        return (self.window_size - self.stride) / self.window_size * 100

    def __repr__(self) -> str:
        overlap_pct = self.get_overlap_percentage()
        return (
            f"SlidingWindowChunker(window_size={self.window_size}, "
            f"stride={self.stride}, overlap={overlap_pct:.1f}%)"
        )
