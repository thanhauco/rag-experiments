"""Semantic chunking strategy based on sentence boundaries."""

from __future__ import annotations

import re
import uuid
from typing import Callable

from rag_experiments.chunking.base import Chunk, Chunker


def simple_sentence_splitter(text: str) -> list[str]:
    """Split text into sentences using regex patterns.
    
    This is a simple fallback when NLTK is not available.
    """
    # Handle common abbreviations
    text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd)\.", r"\1<DOT>", text)
    
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    # Restore abbreviations
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(Chunker):
    """Chunk text based on semantic boundaries (sentences/paragraphs).
    
    This strategy respects natural language boundaries, grouping sentences
    together until reaching the maximum size. It produces more coherent
    chunks than fixed-size chunking.
    
    Args:
        max_sentences: Maximum number of sentences per chunk.
        max_characters: Maximum character length per chunk.
        min_sentences: Minimum sentences before starting a new chunk.
        sentence_splitter: Optional custom sentence splitting function.
        respect_paragraphs: Whether to break at paragraph boundaries.
    
    Example:
        >>> chunker = SemanticChunker(max_sentences=5)
        >>> chunks = chunker.chunk("First sentence. Second sentence. ...")
    """

    def __init__(
        self,
        max_sentences: int = 5,
        max_characters: int = 1000,
        min_sentences: int = 1,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        respect_paragraphs: bool = True,
    ):
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if min_sentences < 1:
            raise ValueError("min_sentences must be at least 1")

        self.max_sentences = max_sentences
        self.max_characters = max_characters
        self.min_sentences = min_sentences
        self.respect_paragraphs = respect_paragraphs
        
        # Try to use NLTK, fall back to simple splitter
        if sentence_splitter is None:
            try:
                import nltk
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                self._sentence_splitter = nltk.sent_tokenize
            except (ImportError, LookupError):
                self._sentence_splitter = simple_sentence_splitter
        else:
            self._sentence_splitter = sentence_splitter

    @property
    def name(self) -> str:
        return "semantic"

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _get_sentence_positions(
        self, full_text: str, sentences: list[str]
    ) -> list[tuple[int, int]]:
        """Find start and end positions of each sentence in the original text."""
        positions = []
        current_pos = 0
        
        for sentence in sentences:
            # Find the sentence in the text starting from current position
            start = full_text.find(sentence, current_pos)
            if start == -1:
                # Fallback: use approximate position
                start = current_pos
            end = start + len(sentence)
            positions.append((start, end))
            current_pos = end
            
        return positions

    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """Split text into semantically coherent chunks.
        
        Args:
            text: The input text to chunk.
            doc_id: Optional document identifier.
            
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        chunks = []

        if self.respect_paragraphs:
            paragraphs = self._split_into_paragraphs(text)
        else:
            paragraphs = [text]

        for para in paragraphs:
            sentences = self._sentence_splitter(para)
            if not sentences:
                continue

            positions = self._get_sentence_positions(text, sentences)
            
            current_sentences: list[str] = []
            current_start: int | None = None
            current_end: int = 0
            current_length = 0

            for i, sentence in enumerate(sentences):
                sentence_start, sentence_end = positions[i]
                sentence_length = len(sentence)

                # Check if adding this sentence exceeds limits
                would_exceed_sentences = (
                    len(current_sentences) >= self.max_sentences
                )
                would_exceed_chars = (
                    current_length + sentence_length > self.max_characters
                    and len(current_sentences) >= self.min_sentences
                )

                if current_sentences and (would_exceed_sentences or would_exceed_chars):
                    # Create chunk from accumulated sentences
                    chunk = Chunk(
                        content=" ".join(current_sentences),
                        start_idx=current_start or 0,
                        end_idx=current_end,
                        chunk_id=str(uuid.uuid4()),
                        metadata={
                            "chunker": self.name,
                            "sentence_count": len(current_sentences),
                        },
                    )
                    if doc_id:
                        chunk.metadata["doc_id"] = doc_id
                    chunks.append(chunk)

                    # Reset accumulators
                    current_sentences = []
                    current_start = None
                    current_length = 0

                # Add sentence to current chunk
                current_sentences.append(sentence)
                if current_start is None:
                    current_start = sentence_start
                current_end = sentence_end
                current_length += sentence_length

            # Handle remaining sentences
            if current_sentences:
                chunk = Chunk(
                    content=" ".join(current_sentences),
                    start_idx=current_start or 0,
                    end_idx=current_end,
                    chunk_id=str(uuid.uuid4()),
                    metadata={
                        "chunker": self.name,
                        "sentence_count": len(current_sentences),
                    },
                )
                if doc_id:
                    chunk.metadata["doc_id"] = doc_id
                chunks.append(chunk)

        return chunks

    def __repr__(self) -> str:
        return (
            f"SemanticChunker(max_sentences={self.max_sentences}, "
            f"max_characters={self.max_characters})"
        )
