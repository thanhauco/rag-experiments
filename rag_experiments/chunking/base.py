"""Base classes for chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A chunk of text with metadata.
    
    Attributes:
        content: The text content of the chunk.
        start_idx: Starting character index in the original document.
        end_idx: Ending character index in the original document.
        metadata: Additional metadata (e.g., parent_id, section_title).
        chunk_id: Unique identifier for this chunk.
        parent_id: ID of parent chunk for hierarchical chunking.
    """

    content: str
    start_idx: int
    end_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None
    parent_id: str | None = None

    def __len__(self) -> int:
        """Return the length of the chunk content."""
        return len(self.content)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(len={len(self)}, preview='{preview}')"

    @property
    def char_length(self) -> int:
        """Return the character length of the content."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "content": self.content,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(**data)


class Chunker(ABC):
    """Abstract base class for text chunking strategies.
    
    Implementations should override the `chunk` method to provide
    specific chunking logic (fixed-size, semantic, hierarchical, etc.).
    """

    @abstractmethod
    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """Split text into chunks.
        
        Args:
            text: The input text to chunk.
            doc_id: Optional document identifier for metadata.
            
        Returns:
            List of Chunk objects.
        """
        pass

    def chunk_documents(
        self, documents: list[str], doc_ids: list[str] | None = None
    ) -> list[Chunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of document texts.
            doc_ids: Optional list of document identifiers.
            
        Returns:
            Combined list of chunks from all documents.
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        all_chunks = []
        for doc_text, doc_id in zip(documents, doc_ids):
            chunks = self.chunk(doc_text, doc_id)
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
            all_chunks.extend(chunks)

        return all_chunks

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this chunking strategy."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
