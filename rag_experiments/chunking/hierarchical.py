"""Hierarchical chunking strategy with parent-child relationships."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from rag_experiments.chunking.base import Chunk, Chunker
from rag_experiments.chunking.semantic import simple_sentence_splitter


@dataclass
class HierarchyLevel:
    """Configuration for a hierarchy level."""

    name: str
    pattern: str | None = None  # Regex pattern to detect this level
    max_size: int = 1000  # Max characters for this level


class HierarchicalChunker(Chunker):
    """Chunk text into hierarchical parent-child structures.
    
    This strategy creates a tree of chunks where parent chunks provide
    context for child chunks. During retrieval, if a child chunk matches,
    its parent context can be included for better understanding.
    
    Default hierarchy:
    - Document (full text)
      - Section (based on headers or large paragraphs)
        - Paragraph
          - Sentence (optional leaf level)
    
    Args:
        levels: List of hierarchy levels to create.
        include_parent_content: Whether to include parent summary in child metadata.
        section_pattern: Regex to detect section headers.
        min_section_length: Minimum characters for a section.
    
    Example:
        >>> chunker = HierarchicalChunker()
        >>> chunks = chunker.chunk("# Section 1\\nParagraph text...")
        >>> # Returns chunks with parent_id linking to parent chunks
    """

    DEFAULT_SECTION_PATTERN = r"^#{1,3}\s+.+$|^[A-Z][^.!?]*:$"

    def __init__(
        self,
        levels: list[HierarchyLevel] | None = None,
        include_parent_content: bool = True,
        section_pattern: str | None = None,
        min_section_length: int = 100,
        max_paragraph_sentences: int = 5,
    ):
        self.levels = levels or [
            HierarchyLevel("section", max_size=2000),
            HierarchyLevel("paragraph", max_size=500),
        ]
        self.include_parent_content = include_parent_content
        self.section_pattern = re.compile(
            section_pattern or self.DEFAULT_SECTION_PATTERN, re.MULTILINE
        )
        self.min_section_length = min_section_length
        self.max_paragraph_sentences = max_paragraph_sentences

    @property
    def name(self) -> str:
        return "hierarchical"

    def _detect_sections(self, text: str) -> list[tuple[str, str, int, int]]:
        """Detect sections in the text.
        
        Returns list of (title, content, start_idx, end_idx) tuples.
        """
        sections = []
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            # No sections found, treat entire text as one section
            return [("", text, 0, len(text))]

        for i, match in enumerate(matches):
            title = match.group().strip()
            start_idx = match.end()
            
            # Find end of this section (start of next or end of text)
            if i + 1 < len(matches):
                end_idx = matches[i + 1].start()
            else:
                end_idx = len(text)

            content = text[start_idx:end_idx].strip()
            
            if len(content) >= self.min_section_length or not sections:
                sections.append((title, content, match.start(), end_idx))
            elif sections:
                # Merge short section with previous
                prev_title, prev_content, prev_start, _ = sections[-1]
                sections[-1] = (
                    prev_title,
                    prev_content + "\n\n" + title + "\n" + content,
                    prev_start,
                    end_idx,
                )

        # Handle content before first section header
        if matches and matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.insert(0, ("", preamble, 0, matches[0].start()))

        return sections

    def _split_into_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into paragraphs with positions.
        
        Returns list of (paragraph_text, start_idx, end_idx) tuples.
        """
        paragraphs = []
        current_pos = 0
        
        # Split on double newlines
        parts = re.split(r"\n\s*\n", text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Find actual position in text
            start = text.find(part, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(part)
            
            paragraphs.append((part, start, end))
            current_pos = end

        return paragraphs

    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """Split text into hierarchical chunks.
        
        Args:
            text: The input text to chunk.
            doc_id: Optional document identifier.
            
        Returns:
            List of Chunk objects with parent_id relationships.
        """
        if not text.strip():
            return []

        all_chunks = []
        
        # Create document-level chunk
        doc_chunk_id = str(uuid.uuid4())
        doc_chunk = Chunk(
            content=text[:500] + "..." if len(text) > 500 else text,
            start_idx=0,
            end_idx=len(text),
            chunk_id=doc_chunk_id,
            parent_id=None,
            metadata={
                "chunker": self.name,
                "level": "document",
                "full_length": len(text),
            },
        )
        if doc_id:
            doc_chunk.metadata["doc_id"] = doc_id
        all_chunks.append(doc_chunk)

        # Detect and process sections
        sections = self._detect_sections(text)

        for section_title, section_content, section_start, section_end in sections:
            section_chunk_id = str(uuid.uuid4())
            
            # Create section chunk
            section_chunk = Chunk(
                content=section_content[:1000] if len(section_content) > 1000 else section_content,
                start_idx=section_start,
                end_idx=section_end,
                chunk_id=section_chunk_id,
                parent_id=doc_chunk_id,
                metadata={
                    "chunker": self.name,
                    "level": "section",
                    "title": section_title,
                },
            )
            if doc_id:
                section_chunk.metadata["doc_id"] = doc_id
            if self.include_parent_content:
                section_chunk.metadata["parent_summary"] = doc_chunk.content[:200]
            all_chunks.append(section_chunk)

            # Split section into paragraphs
            paragraphs = self._split_into_paragraphs(section_content)

            for para_text, para_start, para_end in paragraphs:
                para_chunk_id = str(uuid.uuid4())
                
                # Create paragraph chunk
                para_chunk = Chunk(
                    content=para_text,
                    start_idx=section_start + para_start,
                    end_idx=section_start + para_end,
                    chunk_id=para_chunk_id,
                    parent_id=section_chunk_id,
                    metadata={
                        "chunker": self.name,
                        "level": "paragraph",
                    },
                )
                if doc_id:
                    para_chunk.metadata["doc_id"] = doc_id
                if self.include_parent_content:
                    para_chunk.metadata["section_title"] = section_title
                    para_chunk.metadata["parent_summary"] = section_chunk.content[:200]
                all_chunks.append(para_chunk)

        return all_chunks

    def get_parent_chain(self, chunk: Chunk, all_chunks: list[Chunk]) -> list[Chunk]:
        """Get the full parent chain for a chunk.
        
        Args:
            chunk: The chunk to get parents for.
            all_chunks: All chunks from the same document.
            
        Returns:
            List of parent chunks from immediate parent to root.
        """
        chunk_map = {c.chunk_id: c for c in all_chunks}
        parents = []
        current = chunk

        while current.parent_id and current.parent_id in chunk_map:
            parent = chunk_map[current.parent_id]
            parents.append(parent)
            current = parent

        return parents

    def __repr__(self) -> str:
        level_names = [level.name for level in self.levels]
        return f"HierarchicalChunker(levels={level_names})"
