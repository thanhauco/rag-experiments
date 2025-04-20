"""Unit tests for chunking strategies."""

import pytest
from rag_experiments.chunking import FixedSizeChunker, SemanticChunker, HierarchicalChunker, SlidingWindowChunker

SAMPLE_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines.
Leading AI textbooks define the field as the study of "intelligent agents".
This is a second paragraph with some more text to test chunking boundaries.
It has multiple sentences to make sure semantic chunking works as expected.
"""

def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) <= 50
        assert chunk.metadata["chunker"] == "fixed_size"

def test_semantic_chunker():
    chunker = SemanticChunker(max_sentences=2)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["sentence_count"] <= 2
        assert chunk.metadata["chunker"] == "semantic"

def test_hierarchical_chunker():
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) >= 3 # Document, Section (default), Paragraphs
    levels = [c.metadata["level"] for c in chunks]
    assert "document" in levels
    assert "paragraph" in levels

def test_sliding_window_chunker():
    chunker = SlidingWindowChunker(window_size=100, stride=50)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    assert chunks[0].metadata["chunker"] == "sliding_window"
