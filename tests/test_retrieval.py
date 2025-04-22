"""Unit tests for retrieval methods."""

import pytest
from unittest.mock import MagicMock
from rag_experiments.chunking.base import Chunk
from rag_experiments.indexing.base import SearchResult
from rag_experiments.retrieval import RRFRetriever, WeightedRetriever

@pytest.fixture
def mock_index():
    index = MagicMock()
    index.name = "mock_index"
    
    chunk = Chunk(content="test content", start_idx=0, end_idx=12, chunk_id="c1")
    result = SearchResult(chunk=chunk, score=0.9, rank=1)
    
    index.search.return_value = [result]
    return index

def test_rrf_retriever_fallback(mock_index):
    retriever = RRFRetriever(mock_index)
    results = retriever.retrieve("test query")
    assert len(results) == 1
    assert results[0].chunk.content == "test content"

def test_weighted_retriever_fallback(mock_index):
    retriever = WeightedRetriever(mock_index)
    results = retriever.retrieve("test query")
    assert len(results) == 1
    assert results[0].chunk.content == "test content"
