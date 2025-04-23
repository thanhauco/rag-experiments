"""Unit tests for evaluation metrics."""

import pytest
from rag_experiments.evaluation.metrics import RetrievalEvaluator

def test_metrics_calculation():
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    
    retrieved_ids = ["c1", "c2", "c3", "c4", "c5"]
    ground_truth_ids = ["c1", "c3"]
    
    metrics = evaluator.evaluate(retrieved_ids, ground_truth_ids)
    
    # Hit at 1, so MRR should be 1.0
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@1"] == 1.0
    
    # Hit at 1, Recall@3 should be 0.5 (1 out of 2) or 1.0? 
    # Actually ground_truth_ids = ["c1", "c3"]. Both are in top 3.
    # So Recall@3 should be 1.0
    assert metrics["recall@3"] == 1.0
    
    # NDCG calculation
    assert metrics["ndcg@5"] > 0
    assert metrics["ndcg@5"] <= 1.0
