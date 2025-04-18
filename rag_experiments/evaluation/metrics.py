"""Evaluation metrics for RAG retrieval."""

from __future__ import annotations

import math
from typing import Set


def calculate_hit_rate(retrieved_ids: list[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate hit rate at k.
    
    1 if at least one ground truth document is in top k, else 0.
    """
    top_k = retrieved_ids[:k]
    for gid in ground_truth_ids:
        if gid in top_k:
            return 1.0
    return 0.0


def calculate_mrr(retrieved_ids: list[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank at k."""
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall(retrieved_ids: list[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate Recall at k."""
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(ground_truth_ids))
    return hits / len(ground_truth_ids)


def calculate_precision(retrieved_ids: list[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate Precision at k."""
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(ground_truth_ids))
    return hits / k


def calculate_ndcg(retrieved_ids: list[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate NDCG at k."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in ground_truth_ids:
            dcg += 1.0 / math.log2(i + 2)
            
    # IDCG (best possible ranking)
    idcg = 0.0
    num_relevant = min(len(ground_truth_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """Evaluates retrieval quality using multiple metrics."""

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(self, retrieved_ids: list[str], ground_truth_ids: list[str]) -> dict[str, float]:
        """Compute metrics for a single prediction.
        
        Args:
            retrieved_ids: IDs of retrieved chunks in order.
            ground_truth_ids: IDs of actually relevant chunks.
            
        Returns:
            Dictionary of metric names and values.
        """
        gt_set = set(ground_truth_ids)
        metrics = {}
        
        for k in self.k_values:
            metrics[f"hit_rate@{k}"] = calculate_hit_rate(retrieved_ids, gt_set, k)
            metrics[f"mrr@{k}"] = calculate_mrr(retrieved_ids, gt_set, k)
            metrics[f"recall@{k}"] = calculate_recall(retrieved_ids, gt_set, k)
            metrics[f"ndcg@{k}"] = calculate_ndcg(retrieved_ids, gt_set, k)
            
        return metrics
