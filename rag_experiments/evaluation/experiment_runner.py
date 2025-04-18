"""Main experiment runner for RAG evaluation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rag_experiments.config import ExperimentConfig
from rag_experiments.evaluation.metrics import RetrievalEvaluator
from rag_experiments.evaluation.reasoning import FailureAnalyzer, ReasoningScorer
from rag_experiments.retrieval.base import Retriever


@dataclass
class ExperimentResult:
    """Results of a single experiment run."""
    
    config: Dict[str, Any]
    metrics: Dict[str, float]
    failures: Dict[str, int]
    runtime_sec: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


class ExperimentRunner:
    """Orchestrates RAG experiments across different configurations."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.retrieval_evaluator = RetrievalEvaluator()
        self.reasoning_scorer = ReasoningScorer()
        self.failure_analyzer = FailureAnalyzer()

    def run(
        self,
        retriever: Retriever,
        test_data: List[Dict[str, Any]],
        config: ExperimentConfig | None = None
    ) -> ExperimentResult:
        """Run an experiment with a specific retriever.
        
        Args:
            retriever: The retriever strategy to test.
            test_data: List of dicts with 'query', 'ground_truth_ids', and optional 'ideal_answer'.
            config: Full experiment configuration.
            
        Returns:
            ExperimentResult object.
        """
        start_time = time.time()
        
        all_metrics: List[Dict[str, float]] = []
        failure_counts: Dict[str, int] = {}
        
        for item in test_data:
            query = item["query"]
            gt_ids = item["ground_truth_ids"]
            
            # 1. Retrieve
            results = retriever.retrieve(query, top_k=10)
            retrieved_ids = [r.chunk.chunk_id or r.chunk.content for r in results]
            
            # 2. Evaluate Retrieval
            metrics = self.retrieval_evaluator.evaluate(retrieved_ids, gt_ids)
            all_metrics.append(metrics)
            
            # 3. Analyze Failure
            # Assume reasoning success depends on retrieval for this experiment
            reasoning_success = metrics["ndcg@5"] > 0.5
            failure_mode = self.failure_analyzer.analyze(
                results, gt_ids, reasoning_success, k=5
            )
            failure_counts[failure_mode] = failure_counts.get(failure_mode, 0) + 1
            
        # Aggregate metrics
        agg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                agg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        runtime = time.time() - start_time
        
        result = ExperimentResult(
            config=config.to_dict() if config else {"retriever": retriever.name},
            metrics=agg_metrics,
            failures=failure_counts,
            runtime_sec=runtime
        )
        
        # Save results
        filename = f"result_{retriever.name}_{int(time.time())}.json"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            f.write(result.to_json())
            
        return result
