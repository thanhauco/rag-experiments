"""Benchmarking utilities for RAG experiments."""

from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    name: str
    component: str
    latency_ms: float
    throughput: float  # queries per second
    memory_mb: float
    metrics: Dict[str, float]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": self.timestamp,
        }


class LatencyTracker:
    """Tracks latency across multiple operations."""

    def __init__(self):
        self.measurements: List[float] = []
        self._start_time: float | None = None

    def start(self):
        """Start timing."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and record measurement."""
        if self._start_time is None:
            return 0.0
        elapsed = (time.perf_counter() - self._start_time) * 1000  # ms
        self.measurements.append(elapsed)
        self._start_time = None
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    @property
    def mean(self) -> float:
        return sum(self.measurements) / len(self.measurements) if self.measurements else 0

    @property
    def p50(self) -> float:
        if not self.measurements:
            return 0
        sorted_m = sorted(self.measurements)
        return sorted_m[len(sorted_m) // 2]

    @property
    def p95(self) -> float:
        if not self.measurements:
            return 0
        sorted_m = sorted(self.measurements)
        idx = int(len(sorted_m) * 0.95)
        return sorted_m[min(idx, len(sorted_m) - 1)]

    @property
    def p99(self) -> float:
        if not self.measurements:
            return 0
        sorted_m = sorted(self.measurements)
        idx = int(len(sorted_m) * 0.99)
        return sorted_m[min(idx, len(sorted_m) - 1)]


class RAGBenchmark:
    """Comprehensive benchmark suite for RAG components.
    
    Measures latency, throughput, and memory usage across:
    - Chunking strategies
    - Indexing operations  
    - Retrieval methods
    - End-to-end pipeline
    """

    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def benchmark_chunking(
        self,
        chunker,
        documents: List[str],
        name: str = "chunking",
    ) -> BenchmarkResult:
        """Benchmark a chunking strategy."""
        tracker = LatencyTracker()
        
        for doc in documents:
            with tracker:
                chunker.chunk(doc)

        result = BenchmarkResult(
            name=name,
            component="chunking",
            latency_ms=tracker.mean,
            throughput=1000 / tracker.mean if tracker.mean > 0 else 0,
            memory_mb=0,  # Would need psutil for accurate measurement
            metrics={
                "p50_ms": tracker.p50,
                "p95_ms": tracker.p95,
                "p99_ms": tracker.p99,
                "total_docs": len(documents),
            },
            config={"chunker": chunker.name if hasattr(chunker, 'name') else str(chunker)},
        )
        
        self.results.append(result)
        return result

    def benchmark_retrieval(
        self,
        retriever,
        queries: List[str],
        name: str = "retrieval",
    ) -> BenchmarkResult:
        """Benchmark a retrieval method."""
        tracker = LatencyTracker()
        
        for query in queries:
            with tracker:
                retriever.retrieve(query, top_k=5)

        result = BenchmarkResult(
            name=name,
            component="retrieval",
            latency_ms=tracker.mean,
            throughput=1000 / tracker.mean if tracker.mean > 0 else 0,
            memory_mb=0,
            metrics={
                "p50_ms": tracker.p50,
                "p95_ms": tracker.p95,
                "p99_ms": tracker.p99,
                "total_queries": len(queries),
            },
            config={"retriever": retriever.name if hasattr(retriever, 'name') else str(retriever)},
        )
        
        self.results.append(result)
        return result

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save all results to JSON."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        return path

    def summary(self) -> str:
        """Generate summary of benchmark results."""
        lines = ["RAG Benchmark Summary", "=" * 40]
        
        for r in self.results:
            lines.append(f"\n{r.name} ({r.component})")
            lines.append(f"  Latency: {r.latency_ms:.2f}ms (p95: {r.metrics.get('p95_ms', 0):.2f}ms)")
            lines.append(f"  Throughput: {r.throughput:.1f} ops/sec")
            
        return "\n".join(lines)
