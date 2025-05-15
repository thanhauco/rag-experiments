"""Performance profiler for detailed RAG pipeline analysis."""

from __future__ import annotations

import time
import functools
from typing import Dict, List, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ProfileEntry:
    """Single profiling entry."""
    
    name: str
    duration_ms: float
    calls: int = 1
    children: List["ProfileEntry"] = field(default_factory=list)


class Profiler:
    """Hierarchical profiler for RAG pipeline stages.
    
    Tracks time spent in each stage of the pipeline:
    - Document loading
    - Chunking
    - Embedding
    - Indexing
    - Retrieval
    - Reranking
    - Context compression
    """

    def __init__(self):
        self.entries: Dict[str, ProfileEntry] = {}
        self._stack: List[str] = []
        self._timers: Dict[str, float] = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block.
        
        Usage:
            with profiler.profile("retrieval"):
                results = retriever.retrieve(query)
        """
        self._stack.append(name)
        start = time.perf_counter()
        
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000
            self._record(name, duration)
            self._stack.pop()

    def _record(self, name: str, duration_ms: float):
        """Record a profiling measurement."""
        if name in self.entries:
            entry = self.entries[name]
            # Running average
            entry.duration_ms = (
                entry.duration_ms * entry.calls + duration_ms
            ) / (entry.calls + 1)
            entry.calls += 1
        else:
            self.entries[name] = ProfileEntry(name=name, duration_ms=duration_ms)

    def profile_func(self, name: str | None = None):
        """Decorator for profiling functions.
        
        Usage:
            @profiler.profile_func("my_function")
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(func_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def summary(self) -> str:
        """Generate profiling summary."""
        if not self.entries:
            return "No profiling data collected."

        lines = ["Pipeline Profile Summary", "=" * 50]
        
        # Sort by duration (slowest first)
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.duration_ms,
            reverse=True
        )
        
        total_time = sum(e.duration_ms for e in sorted_entries)
        
        for entry in sorted_entries:
            pct = (entry.duration_ms / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(
                f"{entry.name:20} {entry.duration_ms:8.2f}ms "
                f"({entry.calls:3} calls) [{bar}] {pct:5.1f}%"
            )

        lines.append("-" * 50)
        lines.append(f"{'TOTAL':20} {total_time:8.2f}ms")
        
        return "\n".join(lines)

    def reset(self):
        """Clear all profiling data."""
        self.entries.clear()
        self._stack.clear()
        self._timers.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export profiling data as dictionary."""
        return {
            name: {
                "duration_ms": entry.duration_ms,
                "calls": entry.calls,
            }
            for name, entry in self.entries.items()
        }


# Global profiler instance
_global_profiler: Profiler | None = None


def get_profiler() -> Profiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def profile(name: str):
    """Convenience decorator using global profiler."""
    return get_profiler().profile_func(name)
