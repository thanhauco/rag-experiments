"""Configuration management for RAG experiments."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    SLIDING_WINDOW = "sliding_window"


class IndexType(str, Enum):
    """Available index types."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RetrievalMethod(str, Enum):
    """Available retrieval methods."""

    SIMPLE = "simple"
    RRF = "rrf"
    WEIGHTED = "weighted"
    QUERY_EXPANSION = "query_expansion"
    HYDE = "hyde"


class ChunkingConfig(BaseModel):
    """Configuration for chunking strategies."""

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = Field(default=512, ge=50, le=4096)
    overlap: int = Field(default=50, ge=0)
    max_sentences: int = Field(default=5, ge=1, le=20)
    separator: str = "\n\n"


class IndexingConfig(BaseModel):
    """Configuration for indexing."""

    index_type: IndexType = IndexType.HYBRID
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    use_stemming: bool = True
    remove_stopwords: bool = True


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""

    method: RetrievalMethod = RetrievalMethod.RRF
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    expand_queries: bool = False
    use_reranker: bool = False


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    metrics: list[str] = Field(default_factory=lambda: ["mrr", "ndcg@5", "recall@5"])
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    track_failures: bool = True
    save_predictions: bool = True


class ExperimentConfig(BaseModel):
    """Main experiment configuration."""

    name: str = "default_experiment"
    description: str = ""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    random_seed: int = 42
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
