# RAG Experiments

A comprehensive Python framework for studying how retrieval quality impacts downstream LLM reasoning accuracy and failure rates.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

RAG systems are sensitive to retrieval quality. Poor chunking, weak indexing, or suboptimal retrieval fusion can cause:

- **Hallucinations** — LLM generates plausible but incorrect answers
- **Incomplete reasoning** — Missing context leads to partial conclusions
- **Failure cascades** — Early retrieval errors compound in multi-hop reasoning

This framework enables systematic experimentation to quantify these effects.

## Features

### 🧩 Chunking Strategies

| Strategy           | Description                                      |
| ------------------ | ------------------------------------------------ |
| **Fixed-Size**     | Character/token-based with configurable overlap  |
| **Semantic**       | Sentence/paragraph-based using NLP boundaries    |
| **Hierarchical**   | Parent-child relationships for context expansion |
| **Sliding Window** | Overlapping windows with sentence-aware snapping |

### 📚 Indexing Schemes

| Index             | Description                              |
| ----------------- | ---------------------------------------- |
| **Dense (FAISS)** | Vector similarity with IVF/HNSW options  |
| **Sparse (BM25)** | Lexical matching with stemming           |
| **Hybrid**        | Combined dense + sparse with late fusion |

### 🔍 Hybrid Retrieval

| Method              | Description                           |
| ------------------- | ------------------------------------- |
| **RRF**             | Reciprocal Rank Fusion (k=60)         |
| **Weighted**        | Linear combination with normalization |
| **Query Expansion** | Synonym + LLM-based rewriting         |
| **HyDE**            | Hypothetical Document Embeddings      |

### 📊 Evaluation

- **Retrieval Metrics**: MRR, NDCG@k, Recall@k, Precision@k
- **Reasoning Accuracy**: Exact match, fuzzy match, semantic similarity
- **Failure Analysis**: Categorized failure modes with export

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rag-experiments.git
cd rag-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from rag_experiments.chunking import FixedSizeChunker, SemanticChunker
from rag_experiments.indexing import DenseIndex, HybridIndex
from rag_experiments.retrieval import RRFRetriever
from rag_experiments.evaluation import ExperimentRunner

# Load documents
documents = ["Your document text here..."]

# Configure chunking
chunker = SemanticChunker(max_sentences=5)
chunks = chunker.chunk(documents[0])

# Build index
index = HybridIndex(dense_weight=0.7, sparse_weight=0.3)
index.add(chunks)

# Retrieve
retriever = RRFRetriever(index, k=60)
results = retriever.retrieve("What is the main topic?", top_k=5)

# Evaluate
runner = ExperimentRunner()
metrics = runner.evaluate(retriever, test_queries, ground_truth)
print(f"MRR: {metrics['mrr']:.3f}, NDCG@5: {metrics['ndcg@5']:.3f}")
```

## Running Experiments

```bash
# Run full experiment grid
python -m rag_experiments.evaluation.experiment_runner \
    --config configs/experiment_config.yaml

# Run specific chunking comparison
python -m rag_experiments.evaluation.experiment_runner \
    --chunkers fixed,semantic,hierarchical \
    --output results/chunking_comparison.json
```

## Project Structure

```
rag_experiments/
├── chunking/          # Chunking strategies
├── indexing/          # Index implementations
├── retrieval/         # Retrieval methods
├── evaluation/        # Metrics and experiment runner
├── data/              # Data loading and synthetic QA
└── utils/             # Embeddings and logging
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## License

MIT License - see [LICENSE](LICENSE) for details.
