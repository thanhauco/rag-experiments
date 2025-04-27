"""Example script to run RAG experiments."""

from rag_experiments.chunking import FixedSizeChunker, SemanticChunker
from rag_experiments.data.synthetic_qa import load_sample_documents, SyntheticQAGenerator
from rag_experiments.evaluation.experiment_runner import ExperimentRunner
from rag_experiments.indexing import HybridIndex
from rag_experiments.retrieval import RRFRetriever


def main():
    print("🚀 Starting RAG Experiment Smoke Test...")
    
    # 1. Load Data
    documents = load_sample_documents()
    print(f"Loaded {len(documents)} documents.")
    
    # 2. Chunking
    # Try different chunkers
    fixed_chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    semantic_chunker = SemanticChunker(max_sentences=2)
    
    fixed_chunks = fixed_chunker.chunk_documents(documents)
    semantic_chunks = semantic_chunker.chunk_documents(documents)
    print(f"Generated {len(fixed_chunks)} fixed chunks and {len(semantic_chunks)} semantic chunks.")
    
    # 3. Indexing & Retrieval
    # We'll use semantic chunks for this test
    index = HybridIndex(dense_weight=0.7, sparse_weight=0.3)
    index.add(semantic_chunks)
    print(f"Indexed {index.size} chunks into HybridIndex.")
    
    retriever = RRFRetriever(index, k=60)
    
    # 4. Generate Synthetic Test Data
    qa_gen = SyntheticQAGenerator()
    test_data = qa_gen.generate(semantic_chunks, num_questions=5)
    print(f"Generated {len(test_data)} synthetic questions.")
    
    # 5. Run Experiment
    runner = ExperimentRunner(output_dir="outputs/smoke_test")
    result = runner.run(retriever, test_data)
    
    print("\n📊 Experiment Results:")
    print(f"Retriever: {retriever.name}")
    for metric, value in result.metrics.items():
        if "@5" in metric:
            print(f"  {metric}: {value:.4f}")
            
    print("\n❌ Failure Analysis:")
    for mode, count in result.failures.items():
        print(f"  {mode}: {count}")
        
    print("\n✅ Smoke test complete! Results saved to outputs/smoke_test/")


if __name__ == "__main__":
    main()
