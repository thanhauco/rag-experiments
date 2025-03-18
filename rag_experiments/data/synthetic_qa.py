"""Document loading and synthetic QA generation utilities."""

from __future__ import annotations

import uuid
from typing import Any


class SyntheticQAGenerator:
    """Generates synthetic QA pairs from chunks for experimentation.
    
    In a real system, this would use an LLM. Here we use heuristics
    to create test cases where we know the ground truth chunks.
    """

    def generate(self, chunks: list[Any], num_questions: int = 10) -> list[dict[str, Any]]:
        """Generate QA pairs from a list of chunks."""
        dataset = []
        
        # Pick relevant chunks
        import random
        random.seed(42)
        
        selected_chunks = random.sample(chunks, min(num_questions, len(chunks)))
        
        for i, chunk in enumerate(selected_chunks):
            # Extract a "query" from the chunk (first sentence or title)
            content = chunk.content
            first_sentence = content.split(".")[0]
            
            # Simple "question" generation
            if "?" in first_sentence:
                query = first_sentence
            else:
                query = f"What does the text say about '{first_sentence[:50]}...'?"
                
            dataset.append({
                "id": f"q_{i}",
                "query": query,
                "ground_truth_ids": [chunk.chunk_id],
                "ideal_answer": content  # Use the chunk itself as the ideal answer
            })
            
        return dataset


def load_sample_documents() -> list[str]:
    """Return a few sample documents for testing."""
    return [
        """
        # Artificial Intelligence: A Deep Dive
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals.
        
        ## History of AI
        The field was founded on the claim that human intelligence "can be so 
        precisely described that a machine can be made to simulate it." 
        This raised philosophical arguments about the mind and the ethical 
        responsibilities of creating artificial beings.
        """,
        """
        # Retrieval-Augmented Generation (RAG)
        Retrieval-Augmented Generation (RAG) is a technique for enhancing 
        the accuracy and reliability of generative AI models with facts 
        fetched from external sources. 
        
        ## Why RAG?
        LLMs are trained on vast datasets but they are not updated in real-time.
        RAG allows models to access the latest information without retraining.
        This reduces hallucinations and provides better domain-specific answers.
        """
    ]
