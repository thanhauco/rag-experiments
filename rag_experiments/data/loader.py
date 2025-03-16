"""Document loading utilities."""

import os
from typing import List, Optional


class DocumentLoader:
    """Utility to load documents from various formats."""

    @staticmethod
    def load_from_directory(
        directory: str,
        extensions: Optional[List[str]] = None
    ) -> List[str]:
        """Load text files from a directory.
        
        Args:
            directory: Path to the directory.
            extensions: List of file extensions to include (default: .txt, .md).
            
        Returns:
            List of document contents.
        """
        if extensions is None:
            extensions = [".txt", ".md"]
            
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            documents.append(f.read())
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        
        return documents

    @staticmethod
    def load_from_file(path: str) -> str:
        """Load a single document."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
