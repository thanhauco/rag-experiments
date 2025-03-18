"""Logging utilities for experiment tracking."""

import logging
import sys
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup rich logging for the framework.
    
    Args:
        level: Logging level.
        log_file: Optional path to save logs.
        
    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    logger = logging.getLogger("rag_experiments")
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        
    return logger
