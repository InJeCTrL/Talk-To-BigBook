"""Configuration and logging setup for Layered Summary System"""

import logging
import sys
from typing import Dict, Any

# Logging configuration
def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "chunk_size": 2000,
    "overlap": 200,
    "window_size": 5,
    "stride": 3,
    "level2_budget": 5000,
    "level3_budget": 1000,
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:7b",
    "gemini_model": "gemini-2.0-flash-exp",
    "max_retries": 3,
    "retry_delays": [1, 2, 4]  # Exponential backoff in seconds
}

def get_config() -> Dict[str, Any]:
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()
