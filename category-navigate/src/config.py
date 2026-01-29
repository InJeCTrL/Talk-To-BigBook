"""Configuration for Category-Navigate System"""

import logging
import sys
from typing import Dict, Any

# Logging configuration
def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "chunk_size": 2000,
    "overlap": 200,
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:7b",
    "gemini_model": "gemini-2.0-flash-exp",
    "boundary_preview_chars": 500,
    "hierarchy_threshold": 10,
    "max_title_length": 15,
    "summary_min_tokens": 100,
    "summary_max_tokens": 200,
    "max_retries": 3,
    "retry_delays": [1, 2, 4]
}

def get_config() -> Dict[str, Any]:
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()
