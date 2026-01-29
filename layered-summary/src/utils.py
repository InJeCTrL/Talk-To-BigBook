"""Utility functions for Layered Summary System"""

import logging
import time
from typing import Callable, Any, Dict
from functools import wraps

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding name
        
    Returns:
        Number of tokens
        
    Falls back to character-based estimation if tiktoken fails.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed, using character-based estimation: {e}")
        # Fallback: 1 token ≈ 4 characters
        return len(text) // 4


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Chinese or English.
    
    Args:
        text: Text to analyze
        
    Returns:
        "zh" if more than 50% Chinese characters, else "en"
    """
    if not text:
        return "en"
    
    # Count Chinese characters (Unicode range U+4E00 to U+9FFF)
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text)
    
    if total_chars == 0:
        return "en"
    
    chinese_ratio = chinese_chars / total_chars
    return "zh" if chinese_ratio > 0.5 else "en"


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    chunk_size = config.get("chunk_size", 2000)
    overlap = config.get("overlap", 200)
    window_size = config.get("window_size", 5)
    stride = config.get("stride", 3)
    
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
    
    if window_size <= stride:
        raise ValueError(f"window_size ({window_size}) must be greater than stride ({stride})")
    
    # Check all parameters are positive
    for key in ["chunk_size", "overlap", "window_size", "stride", "level2_budget", "level3_budget"]:
        value = config.get(key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive, got {value}")
    
    return True


def retry_with_backoff(max_retries: int = 3, delays: list = None) -> Callable:
    """
    Decorator to retry function with exponential backoff on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delays: List of delay times in seconds for each retry
        
    Returns:
        Decorated function
    """
    if delays is None:
        delays = [1, 2, 4]  # Default exponential backoff
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = delays[min(attempt, len(delays) - 1)]
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    validate_config(config)
    return config
