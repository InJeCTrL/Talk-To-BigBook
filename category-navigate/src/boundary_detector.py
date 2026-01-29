"""Boundary detection module for Category-Navigate System"""

import logging
import os
import requests
from typing import List, Tuple
from .models import Chunk
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """Detects topic boundaries between adjacent chunks using LLM."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        preview_chars: int = 500,
        max_retries: int = 3,
        provider: str = "ollama",
        api_key: str = None
    ):
        """
        Initialize boundary detector.
        
        Args:
            base_url: API base URL (Ollama or DashScope)
            model: Model name to use
            preview_chars: Number of characters to preview from each chunk
            max_retries: Maximum retry attempts for API calls
            provider: "ollama" or "dashscope"
            api_key: API key for DashScope (or set DASHSCOPE_API_KEY env var)
        """
        self.provider = provider
        self.model = model
        self.preview_chars = preview_chars
        self.max_retries = max_retries
        
        if provider == "dashscope":
            self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
            if not self.api_key:
                raise ValueError("DashScope API key required. Set DASHSCOPE_API_KEY env var or pass api_key.")
            self.base_url = base_url if "dashscope" in base_url else "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.api_url = f"{self.base_url}/chat/completions"
        else:
            self.base_url = base_url.rstrip('/')
            self.api_url = f"{self.base_url}/api/generate"
        
        logger.info(f"BoundaryDetector initialized: provider={provider}, model={model}, preview_chars={preview_chars}")
    
    def _create_boundary_prompt(self, chunk1_text: str, chunk2_text: str) -> str:
        """
        Create prompt for boundary detection.
        
        Args:
            chunk1_text: Text from first chunk
            chunk2_text: Text from second chunk
            
        Returns:
            Formatted prompt string
        """
        # Take preview from end of chunk1 and start of chunk2
        preview1 = chunk1_text[-self.preview_chars:] if len(chunk1_text) > self.preview_chars else chunk1_text
        preview2 = chunk2_text[:self.preview_chars] if len(chunk2_text) > self.preview_chars else chunk2_text
        
        prompt = f"""You are analyzing whether two adjacent text segments discuss the same topic or different topics.

Segment 1 (end):
{preview1}

Segment 2 (beginning):
{preview2}

Question: Are these two segments discussing the same topic or different topics?

Instructions:
- Answer ONLY with "SAME" if they discuss the same topic
- Answer ONLY with "DIFFERENT" if they discuss different topics
- Consider topic transitions, not just keyword overlap
- A topic change means the main subject or focus has shifted

Answer (SAME or DIFFERENT):"""
        
        return prompt
    
    @retry_with_backoff(max_retries=3)
    def _is_boundary(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """
        Check if there's a topic boundary between two chunks.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            True if boundary detected (different topics), False otherwise
        """
        prompt = self._create_boundary_prompt(chunk1.text, chunk2.text)
        
        try:
            if self.provider == "dashscope":
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 10,
                        "temperature": 0.1
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip().upper()
            else:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent responses
                            "num_predict": 10    # We only need one word
                        }
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                answer = result.get("response", "").strip().upper()
            
            # Parse response
            if "DIFFERENT" in answer:
                logger.debug(f"Boundary detected between {chunk1.id} and {chunk2.id}")
                return True
            elif "SAME" in answer:
                logger.debug(f"No boundary between {chunk1.id} and {chunk2.id}")
                return False
            else:
                logger.warning(f"Unexpected response: {answer}. Assuming no boundary.")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Boundary detection failed: {e}")
            raise
    
    def detect_boundaries(self, chunks: List[Chunk]) -> List[int]:
        """
        Detect all topic boundaries in a list of chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            List of boundary positions (indices where new topics start)
            
        Example:
            If boundaries are at positions 3 and 7, it means:
            - Chunks 0-2 are one topic
            - Chunks 3-6 are another topic
            - Chunks 7+ are another topic
        """
        if len(chunks) < 2:
            logger.info("Less than 2 chunks, no boundaries to detect")
            return []
        
        boundaries = []
        total = len(chunks) - 1
        
        logger.info(f"Detecting boundaries for {len(chunks)} chunks...")
        print(f"\n[Boundary Detection] Processing {total} chunk pairs...")
        
        # Check each adjacent pair
        for i in range(total):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Progress output
            print(f"\r[Boundary Detection] {i+1}/{total} ({(i+1)*100//total}%) - checking {chunk1.id} vs {chunk2.id}", end="", flush=True)
            
            if self._is_boundary(chunk1, chunk2):
                # Boundary detected: chunk2 starts a new topic
                boundaries.append(i + 1)
                print(f" -> BOUNDARY")
            else:
                print(f" -> same", end="")
        
        print(f"\n[Boundary Detection] Complete! Found {len(boundaries)} boundaries")
        logger.info(f"Detected {len(boundaries)} boundaries at positions: {boundaries}")
        return boundaries
