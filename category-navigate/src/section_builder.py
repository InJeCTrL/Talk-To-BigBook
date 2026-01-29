"""Section building module for Category-Navigate System"""

import logging
import os
import requests
from typing import List
from .models import Chunk, Section
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


class SectionBuilder:
    """Builds sections from chunks and generates titles (no summaries - pure directory style)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        max_title_length: int = 15,
        max_retries: int = 3,
        provider: str = "ollama",
        api_key: str = None
    ):
        """
        Initialize section builder.
        
        Args:
            base_url: API base URL (Ollama or DashScope)
            model: Model name to use
            max_title_length: Maximum characters for section title
            max_retries: Maximum retry attempts for API calls
            provider: "ollama" or "dashscope"
            api_key: API key for DashScope (or set DASHSCOPE_API_KEY env var)
        """
        self.provider = provider
        self.model = model
        self.max_title_length = max_title_length
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
        
        logger.info(f"SectionBuilder initialized: provider={provider}, model={model}, max_title={max_title_length}")
    
    def build_sections(
        self,
        chunks: List[Chunk],
        boundaries: List[int]
    ) -> List[Section]:
        """
        Build sections from chunks using boundary positions.
        
        Args:
            chunks: List of all chunks
            boundaries: List of boundary positions (indices where new sections start)
            
        Returns:
            List of Section objects with titles (no summaries - pure directory style)
        """
        if not chunks:
            return []
        
        # Create section ranges
        section_ranges = []
        start = 0
        
        for boundary in boundaries:
            section_ranges.append((start, boundary))
            start = boundary
        
        # Add final section
        section_ranges.append((start, len(chunks)))
        
        total = len(section_ranges)
        logger.info(f"Building {total} sections...")
        print(f"\n[Section Building] Creating {total} sections (title only, no summary)...")
        
        # Build each section
        sections = []
        for section_id, (start_idx, end_idx) in enumerate(section_ranges):
            section_chunks = chunks[start_idx:end_idx]
            
            print(f"\r[Section Building] {section_id+1}/{total} ({(section_id+1)*100//total}%) - generating title...", end="", flush=True)
            
            # Generate title only (no summary)
            title = self._generate_title(section_chunks)
            
            # Get position range
            start_pos = section_chunks[0].start_pos
            end_pos = section_chunks[-1].end_pos
            
            section = Section(
                id=f"section_{section_id}",
                title=title,
                chunk_ids=[chunk.id for chunk in section_chunks],
                start_pos=start_pos,
                end_pos=end_pos
            )
            
            sections.append(section)
            print(f" -> '{title}' ({len(section_chunks)} chunks)")
            logger.info(f"Created {section.id}: '{title}' ({len(section_chunks)} chunks)")
        
        print(f"[Section Building] Complete! Created {len(sections)} sections")
        return sections
    
    @retry_with_backoff(max_retries=3)
    def _generate_title(self, chunks: List[Chunk]) -> str:
        """
        Generate a short title for a section.
        
        Args:
            chunks: Chunks in the section
            
        Returns:
            Title string (max 15 characters)
        """
        # Combine chunk texts
        combined_text = "\n\n".join(chunk.text for chunk in chunks)
        
        # Limit text length for prompt (use first 2000 chars)
        if len(combined_text) > 2000:
            combined_text = combined_text[:2000] + "..."
        
        prompt = f"""Generate a very short title (maximum {self.max_title_length} characters) for the following text section.

Text:
{combined_text}

Instructions:
- Title must be {self.max_title_length} characters or less
- Capture the main topic or theme
- Be concise and clear
- Use Chinese if the text is Chinese, English if English

Title:"""
        
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
                        "max_tokens": 30,
                        "temperature": 0.3
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                title = result["choices"][0]["message"]["content"].strip()
            else:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 30
                        }
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                title = result.get("response", "").strip()
            
            # Truncate if too long
            if len(title) > self.max_title_length:
                title = title[:self.max_title_length]
            
            # Remove quotes if present
            title = title.strip('"\'')
            
            return title if title else "Untitled"
            
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            return "Untitled"
