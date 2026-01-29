"""TOC building module for Category-Navigate System"""

import logging
import os
import requests
import json
from typing import List, Optional
from .models import Section, Part, TOC
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


class TOCBuilder:
    """Builds table of contents with recursive hierarchical grouping."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        hierarchy_threshold: int = 10,
        max_retries: int = 3,
        provider: str = "ollama",
        api_key: str = None
    ):
        """
        Initialize TOC builder.
        
        Args:
            base_url: API base URL (Ollama or DashScope)
            model: Model name to use
            hierarchy_threshold: Maximum items to show at each level (triggers hierarchy if exceeded)
            max_retries: Maximum retry attempts for API calls
            provider: "ollama" or "dashscope"
            api_key: API key for DashScope (or set DASHSCOPE_API_KEY env var)
        """
        self.provider = provider
        self.model = model
        self.max_items_per_level = hierarchy_threshold
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
        
        logger.info(f"TOCBuilder initialized: provider={provider}, model={model}, max_items_per_level={hierarchy_threshold}")
    
    def build_toc(self, sections: List[Section]) -> TOC:
        """
        Build table of contents from sections with recursive hierarchy.
        
        Args:
            sections: List of Section objects
            
        Returns:
            TOC object with appropriate hierarchy depth
        """
        if not sections:
            return TOC(parts=[], is_hierarchical=False)
        
        logger.info(f"Building TOC for {len(sections)} sections...")
        
        # Check if we need hierarchy
        if len(sections) <= self.max_items_per_level:
            # Flat TOC: each section is a part
            logger.info("Creating flat TOC (≤ threshold)")
            parts = self._create_flat_toc(sections)
            return TOC(parts=parts, is_hierarchical=False, levels=1)
        else:
            # Recursive hierarchical TOC
            logger.info("Creating hierarchical TOC (> threshold)")
            parts, levels = self._create_recursive_toc(sections)
            return TOC(parts=parts, is_hierarchical=True, levels=levels)
    
    def _create_flat_toc(self, sections: List[Section]) -> List[Part]:
        """Create flat TOC where each section is a part."""
        parts = []
        for i, section in enumerate(sections):
            part = Part(
                id=f"part_{i}",
                title=section.title,
                section_ids=[section.id],
                children=[]
            )
            parts.append(part)
        
        logger.info(f"Created flat TOC with {len(parts)} parts")
        return parts
    
    def _create_recursive_toc(self, sections: List[Section], level: int = 1) -> tuple:
        """
        Recursively create hierarchical TOC.
        
        Args:
            sections: List of sections to group
            level: Current hierarchy level
            
        Returns:
            Tuple of (parts list, total levels)
        """
        # Group sections into parts (target: 3-5 sections per part)
        group_size = max(3, min(5, len(sections) // self.max_items_per_level + 1))
        parts = self._group_into_parts(sections, group_size, level)
        
        logger.info(f"Level {level}: Created {len(parts)} parts from {len(sections)} sections")
        
        # Check if we need another level
        if len(parts) > self.max_items_per_level:
            # Recursively group parts
            logger.info(f"Level {level} has {len(parts)} parts, creating higher level...")
            higher_parts, total_levels = self._create_recursive_toc_from_parts(parts, level + 1)
            
            # Link higher parts to lower parts
            for hp in higher_parts:
                hp.children = [p.id for p in parts if p.id in hp.children]
            
            return higher_parts, total_levels
        else:
            return parts, level
    
    def _create_recursive_toc_from_parts(self, parts: List[Part], level: int) -> tuple:
        """
        Recursively group parts into higher-level parts.
        
        Args:
            parts: List of parts to group
            level: Current hierarchy level
            
        Returns:
            Tuple of (higher parts list, total levels)
        """
        # Group parts into super-parts
        group_size = max(3, min(5, len(parts) // self.max_items_per_level + 1))
        super_parts = []
        
        for i in range(0, len(parts), group_size):
            group = parts[i:i + group_size]
            
            # Create super-part
            super_part = Part(
                id=f"level{level}_part_{len(super_parts)}",
                title=self._generate_group_title(group),
                section_ids=[],  # Super-parts don't directly contain sections
                children=[p.id for p in group]
            )
            
            # Collect all section_ids from child parts
            for p in group:
                super_part.section_ids.extend(p.section_ids)
            
            super_parts.append(super_part)
        
        logger.info(f"Level {level}: Created {len(super_parts)} super-parts from {len(parts)} parts")
        
        # Check if we need another level
        if len(super_parts) > self.max_items_per_level:
            return self._create_recursive_toc_from_parts(super_parts, level + 1)
        else:
            return super_parts, level
    
    def _group_into_parts(self, sections: List[Section], group_size: int, level: int) -> List[Part]:
        """Group sections into parts."""
        parts = []
        
        for i in range(0, len(sections), group_size):
            group = sections[i:i + group_size]
            
            # Create part title from first section or generate
            title = group[0].title if len(group) == 1 else self._generate_group_title_from_sections(group)
            
            part = Part(
                id=f"level{level}_part_{len(parts)}",
                title=title,
                section_ids=[s.id for s in group],
                children=[]
            )
            parts.append(part)
        
        return parts
    
    def _generate_group_title(self, parts: List[Part]) -> str:
        """Generate title for a group of parts."""
        # Simple: use first part's title
        return parts[0].title if parts else "Untitled"
    
    def _generate_group_title_from_sections(self, sections: List[Section]) -> str:
        """Generate title for a group of sections."""
        return sections[0].title if sections else "Untitled"
    
    @retry_with_backoff(max_retries=3)
    def _create_hierarchical_toc_with_llm(self, sections: List[Section]) -> List[Part]:
        """
        Create hierarchical TOC by grouping sections into parts using LLM.
        (Kept for optional LLM-based grouping)
        """
        sections_text = self._format_sections_for_grouping(sections)
        
        prompt = f"""You are organizing sections of a document into logical groups (parts).

Sections:
{sections_text}

Instructions:
- Group related sections together into parts
- Each part should have 2-5 sections
- Create a short title for each part (max 20 characters)
- Create a brief summary for each part (1-2 sentences)
- Output ONLY valid JSON in this exact format:

{{
  "parts": [
    {{
      "title": "Part title",
      "summary": "Part summary",
      "section_ids": ["section_0", "section_1"]
    }}
  ]
}}

JSON output:"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "").strip()
            
            parts = self._parse_grouping_response(response_text, sections)
            return parts
            
        except Exception as e:
            logger.error(f"LLM-based TOC creation failed: {e}")
            raise
    
    def _format_sections_for_grouping(self, sections: List[Section]) -> str:
        """Format sections for LLM grouping prompt."""
        lines = []
        for section in sections:
            lines.append(f"- {section.id}: {section.title}")
            lines.append(f"  Summary: {section.summary[:100]}...")
        return "\n".join(lines)
    
    def _parse_grouping_response(self, response_text: str, sections: List[Section]) -> List[Part]:
        """Parse LLM response to extract part groupings."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[json_start:json_end]
            data = json.loads(json_text)
            
            parts = []
            section_id_map = {s.id: s for s in sections}
            
            for i, part_data in enumerate(data.get("parts", [])):
                title = part_data.get("title", f"Part {i+1}")
                summary = part_data.get("summary", "")
                section_ids = part_data.get("section_ids", [])
                
                valid_section_ids = [sid for sid in section_ids if sid in section_id_map]
                
                if not valid_section_ids:
                    continue
                
                part = Part(
                    id=f"part_{i}",
                    title=title[:20],
                    summary=summary,
                    section_ids=valid_section_ids,
                    children=[]
                )
                parts.append(part)
            
            return parts
            
        except Exception as e:
            logger.error(f"Failed to parse grouping response: {e}")
            raise
