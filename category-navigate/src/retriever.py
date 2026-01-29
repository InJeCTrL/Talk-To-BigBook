"""Query and retrieval module for Category-Navigate System"""

import logging
import json
import requests
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import DocumentIndex, IndexMetadata, Chunk, Section, TOC, Part
from .config import DEFAULT_CONFIG
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


class CategoryRetriever:
    """Retrieves relevant content from indexed documents."""
    
    def __init__(
        self,
        provider: str = "gemini",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize retriever with LLM configuration.
        
        Args:
            provider: LLM provider ("gemini" or "ollama")
            config: Configuration dictionary
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.provider = provider.lower()
        
        if self.provider == "gemini":
            self._setup_gemini()
        elif self.provider == "ollama":
            self._setup_ollama()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"CategoryRetriever initialized with provider: {provider}")
    
    def _setup_gemini(self) -> None:
        """Set up Gemini API client."""
        try:
            from google import genai
            from google.genai import types
            
            self.client = genai.Client()
            self.model_name = self.config["gemini_model"]
            self.genai_types = types
            
            logger.info(f"Gemini client initialized: {self.model_name}")
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
    
    def _setup_ollama(self) -> None:
        """Set up Ollama API configuration."""
        self.base_url = self.config["ollama_base_url"].rstrip('/')
        self.model_name = self.config["ollama_model"]
        
        logger.info(f"Ollama configured: {self.base_url}/{self.model_name}")
    
    def load_index(self, index_path: str) -> DocumentIndex:
        """
        Load index from JSON file.
        
        Args:
            index_path: Path to index JSON file
            
        Returns:
            DocumentIndex object
        """
        logger.info(f"Loading index from: {index_path}")
        
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Deserialize metadata
        metadata = IndexMetadata(
            document_path=data["metadata"]["document_path"],
            created_at=data["metadata"]["created_at"],
            model=data["metadata"]["model"],
            chunk_count=data["metadata"]["chunk_count"],
            section_count=data["metadata"]["section_count"],
            part_count=data["metadata"]["part_count"]
        )
        
        # Deserialize chunks
        chunks = [
            Chunk(
                id=c["id"],
                text=c["text"],
                token_count=c["token_count"],
                start_pos=c["start_pos"],
                end_pos=c["end_pos"]
            )
            for c in data["chunks"]
        ]
        
        # Deserialize sections
        sections = [
            Section(
                id=s["id"],
                title=s["title"],
                chunk_ids=s["chunk_ids"],
                start_pos=s["start_pos"],
                end_pos=s["end_pos"]
            )
            for s in data["sections"]
        ]
        
        # Deserialize TOC
        parts = [
            Part(
                id=p["id"],
                title=p["title"],
                section_ids=p["section_ids"]
            )
            for p in data["toc"]["parts"]
        ]
        
        toc = TOC(
            parts=parts,
            is_hierarchical=data["toc"]["is_hierarchical"]
        )
        
        index = DocumentIndex(
            metadata=metadata,
            chunks=chunks,
            sections=sections,
            toc=toc
        )
        
        logger.info(f"Index loaded: {len(chunks)} chunks, {len(sections)} sections, {len(parts)} parts")
        return index
    
    def query(
        self,
        index: DocumentIndex,
        question: str,
        expand_context: bool = True
    ) -> str:
        """
        Query the indexed document.
        
        Args:
            index: DocumentIndex to query
            question: User question
            expand_context: Whether to include adjacent sections
            
        Returns:
            Answer string
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Select relevant sections
        selected_section_ids = self._select_sections(index, question)
        logger.info(f"Selected {len(selected_section_ids)} sections")
        
        # Step 2: Optionally expand context
        if expand_context:
            selected_section_ids = self._expand_context(index, selected_section_ids)
            logger.info(f"Expanded to {len(selected_section_ids)} sections")
        
        # Step 3: Retrieve chunks
        chunks = self._retrieve_chunks(index, selected_section_ids)
        logger.info(f"Retrieved {len(chunks)} chunks")
        
        # Step 4: Generate answer
        answer = self._generate_answer(chunks, question)
        
        return answer
    
    @retry_with_backoff(max_retries=3)
    def _select_sections(
        self,
        index: DocumentIndex,
        question: str
    ) -> List[str]:
        """
        Select relevant sections using LLM.
        
        Args:
            index: DocumentIndex
            question: User question
            
        Returns:
            List of selected section IDs
        """
        # Format TOC for prompt
        toc_text = self._format_toc(index)
        
        prompt = f"""You are helping to find relevant sections in a document to answer a question.

Question: {question}

Table of Contents:
{toc_text}

Instructions:
- Select the section IDs that are most relevant to answering the question
- Return ONLY the section IDs as a JSON array
- Example: ["section_0", "section_3", "section_5"]

Selected section IDs:"""
        
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            response_text = response.text.strip()
        else:  # ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                },
                timeout=30
            )
            response.raise_for_status()
            response_text = response.json().get("response", "").strip()
        
        # Parse section IDs
        section_ids = self._parse_section_ids(response_text, index)
        return section_ids
    
    def _format_toc(self, index: DocumentIndex) -> str:
        """
        Format TOC for LLM prompt (title only, no summary - pure directory style).
        
        Args:
            index: DocumentIndex
            
        Returns:
            Formatted TOC string
        """
        lines = []
        
        if index.toc.is_hierarchical:
            # Hierarchical format
            for part in index.toc.parts:
                lines.append(f"\n{part.title}")
                
                # Get sections in this part
                for section_id in part.section_ids:
                    section = next((s for s in index.sections if s.id == section_id), None)
                    if section:
                        lines.append(f"  - [{section.id}] {section.title}")
        else:
            # Flat format
            for part in index.toc.parts:
                section_id = part.section_ids[0]
                lines.append(f"[{section_id}] {part.title}")
        
        return "\n".join(lines)
    
    def _parse_section_ids(
        self,
        response_text: str,
        index: DocumentIndex
    ) -> List[str]:
        """
        Parse section IDs from LLM response.
        
        Args:
            response_text: LLM response
            index: DocumentIndex for validation
            
        Returns:
            List of valid section IDs
        """
        try:
            # Try to extract JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start != -1 and json_end > 0:
                json_text = response_text[json_start:json_end]
                section_ids = json.loads(json_text)
            else:
                # Fallback: extract section_N patterns
                import re
                section_ids = re.findall(r'section_\d+', response_text)
            
            # Validate section IDs exist
            valid_ids = [
                sid for sid in section_ids
                if any(s.id == sid for s in index.sections)
            ]
            
            if not valid_ids:
                logger.warning("No valid section IDs found, using first section")
                valid_ids = [index.sections[0].id] if index.sections else []
            
            return valid_ids
            
        except Exception as e:
            logger.error(f"Failed to parse section IDs: {e}")
            # Fallback to first section
            return [index.sections[0].id] if index.sections else []
    
    def _expand_context(
        self,
        index: DocumentIndex,
        section_ids: List[str]
    ) -> List[str]:
        """
        Expand context by adding adjacent sections.
        
        Args:
            index: DocumentIndex
            section_ids: Selected section IDs
            
        Returns:
            Expanded list of section IDs (deduplicated)
        """
        expanded = set(section_ids)
        
        # Create section index map
        section_indices = {s.id: i for i, s in enumerate(index.sections)}
        
        for section_id in section_ids:
            idx = section_indices.get(section_id)
            if idx is None:
                continue
            
            # Add previous section
            if idx > 0:
                expanded.add(index.sections[idx - 1].id)
            
            # Add next section
            if idx < len(index.sections) - 1:
                expanded.add(index.sections[idx + 1].id)
        
        # Return in original order
        return [s.id for s in index.sections if s.id in expanded]
    
    def _retrieve_chunks(
        self,
        index: DocumentIndex,
        section_ids: List[str]
    ) -> List[Chunk]:
        """
        Retrieve chunks for selected sections.
        
        Args:
            index: DocumentIndex
            section_ids: Section IDs to retrieve
            
        Returns:
            List of Chunk objects
        """
        # Get chunk IDs from sections
        chunk_ids = set()
        for section_id in section_ids:
            section = next((s for s in index.sections if s.id == section_id), None)
            if section:
                chunk_ids.update(section.chunk_ids)
        
        # Retrieve chunks in order
        chunks = [c for c in index.chunks if c.id in chunk_ids]
        return chunks
    
    @retry_with_backoff(max_retries=3)
    def _generate_answer(
        self,
        chunks: List[Chunk],
        question: str
    ) -> str:
        """
        Generate answer using retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            question: User question
            
        Returns:
            Answer string
        """
        # Format context
        context = "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in chunks)
        
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- Be clear and concise
- If the context doesn't contain enough information, say so

Answer:"""
        
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            answer = response.text.strip()
        else:  # ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 1000}
                },
                timeout=120
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
        
        return answer
