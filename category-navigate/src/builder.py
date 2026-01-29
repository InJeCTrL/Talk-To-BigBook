"""Index building orchestration for Category-Navigate System"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from .models import DocumentIndex, IndexMetadata
from .chunker import DocumentChunker
from .boundary_detector import BoundaryDetector
from .section_builder import SectionBuilder
from .toc_builder import TOCBuilder
from .config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class CategoryBuilder:
    """Orchestrates the index building pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize builder with configuration.
        
        Args:
            config: Configuration dictionary (uses defaults if not provided)
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Get provider setting
        provider = self.config.get("provider", "ollama")
        
        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=self.config["chunk_size"],
            overlap=self.config["overlap"]
        )
        
        self.boundary_detector = BoundaryDetector(
            base_url=self.config["ollama_base_url"],
            model=self.config["ollama_model"],
            preview_chars=self.config["boundary_preview_chars"],
            max_retries=self.config["max_retries"],
            provider=provider
        )
        
        self.section_builder = SectionBuilder(
            base_url=self.config["ollama_base_url"],
            model=self.config["ollama_model"],
            max_title_length=self.config["max_title_length"],
            max_retries=self.config["max_retries"],
            provider=provider
        )
        
        self.toc_builder = TOCBuilder(
            base_url=self.config["ollama_base_url"],
            model=self.config["ollama_model"],
            hierarchy_threshold=self.config["hierarchy_threshold"],
            max_retries=self.config["max_retries"],
            provider=provider
        )
        
        logger.info(f"CategoryBuilder initialized with provider={provider}")
    
    def build_index(
        self,
        document_path: str,
        output_path: Optional[str] = None
    ) -> DocumentIndex:
        """
        Build complete index for a document.
        
        Args:
            document_path: Path to document file
            output_path: Optional path to save index JSON
            
        Returns:
            DocumentIndex object
        """
        logger.info(f"Building index for: {document_path}")
        
        # Load document
        with open(document_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        logger.info(f"Loaded document: {len(document_text)} characters")
        
        # Step 1: Chunk document
        logger.info("Step 1: Chunking document...")
        chunks = self.chunker.chunk_document(document_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Detect boundaries
        logger.info("Step 2: Detecting topic boundaries...")
        boundaries = self.boundary_detector.detect_boundaries(chunks)
        logger.info(f"Detected {len(boundaries)} boundaries")
        
        # Step 3: Build sections
        logger.info("Step 3: Building sections...")
        sections = self.section_builder.build_sections(chunks, boundaries)
        logger.info(f"Created {len(sections)} sections")
        
        # Step 4: Build TOC
        logger.info("Step 4: Building table of contents...")
        toc = self.toc_builder.build_toc(sections)
        logger.info(f"Created TOC with {len(toc.parts)} parts (hierarchical={toc.is_hierarchical})")
        
        # Step 5: Assemble index
        metadata = IndexMetadata(
            document_path=document_path,
            created_at=datetime.now().isoformat(),
            model=self.config["ollama_model"],
            chunk_count=len(chunks),
            section_count=len(sections),
            part_count=len(toc.parts)
        )
        
        index = DocumentIndex(
            metadata=metadata,
            chunks=chunks,
            sections=sections,
            toc=toc
        )
        
        logger.info("Index building complete")
        
        # Save if output path provided
        if output_path:
            self.save_index(index, output_path)
        
        return index
    
    def save_index(self, index: DocumentIndex, output_path: str) -> None:
        """
        Save index to JSON file.
        
        Args:
            index: DocumentIndex to save
            output_path: Path to output file
        """
        logger.info(f"Saving index to: {output_path}")
        
        # Convert to dict
        index_dict = self._index_to_dict(index)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Index saved successfully ({output_file.stat().st_size} bytes)")
    
    def _index_to_dict(self, index: DocumentIndex) -> Dict[str, Any]:
        """
        Convert DocumentIndex to JSON-serializable dict.
        
        Args:
            index: DocumentIndex object
            
        Returns:
            Dictionary representation
        """
        return {
            "metadata": {
                "document_path": index.metadata.document_path,
                "created_at": index.metadata.created_at,
                "model": index.metadata.model,
                "chunk_count": index.metadata.chunk_count,
                "section_count": index.metadata.section_count,
                "part_count": index.metadata.part_count
            },
            "chunks": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos
                }
                for chunk in index.chunks
            ],
            "sections": [
                {
                    "id": section.id,
                    "title": section.title,
                    "chunk_ids": section.chunk_ids,
                    "start_pos": section.start_pos,
                    "end_pos": section.end_pos
                }
                for section in index.sections
            ],
            "toc": {
                "is_hierarchical": index.toc.is_hierarchical,
                "parts": [
                    {
                        "id": part.id,
                        "title": part.title,
                        "section_ids": part.section_ids
                    }
                    for part in index.toc.parts
                ]
            }
        }
