"""Data models for Category-Navigate System"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Chunk:
    """Represents a chunk of the original document."""
    id: str
    text: str  # Changed from 'content' to match chunker
    token_count: int
    start_pos: int
    end_pos: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create from dictionary."""
        return cls(**data)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return False
        return (self.id == other.id and
                self.text == other.text and
                self.start_pos == other.start_pos and
                self.end_pos == other.end_pos and
                self.token_count == other.token_count)


@dataclass
class Section:
    """Represents a section (group of chunks on same topic)."""
    id: str
    title: str
    chunk_ids: List[str]
    start_pos: int
    end_pos: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Section':
        """Create from dictionary."""
        # Handle old format with summary field
        if 'summary' in data:
            del data['summary']
        return cls(**data)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Section):
            return False
        return (self.id == other.id and
                self.title == other.title and
                self.chunk_ids == other.chunk_ids and
                self.start_pos == other.start_pos and
                self.end_pos == other.end_pos)


@dataclass
class Part:
    """Represents a higher-level grouping of sections."""
    id: str
    title: str
    section_ids: List[str]
    children: List[str] = field(default_factory=list)  # For multi-level hierarchy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Part':
        """Create from dictionary."""
        # Handle missing children field for backward compatibility
        if 'children' not in data:
            data['children'] = []
        # Handle old format with summary field
        if 'summary' in data:
            del data['summary']
        return cls(**data)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Part):
            return False
        return (self.id == other.id and
                self.title == other.title and
                self.section_ids == other.section_ids and
                self.children == other.children)


@dataclass
class TOC:
    """Table of Contents structure with multi-level support."""
    parts: List[Part]
    is_hierarchical: bool = False
    levels: int = 1  # Number of hierarchy levels
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "parts": [p.to_dict() for p in self.parts],
            "is_hierarchical": self.is_hierarchical,
            "levels": self.levels
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TOC':
        """Create from dictionary."""
        return cls(
            parts=[Part.from_dict(p) for p in data["parts"]],
            is_hierarchical=data.get("is_hierarchical", False),
            levels=data.get("levels", 1)
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TOC):
            return False
        return (self.parts == other.parts and
                self.is_hierarchical == other.is_hierarchical and
                self.levels == other.levels)


@dataclass
class IndexMetadata:
    """Metadata for a document index."""
    document_path: str
    created_at: str
    model: str
    chunk_size: int
    overlap: int
    num_chunks: int
    num_sections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexMetadata):
            return False
        return (self.document_path == other.document_path and
                self.created_at == other.created_at and
                self.model == other.model and
                self.chunk_size == other.chunk_size and
                self.overlap == other.overlap and
                self.num_chunks == other.num_chunks and
                self.num_sections == other.num_sections)


@dataclass
class DocumentIndex:
    """Complete document index structure."""
    metadata: IndexMetadata
    chunks: List[Chunk]
    sections: List[Section]
    toc: TOC
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "sections": [s.to_dict() for s in self.sections],
            "toc": self.toc.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentIndex':
        """Create from dictionary."""
        return cls(
            metadata=IndexMetadata.from_dict(data["metadata"]),
            chunks=[Chunk.from_dict(c) for c in data["chunks"]],
            sections=[Section.from_dict(s) for s in data["sections"]],
            toc=TOC.from_dict(data["toc"])
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentIndex):
            return False
        return (self.metadata == other.metadata and
                self.chunks == other.chunks and
                self.sections == other.sections and
                self.toc == other.toc)
