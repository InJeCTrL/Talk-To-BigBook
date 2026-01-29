"""Data models for Layered Summary System"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class Chunk:
    """Represents a chunk of the original document."""
    id: str
    text: str
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
                self.token_count == other.token_count and
                self.start_pos == other.start_pos and
                self.end_pos == other.end_pos)


@dataclass
class Node:
    """Represents a node in the hierarchical index."""
    id: str
    text: str
    token_count: int
    children: List[str]
    start_pos: int
    end_pos: int
    level: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create from dictionary."""
        return cls(**data)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return (self.id == other.id and 
                self.text == other.text and
                self.token_count == other.token_count and
                self.children == other.children and
                self.start_pos == other.start_pos and
                self.end_pos == other.end_pos and
                self.level == other.level)


@dataclass
class IndexMetadata:
    """Metadata for a hierarchical index."""
    document_name: str
    total_tokens: int
    creation_timestamp: str
    language: str
    config: Dict[str, Any]
    
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
        return (self.document_name == other.document_name and
                self.total_tokens == other.total_tokens and
                self.creation_timestamp == other.creation_timestamp and
                self.language == other.language and
                self.config == other.config)


@dataclass
class HierarchicalIndex:
    """Complete hierarchical index structure."""
    metadata: IndexMetadata
    level0: List[Node]
    level1: List[Node]
    level2: List[Node]
    level3: Node
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "level0": [node.to_dict() for node in self.level0],
            "level1": [node.to_dict() for node in self.level1],
            "level2": [node.to_dict() for node in self.level2],
            "level3": self.level3.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchicalIndex':
        """Create from dictionary."""
        return cls(
            metadata=IndexMetadata.from_dict(data["metadata"]),
            level0=[Node.from_dict(n) for n in data["level0"]],
            level1=[Node.from_dict(n) for n in data["level1"]],
            level2=[Node.from_dict(n) for n in data["level2"]],
            level3=Node.from_dict(data["level3"])
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchicalIndex):
            return False
        return (self.metadata == other.metadata and
                self.level0 == other.level0 and
                self.level1 == other.level1 and
                self.level2 == other.level2 and
                self.level3 == other.level3)


@dataclass
class QueryResult:
    """Result of a query operation."""
    answer: str
    source_chunks: List[str]
    traversal_path: Dict[int, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """Create from dictionary."""
        # Convert string keys back to int for traversal_path
        traversal_path = {int(k): v for k, v in data.get("traversal_path", {}).items()}
        return cls(
            answer=data["answer"],
            source_chunks=data["source_chunks"],
            traversal_path=traversal_path
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryResult):
            return False
        return (self.answer == other.answer and
                self.source_chunks == other.source_chunks and
                self.traversal_path == other.traversal_path)
