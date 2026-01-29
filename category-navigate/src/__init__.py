"""Category-Navigate: Topic-based document indexing and retrieval system."""

from .models import (
    Chunk,
    Section,
    Part,
    TOC,
    IndexMetadata,
    DocumentIndex
)
from .chunker import DocumentChunker
from .boundary_detector import BoundaryDetector
from .section_builder import SectionBuilder
from .toc_builder import TOCBuilder
from .builder import CategoryBuilder
from .retriever import CategoryRetriever
from .config import get_config, setup_logging

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Section",
    "Part",
    "TOC",
    "IndexMetadata",
    "DocumentIndex",
    "DocumentChunker",
    "BoundaryDetector",
    "SectionBuilder",
    "TOCBuilder",
    "CategoryBuilder",
    "CategoryRetriever",
    "get_config",
    "setup_logging",
]
