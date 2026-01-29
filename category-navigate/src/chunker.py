"""Document chunking module for Category-Navigate System"""

import logging
from typing import List
import tiktoken

from .models import Chunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits documents into overlapping token-based chunks."""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Target tokens per chunk
            overlap: Overlap tokens between chunks
        """
        if chunk_size <= overlap:
            raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"DocumentChunker initialized: chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk_document(self, text: str) -> List[Chunk]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of Chunk objects with id, text, token_count, start_pos, end_pos
        """
        if not text:
            return []
        
        # Tokenize the entire document
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        logger.info(f"Chunking document: {total_tokens} tokens")
        
        # If document is shorter than chunk_size, create single chunk
        if total_tokens <= self.chunk_size:
            logger.info("Document shorter than chunk_size, creating single chunk")
            chunk = Chunk(
                id="chunk_0",
                text=text,
                token_count=total_tokens,
                start_pos=0,
                end_pos=len(text)
            )
            return [chunk]
        
        chunks = []
        position = 0
        chunk_id = 0
        stride = self.chunk_size - self.overlap
        
        while position < total_tokens:
            # Extract chunk tokens
            end_position = min(position + self.chunk_size, total_tokens)
            chunk_tokens = tokens[position:end_position]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Find character positions in original text
            # Approximate by decoding from start
            if position == 0:
                start_char = 0
            else:
                start_char = len(self.encoding.decode(tokens[:position]))
            
            end_char = len(self.encoding.decode(tokens[:end_position]))
            
            chunk = Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                token_count=len(chunk_tokens),
                start_pos=start_char,
                end_pos=end_char
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move position forward by stride
            position += stride
            
            # Break if we've covered the document
            if end_position >= total_tokens:
                break
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        return len(self.encoding.encode(text))
