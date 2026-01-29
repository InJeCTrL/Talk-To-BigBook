"""Hierarchical index builder for Layered Summary System"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any

from .models import Node, HierarchicalIndex, IndexMetadata
from .chunker import DocumentChunker
from .summarizer import OllamaSummarizer
from .utils import count_tokens, detect_language

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Orchestrates hierarchical index creation using bottom-up approach."""
    
    def __init__(
        self,
        chunker: DocumentChunker,
        summarizer: OllamaSummarizer,
        window_size: int = 5,
        stride: int = 3,
        level2_budget: int = 5000,
        level3_budget: int = 1000
    ):
        """
        Initialize builder with dependencies and configuration.
        
        Args:
            chunker: DocumentChunker instance
            summarizer: OllamaSummarizer instance
            window_size: Window size for Level 1 sliding window
            stride: Stride for Level 1 sliding window
            level2_budget: Token budget for Level 2 grouping
            level3_budget: Target tokens for Level 3 summary
        """
        self.chunker = chunker
        self.summarizer = summarizer
        self.window_size = window_size
        self.stride = stride
        self.level2_budget = level2_budget
        self.level3_budget = level3_budget
        
        logger.info(
            f"IndexBuilder initialized: window_size={window_size}, stride={stride}, "
            f"level2_budget={level2_budget}, level3_budget={level3_budget}"
        )
    
    def build_index(self, document_path: str, output_path: str) -> HierarchicalIndex:
        """
        Build complete hierarchical index from document.
        
        Args:
            document_path: Path to input document
            output_path: Path to save JSON index
            
        Returns:
            HierarchicalIndex object
        """
        logger.info(f"Building index for document: {document_path}")
        
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Detect language
        language = detect_language(text)
        logger.info(f"Detected language: {language}")
        
        # Build levels
        level0_nodes = self._build_level0(text)
        level1_nodes = self._build_level1(level0_nodes, language)
        level2_nodes = self._build_level2(level1_nodes, language)
        level3_node = self._build_level3(level2_nodes, language)
        
        # Create metadata
        total_tokens = sum(node.token_count for node in level0_nodes)
        metadata = IndexMetadata(
            document_name=document_path.split('/')[-1],
            total_tokens=total_tokens,
            creation_timestamp=datetime.now().isoformat(),
            language=language,
            config={
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.overlap,
                "window_size": self.window_size,
                "stride": self.stride,
                "level2_budget": self.level2_budget,
                "level3_budget": self.level3_budget
            }
        )
        
        # Create index
        index = HierarchicalIndex(
            metadata=metadata,
            level0=level0_nodes,
            level1=level1_nodes,
            level2=level2_nodes,
            level3=level3_node
        )
        
        # Save to JSON
        self._save_index(index, output_path)
        
        logger.info(f"Index built successfully: {output_path}")
        return index
    
    def _build_level0(self, text: str) -> List[Node]:
        """
        Create Level 0 nodes (original chunks).
        
        Args:
            text: Document text
            
        Returns:
            List of Level 0 nodes
        """
        logger.info("Building Level 0 (chunks)...")
        chunks = self.chunker.chunk_document(text)
        nodes = self.chunker.chunks_to_nodes(chunks)
        logger.info(f"Level 0 complete: {len(nodes)} nodes")
        return nodes
    
    def _build_level1(self, level0_nodes: List[Node], language: str) -> List[Node]:
        """
        Create Level 1 using sliding window summarization.
        
        Args:
            level0_nodes: Level 0 nodes
            language: Document language
            
        Returns:
            List of Level 1 nodes
        """
        # Calculate total windows
        total_windows = (len(level0_nodes) - self.window_size) // self.stride + 1
        if (len(level0_nodes) - self.window_size) % self.stride != 0:
            total_windows += 1
        
        logger.info(f"Building Level 1 (sliding window summaries)... Total: {total_windows} windows")
        level1_nodes = []
        position = 0
        
        while position < len(level0_nodes):
            # Get window
            window_end = min(position + self.window_size, len(level0_nodes))
            window_nodes = level0_nodes[position:window_end]
            
            # Concatenate texts
            combined_text = "\n\n".join([node.text for node in window_nodes])
            
            # Generate summary
            summary_text = self.summarizer.summarize(
                combined_text,
                language=language,
                max_tokens=500
            )
            
            # Create node
            node = Node(
                id=f"L1_{len(level1_nodes)}",
                text=summary_text,
                token_count=count_tokens(summary_text),
                children=[node.id for node in window_nodes],
                start_pos=window_nodes[0].start_pos,
                end_pos=window_nodes[-1].end_pos,
                level=1
            )
            
            level1_nodes.append(node)
            
            # Progress output
            progress = len(level1_nodes)
            percent = (progress / total_windows) * 100
            logger.info(f"Level 1 progress: {progress}/{total_windows} ({percent:.1f}%)")
            
            # Move position by stride
            position += self.stride
            
            # Break if we've covered all nodes
            if window_end >= len(level0_nodes):
                break
        
        logger.info(f"Level 1 complete: {len(level1_nodes)} nodes")
        return level1_nodes
    
    def _build_level2(self, level1_nodes: List[Node], language: str) -> List[Node]:
        """
        Create Level 2 using dynamic token-based grouping.
        
        Args:
            level1_nodes: Level 1 nodes
            language: Document language
            
        Returns:
            List of Level 2 nodes
        """
        # Estimate total groups
        total_tokens = sum(node.token_count for node in level1_nodes)
        estimated_groups = max(1, total_tokens // self.level2_budget)
        
        logger.info(f"Building Level 2 (grouped summaries)... Estimated: ~{estimated_groups} groups")
        level2_nodes = []
        current_group = []
        current_tokens = 0
        processed_nodes = 0
        
        for node in level1_nodes:
            processed_nodes += 1
            
            # Check if adding this node would exceed budget
            if current_tokens + node.token_count > self.level2_budget and current_group:
                # Summarize current group
                summary_node = self._summarize_group(
                    current_group,
                    f"L2_{len(level2_nodes)}",
                    2,
                    language,
                    max_tokens=200
                )
                level2_nodes.append(summary_node)
                
                # Progress output
                percent = (processed_nodes / len(level1_nodes)) * 100
                logger.info(f"Level 2 progress: {len(level2_nodes)} groups created, {processed_nodes}/{len(level1_nodes)} nodes processed ({percent:.1f}%)")
                
                # Reset group
                current_group = [node]
                current_tokens = node.token_count
            else:
                current_group.append(node)
                current_tokens += node.token_count
        
        # Handle remaining group
        if current_group:
            summary_node = self._summarize_group(
                current_group,
                f"L2_{len(level2_nodes)}",
                2,
                language,
                max_tokens=200
            )
            level2_nodes.append(summary_node)
            logger.info(f"Level 2 progress: {len(level2_nodes)} groups created (100%)")
        
        logger.info(f"Level 2 complete: {len(level2_nodes)} nodes")
        return level2_nodes
    
    def _build_level3(self, level2_nodes: List[Node], language: str) -> Node:
        """
        Create Level 3 (single top summary).
        
        Args:
            level2_nodes: Level 2 nodes
            language: Document language
            
        Returns:
            Level 3 node
        """
        logger.info("Building Level 3 (top summary)...")
        
        # Concatenate all Level 2 summaries
        combined_text = "\n\n".join([node.text for node in level2_nodes])
        
        # Generate top summary
        summary_text = self.summarizer.summarize(
            combined_text,
            language=language,
            max_tokens=self.level3_budget
        )
        
        # Create node
        node = Node(
            id="L3_0",
            text=summary_text,
            token_count=count_tokens(summary_text),
            children=[node.id for node in level2_nodes],
            start_pos=level2_nodes[0].start_pos,
            end_pos=level2_nodes[-1].end_pos,
            level=3
        )
        
        logger.info(f"Level 3 complete: 1 node with {node.token_count} tokens")
        return node
    
    def _summarize_group(
        self,
        nodes: List[Node],
        node_id: str,
        level: int,
        language: str,
        max_tokens: int
    ) -> Node:
        """
        Summarize a group of nodes.
        
        Args:
            nodes: Nodes to summarize
            node_id: ID for new node
            level: Level of new node
            language: Document language
            max_tokens: Maximum tokens for summary
            
        Returns:
            Summary node
        """
        combined_text = "\n\n".join([node.text for node in nodes])
        summary_text = self.summarizer.summarize(
            combined_text,
            language=language,
            max_tokens=max_tokens
        )
        
        return Node(
            id=node_id,
            text=summary_text,
            token_count=count_tokens(summary_text),
            children=[node.id for node in nodes],
            start_pos=nodes[0].start_pos,
            end_pos=nodes[-1].end_pos,
            level=level
        )
    
    def _save_index(self, index: HierarchicalIndex, output_path: str) -> None:
        """
        Save index to JSON file.
        
        Args:
            index: HierarchicalIndex to save
            output_path: Path to output file
        """
        logger.info(f"Saving index to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info("Index saved successfully")
