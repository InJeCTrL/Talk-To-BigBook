"""Query retrieval system for Layered Summary System"""

import logging
import json
from typing import List, Dict

from .models import HierarchicalIndex, Node, QueryResult
from .summarizer import GeminiRetriever

logger = logging.getLogger(__name__)


class QueryRetriever:
    """Handles query processing with top-down traversal."""
    
    def __init__(self, gemini_retriever: GeminiRetriever):
        """
        Initialize with Gemini client.
        
        Args:
            gemini_retriever: GeminiRetriever instance
        """
        self.gemini = gemini_retriever
        self.index = None
        
        logger.info("QueryRetriever initialized")
    
    def query(self, question: str, index_path: str) -> QueryResult:
        """
        Process query and return answer with sources.
        
        Args:
            question: User query
            index_path: Path to JSON index file
            
        Returns:
            QueryResult with answer and source chunk IDs
        """
        logger.info(f"Processing query: {question}")
        
        # Load index
        self.index = self._load_index(index_path)
        
        # Initialize traversal path
        traversal_path = {}
        
        # Level 3: Start at top
        logger.info("Traversing Level 3...")
        level3_node = self.index.level3
        selected_l2_ids = self._traverse_level(question, [level3_node], 3)
        traversal_path[3] = selected_l2_ids
        
        # Level 2: Get selected nodes
        logger.info(f"Traversing Level 2 with {len(selected_l2_ids)} selected nodes...")
        level2_nodes = [node for node in self.index.level2 if node.id in selected_l2_ids]
        if not level2_nodes:
            level2_nodes = self.index.level2  # Fallback to all if none selected
        
        selected_l1_ids = self._traverse_level(question, level2_nodes, 2)
        traversal_path[2] = selected_l1_ids
        
        # Level 1: Get selected nodes
        logger.info(f"Traversing Level 1 with {len(selected_l1_ids)} selected nodes...")
        level1_nodes = [node for node in self.index.level1 if node.id in selected_l1_ids]
        if not level1_nodes:
            level1_nodes = self.index.level1  # Fallback to all if none selected
        
        selected_l0_ids = self._traverse_level(question, level1_nodes, 1)
        traversal_path[1] = selected_l0_ids
        
        # Retrieve Level 0 chunks with context expansion
        logger.info(f"Retrieving {len(selected_l0_ids)} Level 0 chunks...")
        contexts = self._retrieve_chunks(selected_l0_ids)
        
        # Generate answer
        logger.info("Generating answer...")
        answer = self.gemini.generate_answer(question, contexts)
        
        result = QueryResult(
            answer=answer,
            source_chunks=selected_l0_ids,
            traversal_path=traversal_path
        )
        
        logger.info("Query processing complete")
        return result
    
    def _load_index(self, index_path: str) -> HierarchicalIndex:
        """
        Load index from JSON file.
        
        Args:
            index_path: Path to JSON index file
            
        Returns:
            HierarchicalIndex object
            
        Raises:
            ValueError: If JSON is malformed or missing required fields
        """
        logger.info(f"Loading index from {index_path}")
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ["metadata", "level0", "level1", "level2", "level3"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            index = HierarchicalIndex.from_dict(data)
            logger.info(f"Index loaded: {len(index.level0)} L0 nodes, {len(index.level1)} L1 nodes, "
                       f"{len(index.level2)} L2 nodes")
            
            return index
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required field in index: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load index: {e}")
    
    def _traverse_level(self, query: str, nodes: List[Node], level: int) -> List[str]:
        """
        Use LLM to select relevant nodes at current level.
        
        Args:
            query: User query
            nodes: Available nodes at current level
            level: Current level (3, 2, or 1)
            
        Returns:
            Selected node IDs (or children IDs if at level > 0)
        """
        if not nodes:
            return []
        
        # For Level 3, we need to get children IDs (Level 2 IDs)
        # For Level 2, we need to get children IDs (Level 1 IDs)
        # For Level 1, we need to get children IDs (Level 0 IDs)
        
        if level == 3:
            # Level 3 has only one node, return its children (L2 IDs)
            return nodes[0].children
        
        # For Level 2 and 1, ask LLM to select relevant nodes
        selected_node_ids = self.gemini.select_relevant_nodes(query, nodes, level)
        
        # Get children IDs from selected nodes
        selected_nodes = [node for node in nodes if node.id in selected_node_ids]
        
        if not selected_nodes:
            # Fallback: use all nodes
            selected_nodes = nodes
        
        # Collect all children IDs
        children_ids = []
        for node in selected_nodes:
            children_ids.extend(node.children)
        
        return children_ids
    
    def _retrieve_chunks(self, selected_l0_ids: List[str]) -> List[str]:
        """
        Retrieve Level 0 chunks with context expansion (±1 chunk).
        
        Args:
            selected_l0_ids: Selected Level 0 chunk IDs
            
        Returns:
            List of text chunks with context
        """
        contexts = []
        
        # Create ID to index mapping
        id_to_idx = {node.id: idx for idx, node in enumerate(self.index.level0)}
        
        # Get unique indices and sort them
        indices = sorted(set(id_to_idx.get(chunk_id, -1) for chunk_id in selected_l0_ids))
        indices = [i for i in indices if i >= 0]
        
        if not indices:
            return []
        
        # Expand context for each chunk
        expanded_indices = set()
        for idx in indices:
            # Add previous chunk
            if idx > 0:
                expanded_indices.add(idx - 1)
            # Add current chunk
            expanded_indices.add(idx)
            # Add next chunk
            if idx < len(self.index.level0) - 1:
                expanded_indices.add(idx + 1)
        
        # Get texts in order
        for idx in sorted(expanded_indices):
            contexts.append(self.index.level0[idx].text)
        
        logger.debug(f"Retrieved {len(contexts)} chunks (with context expansion)")
        return contexts
    
    def _expand_context(self, chunk_id: str) -> List[str]:
        """
        Get chunk and its adjacent chunks (±1 position).
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            List of chunk texts [prev, current, next] (as available)
        """
        # Find chunk index
        chunk_idx = None
        for idx, node in enumerate(self.index.level0):
            if node.id == chunk_id:
                chunk_idx = idx
                break
        
        if chunk_idx is None:
            return []
        
        contexts = []
        
        # Previous chunk
        if chunk_idx > 0:
            contexts.append(self.index.level0[chunk_idx - 1].text)
        
        # Current chunk
        contexts.append(self.index.level0[chunk_idx].text)
        
        # Next chunk
        if chunk_idx < len(self.index.level0) - 1:
            contexts.append(self.index.level0[chunk_idx + 1].text)
        
        return contexts
