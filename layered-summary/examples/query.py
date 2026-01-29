#!/usr/bin/env python3
"""Example script to query a hierarchical index."""

import sys
import os
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import setup_logging
from src.summarizer import GeminiRetriever
from src.retriever import QueryRetriever


def main():
    """Query hierarchical index."""
    parser = argparse.ArgumentParser(description='Query hierarchical index')
    parser.add_argument('index', help='Path to JSON index file')
    parser.add_argument('query', help='Query question')
    parser.add_argument('--gemini-key', required=True, help='Google Gemini API key')
    parser.add_argument('--gemini-model', default='gemini-2.0-flash-exp', help='Gemini model name')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--show-sources', action='store_true', help='Show source chunks')
    parser.add_argument('--show-path', action='store_true', help='Show traversal path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Querying Hierarchical Index")
    logger.info("=" * 60)
    logger.info(f"Index: {args.index}")
    logger.info(f"Query: {args.query}")
    logger.info(f"Gemini model: {args.gemini_model}")
    logger.info("=" * 60)
    
    # Check if index exists
    if not os.path.exists(args.index):
        logger.error(f"Index not found: {args.index}")
        sys.exit(1)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        gemini = GeminiRetriever(api_key=args.gemini_key, model=args.gemini_model)
        retriever = QueryRetriever(gemini_retriever=gemini)
        
        # Process query
        logger.info("Processing query...")
        result = retriever.query(args.query, args.index)
        
        # Print results
        logger.info("=" * 60)
        logger.info("Query Results")
        logger.info("=" * 60)
        print("\nAnswer:")
        print("-" * 60)
        print(result.answer)
        print("-" * 60)
        
        if args.show_sources:
            print(f"\nSource Chunks ({len(result.source_chunks)}):")
            print("-" * 60)
            for chunk_id in result.source_chunks:
                print(f"  - {chunk_id}")
            print("-" * 60)
        
        if args.show_path:
            print("\nTraversal Path:")
            print("-" * 60)
            for level in sorted(result.traversal_path.keys(), reverse=True):
                nodes = result.traversal_path[level]
                print(f"  Level {level}: {len(nodes)} nodes selected")
                if level > 0:  # Don't print all L0 IDs
                    print(f"    {', '.join(nodes[:5])}" + 
                          (f" ... (+{len(nodes)-5} more)" if len(nodes) > 5 else ""))
            print("-" * 60)
        
        logger.info("Query completed successfully")
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
