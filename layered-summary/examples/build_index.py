#!/usr/bin/env python3
"""Example script to build a hierarchical index from a document."""

import sys
import os
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import setup_logging, get_config
from src.chunker import DocumentChunker
from src.summarizer import OllamaSummarizer, DashScopeSummarizer
from src.builder import IndexBuilder


def main():
    """Build index from document."""
    parser = argparse.ArgumentParser(description='Build hierarchical index from document')
    parser.add_argument('document', help='Path to input document')
    parser.add_argument('--output', '-o', help='Path to output JSON index', required=True)
    parser.add_argument('--chunk-size', type=int, default=2000, help='Chunk size in tokens')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap size in tokens')
    parser.add_argument('--window-size', type=int, default=5, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=3, help='Sliding window stride')
    
    # Provider selection
    parser.add_argument('--provider', choices=['ollama', 'dashscope'], default='ollama',
                        help='LLM provider for summarization (default: ollama)')
    
    # Ollama options
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama API URL')
    parser.add_argument('--ollama-model', default='qwen2.5:7b', help='Ollama model name')
    
    # DashScope options
    parser.add_argument('--dashscope-key', help='DashScope API key (or set DASHSCOPE_API_KEY env)')
    parser.add_argument('--dashscope-model', default='qwen2-7b-instruct', 
                        help='DashScope model name (default: qwen2-7b-instruct)')
    
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Building Hierarchical Index")
    logger.info("=" * 60)
    logger.info(f"Document: {args.document}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    logger.info(f"Window size: {args.window_size}, Stride: {args.stride}")
    logger.info(f"Provider: {args.provider}")
    
    if args.provider == 'ollama':
        logger.info(f"Ollama: {args.ollama_url} ({args.ollama_model})")
    else:
        logger.info(f"DashScope model: {args.dashscope_model}")
    
    logger.info("=" * 60)
    
    # Check if document exists
    if not os.path.exists(args.document):
        logger.error(f"Document not found: {args.document}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        chunker = DocumentChunker(chunk_size=args.chunk_size, overlap=args.overlap)
        
        # Select summarizer based on provider
        if args.provider == 'dashscope':
            api_key = args.dashscope_key or os.environ.get('DASHSCOPE_API_KEY')
            if not api_key:
                logger.error("DashScope API key required. Set DASHSCOPE_API_KEY env or use --dashscope-key")
                sys.exit(1)
            summarizer = DashScopeSummarizer(api_key=api_key, model=args.dashscope_model)
        else:
            summarizer = OllamaSummarizer(model=args.ollama_model, base_url=args.ollama_url)
        
        builder = IndexBuilder(
            chunker=chunker,
            summarizer=summarizer,
            window_size=args.window_size,
            stride=args.stride
        )
        
        # Build index
        logger.info("Building index (this may take a while)...")
        index = builder.build_index(args.document, args.output)
        
        # Print statistics
        logger.info("=" * 60)
        logger.info("Index Built Successfully!")
        logger.info("=" * 60)
        logger.info(f"Document: {index.metadata.document_name}")
        logger.info(f"Language: {index.metadata.language}")
        logger.info(f"Total tokens: {index.metadata.total_tokens:,}")
        logger.info(f"Level 0 (chunks): {len(index.level0)} nodes")
        logger.info(f"Level 1 (summaries): {len(index.level1)} nodes")
        logger.info(f"Level 2 (summaries): {len(index.level2)} nodes")
        logger.info(f"Level 3 (top): 1 node ({index.level3.token_count} tokens)")
        logger.info(f"Output saved to: {args.output}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
