#!/usr/bin/env python3
"""
Example script for building a category-navigate index.

Usage:
    python build_index.py <document_path> [output_path]
    python build_index.py <document_path> --provider dashscope --dashscope-key your_key

Example:
    python build_index.py ../data/documents/资治通鉴.txt ../data/indexes/资治通鉴_index.json
    python build_index.py ../data/documents/资治通鉴.txt --provider dashscope
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.builder import CategoryBuilder
from src.config import setup_logging, get_config


def main():
    parser = argparse.ArgumentParser(
        description="Build category-navigate index for a document"
    )
    parser.add_argument(
        "document_path",
        help="Path to document file"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Path to save index JSON (optional)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size in tokens (default: 2000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap size in tokens (default: 200)"
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "dashscope"],
        default="ollama",
        help="LLM provider: ollama (local) or dashscope (online, faster)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: qwen2.5:7b for ollama, qwen2-7b-instruct for dashscope)"
    )
    parser.add_argument(
        "--dashscope-key",
        help="DashScope API key (or set DASHSCOPE_API_KEY env var)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Set default model based on provider
    if args.model is None:
        if args.provider == "dashscope":
            args.model = "qwen2-7b-instruct"
        else:
            args.model = "qwen2.5:7b"
    
    # Handle DashScope API key
    if args.provider == "dashscope":
        api_key = args.dashscope_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("Error: DashScope API key required. Set DASHSCOPE_API_KEY env or use --dashscope-key", file=sys.stderr)
            sys.exit(1)
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    # Prepare configuration
    config = get_config()
    config["chunk_size"] = args.chunk_size
    config["overlap"] = args.overlap
    config["provider"] = args.provider
    config["ollama_base_url"] = args.ollama_url
    config["ollama_model"] = args.model
    
    # Generate output path if not provided
    if not args.output_path:
        doc_path = Path(args.document_path)
        output_dir = doc_path.parent.parent / "indexes"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_path = str(output_dir / f"{doc_path.stem}_index.json")
    
    provider_info = f"{args.provider} ({args.model})"
    if args.provider == "dashscope":
        provider_info += " [online - faster]"
    else:
        provider_info += " [local]"
    
    print(f"\n{'='*60}")
    print("Category-Navigate Index Builder")
    print(f"{'='*60}")
    print(f"Document: {args.document_path}")
    print(f"Output: {args.output_path}")
    print(f"Provider: {provider_info}")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Overlap: {args.overlap} tokens")
    print(f"{'='*60}\n")
    
    # Build index
    try:
        builder = CategoryBuilder(config)
        index = builder.build_index(args.document_path, args.output_path)
        
        print(f"\n{'='*60}")
        print("Index Building Complete!")
        print(f"{'='*60}")
        print(f"Chunks: {len(index.chunks)}")
        print(f"Sections: {len(index.sections)}")
        print(f"Parts: {len(index.toc.parts)}")
        print(f"Hierarchical: {index.toc.is_hierarchical}")
        print(f"Index saved to: {args.output_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
