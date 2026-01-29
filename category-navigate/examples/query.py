#!/usr/bin/env python3
"""
Example script for querying a category-navigate index.

Usage:
    python query.py <index_path> <question> [--provider gemini|ollama]

Example:
    python query.py ../data/indexes/资治通鉴_index.json "有没有类似less is more的事情？" --provider gemini
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import CategoryRetriever
from src.config import setup_logging, get_config


def main():
    parser = argparse.ArgumentParser(
        description="Query category-navigate index"
    )
    parser.add_argument(
        "index_path",
        help="Path to index JSON file"
    )
    parser.add_argument(
        "question",
        help="Question to ask"
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable context expansion (don't include adjacent sections)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:7b",
        help="Ollama model name (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash-exp",
        help="Gemini model name (default: gemini-2.0-flash-exp)"
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
    
    # Prepare configuration
    config = get_config()
    config["ollama_base_url"] = args.ollama_url
    config["ollama_model"] = args.ollama_model
    config["gemini_model"] = args.gemini_model
    
    print(f"\n{'='*60}")
    print("Category-Navigate Query")
    print(f"{'='*60}")
    print(f"Index: {args.index_path}")
    print(f"Provider: {args.provider}")
    print(f"Question: {args.question}")
    print(f"Context expansion: {'disabled' if args.no_expand else 'enabled'}")
    print(f"{'='*60}\n")
    
    # Query index
    try:
        retriever = CategoryRetriever(provider=args.provider, config=config)
        index = retriever.load_index(args.index_path)
        
        print(f"Index loaded: {index.metadata.chunk_count} chunks, "
              f"{index.metadata.section_count} sections\n")
        
        answer = retriever.query(
            index,
            args.question,
            expand_context=not args.no_expand
        )
        
        print(f"{'='*60}")
        print("Answer:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
