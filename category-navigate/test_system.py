#!/usr/bin/env python3
"""Quick test script to verify the system works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.builder import CategoryBuilder
from src.config import setup_logging

def main():
    setup_logging("INFO")
    
    print("\n" + "="*60)
    print("Testing Category-Navigate System")
    print("="*60 + "\n")
    
    # Test with small document
    doc_path = "data/documents/test_doc.txt"
    output_path = "data/indexes/test_doc_index.json"
    
    print(f"Building index for: {doc_path}")
    print("This will take a few minutes as it calls Ollama multiple times...\n")
    
    try:
        builder = CategoryBuilder()
        index = builder.build_index(doc_path, output_path)
        
        print("\n" + "="*60)
        print("SUCCESS! Index built successfully")
        print("="*60)
        print(f"Chunks: {len(index.chunks)}")
        print(f"Sections: {len(index.sections)}")
        print(f"Parts: {len(index.toc.parts)}")
        print(f"Hierarchical: {index.toc.is_hierarchical}")
        print(f"\nIndex saved to: {output_path}")
        
        print("\n" + "="*60)
        print("Section Titles:")
        print("="*60)
        for i, section in enumerate(index.sections):
            print(f"{i+1}. {section.title}")
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("="*60)
        print("1. Test querying with:")
        print(f"   python examples/query.py {output_path} \"你的问题\" --provider gemini")
        print("\n2. Or use Ollama (offline):")
        print(f"   python examples/query.py {output_path} \"你的问题\" --provider ollama")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
