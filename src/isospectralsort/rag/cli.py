"""
Command-line interface to query the foundational paper knowledge base.

Usage:
    python -m isospectralsort.rag.cli "How does Brockett prove sorting via Rearrangement Inequality?"
"""

import sys
from .engine import PaperRAG


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m isospectralsort.rag.cli \"<question or query>\"")
        print("Example: python -m isospectralsort.rag.cli \"How does Brockett prove sorting?\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    rag = PaperRAG()
    
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}\n")
    
    results = rag.query(query, top_k=2)
    if not results:
        print("No matching excerpts found in paper knowledge base.")
        return

    for rank, (chunk, score) in enumerate(results, 1):
        print(f"[{rank}] Match Score: {score:.4f}")
        print(f"Paper:   {chunk.paper_title}")
        print(f"Author:  {chunk.author}")
        print(f"Section: {chunk.section_title}")
        print("-" * 70)
        print(chunk.content[:600] + ("..." if len(chunk.content) > 600 else ""))
        if chunk.equations:
            print("\nKey Equations:")
            for eq in chunk.equations[:2]:
                print(f"  $$ {eq} $$")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
