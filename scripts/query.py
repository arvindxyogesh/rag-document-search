"""
Interactive RAG query script.
Loads the persisted vector store and answers questions using retrieved context.

Usage:
    cd rag-document-search
    python scripts/query.py

    # With OpenAI LLM generation:
    OPENAI_API_KEY=your-key python scripts/query.py
"""
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress verbose library logs during interactive use
logging.basicConfig(level=logging.WARNING)

from src.embeddings.embedder import Embedder
from src.vector_store.store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import RAGGenerator


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading vector store...")
    embedder = Embedder(config)
    store = VectorStore(config)
    store.load()
    retriever = Retriever(embedder, store, config)
    generator = RAGGenerator(config)

    mode = "OpenAI GPT" if os.getenv("OPENAI_API_KEY") else "local template"
    print(f"RAG system ready (generation mode: {mode})")
    print("Type 'quit' to exit\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        results = retriever.retrieve(query)
        context = retriever.format_context(results)
        answer = generator.generate(query, context)

        print(f"\n{'='*60}")
        print(f"Answer:\n{answer}")
        print(f"\nRetrieved chunks ({len(results)}):")
        for meta, score in results:
            print(f"  [{score:.3f}] {meta['doc_id']} — chunk {meta.get('chunk_index', '?')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
