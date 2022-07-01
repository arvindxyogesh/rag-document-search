"""
Ingestion script: loads documents, chunks, embeds, and builds/saves the vector store.
Run once before querying.

Usage:
    cd rag-document-search
    python scripts/ingest.py
"""
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vector_store.store import VectorStore


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    loader = DocumentLoader(config)
    chunker = TextChunker(config)
    embedder = Embedder(config)
    store = VectorStore(config)

    print("Step 1/4: Loading documents...")
    docs = loader.load()
    print(f"  Loaded {len(docs)} documents")

    print("Step 2/4: Chunking documents...")
    chunks = chunker.chunk(docs)
    print(f"  Created {len(chunks)} chunks")

    print("Step 3/4: Generating embeddings...")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Step 4/4: Building and saving vector store...")
    chunks_meta = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "chunk_index": c.chunk_index,
            **c.metadata,
        }
        for c in chunks
    ]
    store.build(embeddings, chunks_meta)
    store.save()

    print(f"\nIngestion complete!")
    print(f"  {len(docs)} documents → {len(chunks)} chunks → {embeddings.shape[0]} vectors")
    print(f"  Vector store saved to: {config['vector_store']['store_dir']}")
    print(f"\nRun 'python scripts/query.py' to start querying.")


if __name__ == "__main__":
    main()
