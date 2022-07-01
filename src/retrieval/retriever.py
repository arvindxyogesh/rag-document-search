"""
Retrieval module.
Combines the Embedder and VectorStore to retrieve relevant chunks for a query.
"""
import logging
from typing import List, Tuple

from src.embeddings.embedder import Embedder
from src.vector_store.store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Embeds queries and retrieves the top-k most relevant document chunks."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, config: dict):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = config["retrieval"]["top_k"]

    def retrieve(self, query: str) -> List[Tuple[dict, float]]:
        """Return top-k (chunk_metadata, similarity_score) pairs for a query."""
        query_emb = self.embedder.embed_query(query)
        results = self.vector_store.search(query_emb, top_k=self.top_k)
        logger.info(f"Retrieved {len(results)} chunks for: '{query[:60]}'")
        return results

    def format_context(self, results: List[Tuple[dict, float]]) -> str:
        """Format retrieved chunks into a context string for the LLM prompt."""
        parts = []
        for i, (meta, score) in enumerate(results, 1):
            header = f"[Source {i}: {meta.get('doc_id', 'unknown')} (score={score:.3f})]"
            parts.append(f"{header}\n{meta['text']}")
        return "\n\n".join(parts)
