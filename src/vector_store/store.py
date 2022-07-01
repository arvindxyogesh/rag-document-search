"""
Vector store module.
Uses FAISS for efficient approximate nearest-neighbor search.
Stores chunk metadata alongside the embedding index.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with chunk metadata persistence."""

    def __init__(self, config: dict):
        self.store_dir = Path(config["vector_store"]["store_dir"])
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._chunks_meta: List[dict] = []

    def build(self, embeddings: np.ndarray, chunks_meta: List[dict]) -> None:
        """Build a FAISS IndexFlatIP from normalized embeddings."""
        import faiss
        dim = embeddings.shape[1]
        # IndexFlatIP + normalized vectors = exact cosine similarity search
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks_meta = chunks_meta
        logger.info(f"Built FAISS index: {self._index.ntotal} vectors (dim={dim})")

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Return top-k (chunk_metadata, similarity_score) pairs."""
        if self._index is None:
            raise RuntimeError("Vector store not built. Call build() or load() first.")
        scores, indices = self._index.search(query_emb, top_k)
        results = [
            (self._chunks_meta[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss
        faiss.write_index(self._index, str(self.store_dir / "index.faiss"))
        with open(self.store_dir / "chunks_meta.pkl", "wb") as f:
            pickle.dump(self._chunks_meta, f)
        logger.info(f"Vector store saved to {self.store_dir}/")

    def load(self) -> None:
        """Load persisted index and metadata from disk."""
        import faiss
        index_path = self.store_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No saved index at {index_path}. Run scripts/ingest.py first."
            )
        self._index = faiss.read_index(str(index_path))
        with open(self.store_dir / "chunks_meta.pkl", "rb") as f:
            self._chunks_meta = pickle.load(f)
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")
