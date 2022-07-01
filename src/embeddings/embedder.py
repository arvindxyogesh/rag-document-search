"""
Embedding module.
Uses sentence-transformers for local, offline embedding generation.
Embeddings are L2-normalized so that inner product == cosine similarity.
"""
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Generates dense vector embeddings using sentence-transformers."""

    def __init__(self, config: dict):
        self.model_name = config["embeddings"]["model_name"]
        self.batch_size = config["embeddings"].get("batch_size", 64)
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model ready")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) float32 array, L2-normalized."""
        self._load_model()
        logger.info(f"Embedding {len(texts)} texts (batch_size={self.batch_size})...")
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2-norm → dot product == cosine similarity
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (1, dim) array."""
        return self.embed([query])
