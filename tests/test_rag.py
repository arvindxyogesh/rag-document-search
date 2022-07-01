"""Tests for RAG pipeline components."""
import numpy as np
import pytest
import yaml

from src.ingestion.document_loader import Document, DocumentLoader
from src.ingestion.chunker import TextChunker
from src.vector_store.store import VectorStore


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def test_document_loader_loads_samples(config):
    loader = DocumentLoader(config)
    docs = loader.load()
    assert len(docs) > 0
    for doc in docs:
        assert doc.content.strip(), f"Empty document: {doc.doc_id}"
        assert doc.doc_id


def test_chunker_word_overlap(config):
    chunker = TextChunker(config)
    doc = Document(doc_id="test.txt", content=" ".join([f"word{i}" for i in range(500)]), source="test")
    chunks = chunker.chunk([doc])
    assert len(chunks) > 1
    # Verify all chunks belong to the same doc
    assert all(c.doc_id == "test.txt" for c in chunks)
    # Verify chunk indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_short_document(config):
    chunker = TextChunker(config)
    doc = Document(doc_id="short.txt", content="short document", source="short")
    chunks = chunker.chunk([doc])
    assert len(chunks) == 1


def test_vector_store_build_search(config, tmp_path):
    cfg = {**config, "vector_store": {"store_dir": str(tmp_path)}}
    store = VectorStore(cfg)

    dim = 384
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, dim)).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    chunks_meta = [
        {"chunk_id": f"c{i}", "doc_id": "test.txt", "text": f"chunk {i}", "chunk_index": i}
        for i in range(20)
    ]
    store.build(embeddings, chunks_meta)
    store.save()

    # Reload and verify search
    store2 = VectorStore(cfg)
    store2.load()
    query = rng.standard_normal((1, dim)).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store2.search(query, top_k=3)
    assert len(results) == 3
    for meta, score in results:
        assert -1.0 <= score <= 1.0  # cosine similarity range


def test_vector_store_load_missing_raises(config, tmp_path):
    cfg = {**config, "vector_store": {"store_dir": str(tmp_path / "empty")}}
    store = VectorStore(cfg)
    with pytest.raises(FileNotFoundError):
        store.load()
