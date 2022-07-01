# RAG Document Search System

A production-grade **Retrieval Augmented Generation (RAG)** system for semantic document search and question answering. Built with sentence-transformers for local embeddings, FAISS for vector search, and optional OpenAI GPT integration for answer generation.

## Architecture

```
rag-document-search/
├── config/config.yaml          # Chunk size, model name, top-k, generation settings
├── data/sample_docs/           # Source documents (.txt / .md)
│   ├── ml_fundamentals.txt
│   ├── deep_learning.txt
│   └── mlops.txt
├── vector_store/               # Persisted FAISS index + chunk metadata
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  # Recursive directory loading
│   │   └── chunker.py          # Overlapping word-boundary chunking
│   ├── embeddings/
│   │   └── embedder.py         # sentence-transformers, L2-normalized
│   ├── vector_store/
│   │   └── store.py            # FAISS IndexFlatIP (cosine via dot product)
│   ├── retrieval/
│   │   └── retriever.py        # Query embed + top-k search + context formatting
│   └── generation/
│       └── generator.py        # OpenAI GPT or local template fallback
├── scripts/
│   ├── ingest.py               # One-time document ingestion pipeline
│   └── query.py                # Interactive query loop
└── tests/
    └── test_rag.py
```

## RAG Pipeline Flow

```
Documents
    │
    ▼
Chunking (overlapping windows)
    │
    ▼
Sentence-Transformer Embeddings (L2-normalized)
    │
    ▼
FAISS IndexFlatIP (exact cosine similarity)
    │
    ▼  ← Query embedding
Top-K Retrieval
    │
    ▼
Context Assembly + LLM Prompt
    │
    ▼
Answer Generation (OpenAI GPT or template)
```

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Ingest documents (run once)
python scripts/ingest.py

# Step 2: Query interactively
python scripts/query.py
```

### Enable LLM Generation (optional)

```bash
export OPENAI_API_KEY=sk-...
python scripts/query.py
```

## Example Queries

```
Query: What is the difference between supervised and unsupervised learning?
Query: How does the Transformer attention mechanism work?
Query: What is concept drift in MLOps?
Query: How do CNNs process image data?
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 150 | Words per chunk |
| `overlap` | 20 | Word overlap between adjacent chunks |
| `model_name` | all-MiniLM-L6-v2 | Sentence-transformer model (local, no API key needed) |
| `top_k` | 5 | Chunks retrieved per query |
| `generation.model` | gpt-3.5-turbo | OpenAI model (only used if API key set) |

## Adding Your Own Documents

Drop `.txt` or `.md` files into `data/sample_docs/` and re-run ingestion:

```bash
python scripts/ingest.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Technical Notes

- **L2 normalization** — embeddings are normalized before indexing so that inner product equals cosine similarity, enabling efficient exact search with `IndexFlatIP`
- **Lazy model loading** — the sentence-transformer model is downloaded on first use
- **Offline capable** — sentence-transformers run entirely locally; no API key required for retrieval
- **Persistent store** — FAISS index written to disk, avoiding re-embedding on each query session
