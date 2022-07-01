"""
Text chunking module.
Splits documents into overlapping word-boundary chunks for embedding.
Overlap preserves context across chunk boundaries.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from src.ingestion.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A sub-segment of a document ready for embedding."""
    chunk_id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Splits documents into fixed-size overlapping word chunks."""

    def __init__(self, config: dict):
        self.chunk_size = config["chunking"]["chunk_size"]   # words per chunk
        self.overlap = config["chunking"]["overlap"]          # word overlap

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Chunk all documents and return a flat list of Chunk objects."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self._chunk_document(doc))
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        words = doc.content.split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.overlap)
        chunks = []
        idx = 0
        pos = 0

        while pos < len(words):
            end = min(pos + self.chunk_size, len(words))
            text = " ".join(words[pos:end])
            if text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::chunk_{idx}",
                    text=text,
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "word_start": pos, "word_end": end},
                ))
                idx += 1
            pos += step
            if end >= len(words):
                break

        return chunks
