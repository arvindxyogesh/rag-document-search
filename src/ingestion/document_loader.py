"""
Document loading module.
Loads plain-text and Markdown documents from a directory tree.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a raw document with metadata."""
    doc_id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Recursively loads .txt and .md files from the configured directory."""

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def __init__(self, config: dict):
        self.data_dir = Path(config["ingestion"]["data_dir"])

    def load(self) -> List[Document]:
        """Return all non-empty documents found in the data directory."""
        docs = []
        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file() and path.suffix in self.SUPPORTED_EXTENSIONS:
                try:
                    content = path.read_text(encoding="utf-8")
                    if content.strip():
                        docs.append(Document(
                            doc_id=str(path.relative_to(self.data_dir)),
                            content=content,
                            source=str(path),
                            metadata={"filename": path.name, "extension": path.suffix},
                        ))
                        logger.debug(f"Loaded: {path.name} ({len(content)} chars)")
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        logger.info(f"Loaded {len(docs)} documents from {self.data_dir}")
        return docs
