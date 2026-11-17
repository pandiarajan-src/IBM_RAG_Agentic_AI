"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> ChromaDB.

Analogy: cutting a manual into index cards. Each card is small enough to read at
a glance and repeats a line from the previous card so no sentence is cut in half.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config, db
from .loader import PdfLoader
from .store import get_vectorstore


@dataclass
class IngestResult:
    filename: str
    pages: int
    chunks: int
    skipped: bool = False     # True when this exact file was already in the library


def file_id(path: str | Path) -> str:
    """Fingerprint of the file's bytes: same content -> same id, whatever the name."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def split(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size(),
        chunk_overlap=config.chunk_overlap(),
        add_start_index=True,     # remember where in the page each chunk started
    )
    return splitter.split_documents(pages)


def ingest_pdf(path: str | Path, filename: str | None = None) -> IngestResult:
    path = Path(path)
    name = filename or path.name
    doc_id = file_id(path)

    known = db.find_document(doc_id)
    if known:  # idempotent: uploading the same file twice does nothing
        return IngestResult(known["filename"], known["pages"], known["chunks"], skipped=True)

    pages = PdfLoader(path, source_name=name).load()
    if not pages:
        raise ValueError(f"{name}: no text found. Scanned PDFs need OCR first (see PRODUCTION.md).")

    chunks = split(pages)
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]   # stable ids -> safe to retry

    get_vectorstore().add_documents(chunks, ids=ids)      # embeds each chunk, then stores it
    db.add_document(doc_id, name, len(pages), len(chunks))
    return IngestResult(name, len(pages), len(chunks))


def remove_document(doc_id: str) -> bool:
    known = db.find_document(doc_id)
    if not known:
        return False
    get_vectorstore().delete(ids=[f"{doc_id}-{i}" for i in range(known["chunks"])])
    db.delete_document(doc_id)
    return True
