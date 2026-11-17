"""SQLite bookkeeping: which PDFs are in the library, and what was asked.

Analogy: the library's front-desk ledger. The books (chunks) live on the shelves
(ChromaDB); the ledger records which books we own and who asked what.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,      -- hash of the file's bytes
    filename    TEXT NOT NULL,
    pages       INTEGER NOT NULL,
    chunks      INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS qa_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer   TEXT NOT NULL,
    sources  TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    path = Path(config.db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_document(doc_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def add_document(doc_id: str, filename: str, pages: int, chunks: int) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, pages, chunks, _now()),
        )


def delete_document(doc_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC, filename")
        return [dict(r) for r in rows]


def log_qa(question: str, answer: str, sources: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO qa_log (asked_at, question, answer, sources) VALUES (?, ?, ?, ?)",
            (_now(), question, answer, sources),
        )


def recent_qa(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM qa_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
