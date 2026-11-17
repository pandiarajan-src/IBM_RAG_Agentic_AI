"""SQLite conversation log.

Analogy: the call-centre logbook. Every question and answer is written down
so a supervisor can review quality later.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    session_id  TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    path = Path(config.db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def log_turn(session_id: str, model: str, question: str, answer: str) -> None:
    # A new connection per call keeps this safe across Gradio's worker threads.
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO chat_log (session_id, model, question, answer) VALUES (?, ?, ?, ?)",
            (session_id, model, question, answer),  # parameters, never f-strings: no SQL injection
        )


def recent_turns(limit: int = 20) -> list[list]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT created_at, session_id, model, question, answer "
            "FROM chat_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [list(r) for r in rows]
