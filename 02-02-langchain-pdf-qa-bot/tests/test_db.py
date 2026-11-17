"""SQLite ledger."""
from qabot import db


def test_document_registry_round_trip():
    assert db.list_documents() == []
    db.add_document("abc", "manual.pdf", pages=3, chunks=9)
    assert db.find_document("abc")["filename"] == "manual.pdf"
    assert [d["doc_id"] for d in db.list_documents()] == ["abc"]
    db.delete_document("abc")
    assert db.find_document("abc") is None


def test_qa_log_returns_newest_first():
    db.log_qa("first?", "one", "[1] a.pdf, page 1")
    db.log_qa("second?", "two", "")
    assert [r["question"] for r in db.recent_qa()] == ["second?", "first?"]
    assert len(db.recent_qa(limit=1)) == 1
