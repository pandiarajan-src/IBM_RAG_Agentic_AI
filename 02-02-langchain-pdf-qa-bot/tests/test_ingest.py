"""Ingestion + retrieval against the fake Ollama."""
import fake_ollama
import pytest

from qabot import db
from qabot.ingest import ingest_pdf, remove_document
from qabot.store import explain_error, get_retriever, get_vectorstore


def embed_calls() -> list[dict]:
    return [r for r in fake_ollama.Handler.requests_seen if r["path"] == "/api/embed"]


def test_ingest_stores_every_chunk_and_records_the_file(sample_pdf):
    result = ingest_pdf(sample_pdf)
    assert (result.pages, result.skipped) == (6, False)
    assert get_vectorstore()._collection.count() == result.chunks
    assert db.list_documents()[0]["filename"] == "acme_broadband_handbook.pdf"
    assert embed_calls()[0]["model"] == "nomic-embed-text"


def test_ingesting_the_same_file_twice_is_a_no_op(sample_pdf, tmp_path):
    first = ingest_pdf(sample_pdf)
    calls = len(embed_calls())
    renamed = tmp_path / "copy.pdf"
    renamed.write_bytes(sample_pdf.read_bytes())

    second = ingest_pdf(renamed)

    assert second.skipped
    assert len(embed_calls()) == calls                         # nothing re-embedded
    assert get_vectorstore()._collection.count() == first.chunks


def test_scanned_pdf_with_no_text_is_rejected(tmp_path):
    from pypdf import PdfWriter

    blank = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(blank)
    with pytest.raises(ValueError, match="no text found"):
        ingest_pdf(blank)
    assert db.list_documents() == []


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How much is the late payment fee?", "late payment fee of 7.50 dollars"),
        ("What does a solid red light on the router mean?", "no internet signal"),
        ("What are the phone support hours?", "7 am to 10 pm"),
    ],
)
def test_retriever_finds_the_relevant_passage(loaded_library, question, expected):
    docs = get_retriever(k=4).invoke(question)
    assert len(docs) == 4
    assert any(expected in d.page_content.replace("\n", " ") for d in docs)
    assert all(d.metadata["source"] == "acme_broadband_handbook.pdf" for d in docs)


def test_remove_document_deletes_chunks_and_record(loaded_library):
    doc_id = db.list_documents()[0]["doc_id"]
    assert remove_document(doc_id) is True
    assert get_vectorstore()._collection.count() == 0
    assert db.list_documents() == []
    assert remove_document(doc_id) is False


def test_missing_embedding_model_gives_a_helpful_message(sample_pdf, monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "missing-model")
    with pytest.raises(Exception) as err:
        ingest_pdf(sample_pdf)
    assert "ollama pull" in explain_error(err.value)
    assert db.list_documents() == []                           # a failed ingest leaves no record


def test_ollama_down_gives_a_helpful_message(sample_pdf, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")    # nothing listens here
    with pytest.raises(Exception) as err:
        ingest_pdf(sample_pdf)
    assert "Cannot reach Ollama" in explain_error(err.value)
