"""The Gradio app: handlers directly, then end to end over HTTP like a browser."""
import gradio as gr
import pytest

import app
from qabot import db


def last(generator):
    *_, final = generator
    return final


def test_chat_streams_then_appends_sources_and_logs(loaded_library):
    updates = list(app.chat("How much is the late payment fee?", [], 4))
    assert len(updates) > 3                                    # it streamed
    assert "7.50 dollars" in updates[-1]
    assert "**Sources**\n[1] acme_broadband_handbook.pdf, page 3" in updates[-1]
    row = db.recent_qa()[0]
    assert row["question"] == "How much is the late payment fee?"
    assert "Sources" not in row["answer"]


def test_chat_with_empty_library_tells_the_user_what_to_do():
    assert "Documents" in last(app.chat("Hello?", [], 4))


def test_chat_turns_a_missing_model_into_a_toast(loaded_library, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "missing-model")
    with pytest.raises(gr.Error, match="ollama pull"):
        list(app.chat("How much is the late payment fee?", [], 4))


def test_ingest_reports_each_file_and_survives_a_bad_one(sample_pdf, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_text("not a pdf")
    status, rows = app.ingest([str(sample_pdf), str(bad)])
    assert "**acme_broadband_handbook.pdf**: added (6 pages" in status
    assert "broken.pdf: not a readable PDF" in status
    assert [r[0] for r in rows] == ["acme_broadband_handbook.pdf"]

    status, _ = app.ingest([str(sample_pdf)])
    assert "already in the library" in status


def test_ingest_without_files_is_an_error():
    with pytest.raises(gr.Error, match="Choose at least one PDF"):
        app.ingest(None)


def test_end_to_end_over_http(sample_pdf):
    """Launch the real app, upload a PDF and ask a question through the public API."""
    from gradio_client import Client, handle_file

    _, url, _ = app.demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        client = Client(url, verbose=False)
        status, table = client.predict([handle_file(str(sample_pdf))], api_name="/ingest")
        assert "added (6 pages" in status
        assert table["data"][0][0] == "acme_broadband_handbook.pdf"

        answer = client.predict("What does a solid red light on the router mean?", 4, api_name="/chat")
        assert "no internet signal" in answer
        assert "Sources" in answer

        log = client.predict(api_name="/recent_questions")
        assert log["data"][0][1] == "What does a solid red light on the router mean?"
    finally:
        app.demo.close()
