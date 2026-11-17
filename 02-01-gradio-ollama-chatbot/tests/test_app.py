"""Tests the Gradio layer: handlers directly, then the real app over HTTP."""
import gradio as gr
import pytest
from gradio_client import Client

import fake_ollama
import step1_interface
import step2_blocks
import step3_chat
from supportbot import db


def test_step1_returns_full_answer():
    assert step1_interface.answer("help") == fake_ollama.FULL_REPLY


def test_step2_rejects_empty_question():
    with pytest.raises(gr.Error):
        next(step2_blocks.answer("   ", 0.3))


def test_step3_streams_then_logs_to_sqlite():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out = list(step3_chat.chat("my wifi is down", history, "be brief", "llama3.2", 0.2, None))
    assert out[-1] == fake_ollama.FULL_REPLY
    sent = fake_ollama.Handler.requests_seen[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]  # history was forwarded
    assert db.recent_turns()[0][3:] == ["my wifi is down", fake_ollama.FULL_REPLY]


def test_step3_turns_llm_failure_into_ui_error():
    with pytest.raises(gr.Error, match="ollama pull"):
        list(step3_chat.chat("q", [], "", "missing-model", 0.2, None))
    assert db.recent_turns() == []  # failed turns are not logged


def test_end_to_end_over_http():
    """Launch the real app and talk to it like a browser would."""
    demo = step3_chat.build_demo()
    _, url, _ = demo.launch(prevent_thread_lock=True, server_port=None, quiet=True)
    try:
        reply = Client(url, verbose=False).predict(
            "my wifi is down", "be brief", "llama3.2", 0.2, api_name="/chat"
        )
        assert reply == fake_ollama.FULL_REPLY
        assert db.recent_turns()[0][3] == "my wifi is down"
    finally:
        demo.close()
