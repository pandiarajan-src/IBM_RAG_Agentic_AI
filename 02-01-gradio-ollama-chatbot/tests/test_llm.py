import pytest

import fake_ollama
from supportbot import llm


def test_build_messages_orders_system_history_user():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},  # Gradio block format
    ]
    msgs = llm.build_messages("be nice", history, "help")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "hello"
    assert msgs[-1]["content"] == "help"


def test_build_messages_skips_empty_system_prompt():
    assert llm.build_messages("  ", [], "q") == [{"role": "user", "content": "q"}]


def test_stream_chat_yields_growing_text():
    chunks = list(llm.stream_chat([{"role": "user", "content": "q"}]))
    assert chunks[0] == "Please "
    assert chunks[-1] == fake_ollama.FULL_REPLY
    assert all(later.startswith(earlier) for earlier, later in zip(chunks, chunks[1:]))


def test_temperature_and_model_reach_ollama():
    llm.ask("q", model="llama3.2", temperature=0.9)
    sent = fake_ollama.Handler.requests_seen[-1]
    assert sent["model"] == "llama3.2"
    assert sent["options"]["temperature"] == 0.9
    assert sent["messages"][0]["role"] == "system"


def test_missing_model_gives_actionable_error():
    with pytest.raises(llm.LLMUnavailable, match="ollama pull missing-model"):
        llm.ask("q", model="missing-model")


def test_ollama_down_gives_actionable_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    with pytest.raises(llm.LLMUnavailable, match="Cannot reach Ollama"):
        llm.ask("q")


def test_list_models_falls_back_when_ollama_down(monkeypatch):
    assert llm.list_models() == ["llama3.2:latest"]
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    assert llm.list_models() == ["llama3.2"]
