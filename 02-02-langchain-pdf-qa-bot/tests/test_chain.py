"""The RAG chain: retrieved context must reach the model."""
import fake_ollama
from langchain_core.documents import Document

from qabot import config
from qabot.chain import NOT_LOADED, build_rag_chain, format_docs, format_sources, retrieve, stream_answer


def chat_calls() -> list[dict]:
    return [r for r in fake_ollama.Handler.requests_seen if r["path"] == "/api/chat"]


def test_format_docs_numbers_passages_and_shows_origin():
    docs = [Document(page_content="Alpha", metadata={"source": "a.pdf", "page": 2}), Document(page_content="Beta")]
    assert format_docs(docs) == "[1] (a.pdf, page 2)\nAlpha\n\n[2] (?, page ?)\nBeta"
    assert format_sources(docs) == "[1] a.pdf, page 2\n[2] ?, page ?"


def test_rag_chain_answers_from_the_document(loaded_library):
    answer = build_rag_chain().invoke("How much is the late payment fee?")
    assert "7.50 dollars" in answer


def test_prompt_contains_rules_context_and_question(loaded_library):
    build_rag_chain(k=2).invoke("What does a solid red light on the router mean?")
    request = chat_calls()[-1]
    system, human = request["messages"]
    assert request["model"] == "llama3.2"
    assert request["options"]["temperature"] == 0
    assert system["role"] == "system" and system["content"] == config.SYSTEM_PROMPT
    assert "[1] (acme_broadband_handbook.pdf, page 4)" in human["content"]
    assert "[3]" not in human["content"]                       # k=2 -> two passages only
    assert human["content"].endswith("Question: What does a solid red light on the router mean?")


def test_stream_answer_yields_several_pieces(loaded_library):
    question = "What are the phone support hours?"
    pieces = list(stream_answer(question, retrieve(question)))
    assert len(pieces) > 3
    assert "7 am to 10 pm" in "".join(pieces)


def test_empty_library_never_calls_the_model():
    question = "Anything?"
    docs = retrieve(question)
    assert docs == []
    assert list(stream_answer(question, docs)) == [NOT_LOADED]
    assert chat_calls() == []
