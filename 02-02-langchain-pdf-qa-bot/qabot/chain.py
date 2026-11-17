"""Retrieval + prompt + LLM = the RAG chain.

Analogy: an open-book exam. The retriever finds the right pages, the prompt puts
them on the desk, and the LLM writes the answer using only those pages.
"""
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama import ChatOllama

from . import config
from .store import get_retriever

NOT_LOADED = "No documents are loaded yet. Open the **Documents** tab and add a PDF first."

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", config.SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def get_llm() -> ChatOllama:
    # temperature 0: for factual Q&A we want the same answer every time
    return ChatOllama(model=config.chat_model(), base_url=config.ollama_host(), temperature=0)


def format_docs(docs: list[Document]) -> str:
    """Number each passage so the model can cite it as [1], [2]..."""
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', '?')}, page {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def format_sources(docs: list[Document]) -> str:
    return "\n".join(
        f"[{i}] {d.metadata.get('source', '?')}, page {d.metadata.get('page', '?')}"
        for i, d in enumerate(docs, start=1)
    )


def build_answer_chain() -> Runnable:
    """prompt | llm | parser. Input: {"context": str, "question": str}. Output: str."""
    return PROMPT | get_llm() | StrOutputParser()


def build_rag_chain(k: int | None = None) -> Runnable:
    """The whole pipeline as one LCEL expression. Input: the question. Output: str."""
    return (
        {"context": get_retriever(k) | format_docs, "question": RunnablePassthrough()}
        | build_answer_chain()
    )


def retrieve(question: str, k: int | None = None) -> list[Document]:
    return get_retriever(k).invoke(question)


def stream_answer(question: str, docs: list[Document]) -> Iterator[str]:
    """Yield the answer piece by piece. Kept separate from `retrieve` so the UI can show sources."""
    if not docs:
        yield NOT_LOADED
        return
    yield from build_answer_chain().stream({"context": format_docs(docs), "question": question})
