"""Embedding model + vector database + retriever.

Analogy: embeddings are GPS coordinates for meaning - passages about the same
topic land close together. ChromaDB is the map that stores the coordinates and
finds the nearest neighbours of a question.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from . import config


@lru_cache(maxsize=8)
def _vectorstore(chroma_dir: str, host: str, model: str) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=OllamaEmbeddings(model=model, base_url=host),
        persist_directory=chroma_dir,                               # saved to disk
        collection_configuration={"hnsw": {"space": "cosine"}},     # compare by angle
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.embed_model(), base_url=config.ollama_host())


def get_vectorstore() -> Chroma:
    """One shared store per (folder, Ollama host, embedding model)."""
    return _vectorstore(config.chroma_dir(), config.ollama_host(), config.embed_model())


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": k or config.top_k()})


def explain_error(exc: Exception) -> str:
    """Turn a stack trace into a message a human can act on."""
    text = str(exc)
    low = text.lower()
    if isinstance(exc, ConnectionError) or "connect" in low:
        return f"Cannot reach Ollama at {config.ollama_host()}. Start the Ollama app or run `ollama serve`."
    if "not found" in low:
        return (
            f"A model is not installed. Run `ollama pull {config.chat_model()}` "
            f"and `ollama pull {config.embed_model()}`."
        )
    return text or exc.__class__.__name__
