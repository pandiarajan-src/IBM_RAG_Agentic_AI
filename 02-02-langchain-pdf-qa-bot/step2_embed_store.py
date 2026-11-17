"""Step 2 - Embedding model + vector database. Needs Ollama (or the simulator).

    python step2_embed_store.py
"""
import math

from qabot import config
from qabot.ingest import ingest_pdf
from qabot.store import explain_error, get_embeddings, get_vectorstore


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


try:
    # 1. EMBED: text in, list of numbers out.
    embeddings = get_embeddings()
    question = "How much is the late payment fee?"
    close = "A late payment fee of 7.50 dollars is added to the account."
    far = "A solid green light means the router is online."
    q, c, f = (embeddings.embed_query(t) for t in (question, close, far))
    print(f"Model {config.embed_model()} turns text into {len(q)} numbers. First 5: {[round(x, 3) for x in q[:5]]}")
    print(f"Similarity to the billing sentence: {cosine(q, c):.3f}")
    print(f"Similarity to the router sentence:  {cosine(q, f):.3f}   <- lower: different topic\n")

    # 2. STORE: embed every chunk of the handbook and save it in ChromaDB (on disk).
    result = ingest_pdf("sample_docs/acme_broadband_handbook.pdf")
    state = "already in the library" if result.skipped else "ingested"
    print(f"{result.filename}: {state} ({result.pages} pages, {result.chunks} chunks)")
    print(f"ChromaDB at {config.chroma_dir()} now holds {get_vectorstore()._collection.count()} chunks.")
    print("Run this script again: nothing is re-embedded, because the file's fingerprint is in SQLite.")
except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a stack trace
    raise SystemExit(f"ERROR: {explain_error(exc)}")
