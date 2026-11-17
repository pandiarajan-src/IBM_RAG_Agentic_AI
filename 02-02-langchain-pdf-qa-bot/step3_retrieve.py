"""Step 3 - Retriever: find the chunks closest in meaning to a question. Run step 2 first.

    python step3_retrieve.py "What does a red light on the router mean?"
"""
import sys

from qabot import config
from qabot.store import explain_error, get_retriever, get_vectorstore

question = " ".join(sys.argv[1:]) or "How much is the late payment fee?"

try:
    # The raw vector-store call, with scores. Cosine distance: 0 = same meaning, 1 = unrelated.
    hits = get_vectorstore().similarity_search_with_score(question, k=config.top_k())
    if not hits:
        raise SystemExit("The library is empty. Run: python step2_embed_store.py")
    print(f"Question: {question}\n")
    for doc, distance in hits:
        print(f"distance {distance:.3f} | {doc.metadata['source']} page {doc.metadata['page']}")
        print(f"    {doc.page_content[:160]!r}\n")

    # A retriever is the same search behind LangChain's standard interface: question in, Documents out.
    docs = get_retriever().invoke(question)
    print(f"retriever.invoke() returned {len(docs)} Documents - the same ones, ready to plug into a chain.")
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"ERROR: {explain_error(exc)}")
