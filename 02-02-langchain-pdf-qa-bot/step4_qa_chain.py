"""Step 4 - The RAG chain: retriever + prompt + LLM, in the terminal. Run step 2 first.

    python step4_qa_chain.py "Can I get my money back?"
"""
import sys

from qabot.chain import PROMPT, build_rag_chain, format_docs, retrieve
from qabot.store import explain_error

question = " ".join(sys.argv[1:]) or "How much is the late payment fee?"

try:
    # What the model actually receives: instructions + retrieved passages + the question.
    docs = retrieve(question)
    prompt = PROMPT.invoke({"context": format_docs(docs), "question": question})
    print("=== PROMPT SENT TO THE MODEL ===")
    for message in prompt.messages:
        print(f"[{message.type}] {message.content}\n")

    # The whole pipeline is one expression (see build_rag_chain):
    #   {"context": retriever | format_docs, "question": passthrough} | prompt | llm | parser
    print("=== ANSWER (streamed) ===")
    for piece in build_rag_chain().stream(question):
        print(piece, end="", flush=True)
    print()
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"ERROR: {explain_error(exc)}")
