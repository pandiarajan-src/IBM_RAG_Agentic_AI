"""
04_embeddings_vectorstores_rag.py - Embeddings, vector stores, retrievers and RAG (+ Exercise 4)
===============================================================================================

LEARNING NOTES
--------------
This script finishes the RAG pipeline started in script 03:

    chunks --embed--> vectors --store--> Chroma --search--> relevant chunks --stuff into prompt--> Claude --> answer
                                                  ^ the question is embedded the same way

Needs BOTH keys in .env: ANTHROPIC_API_KEY (answers) and OPENROUTER_API_KEY (embeddings).

Embeddings
----------
An embedding model turns text into a vector (a list of floats). Texts with similar
MEANING get vectors that point in similar directions, so "how close are these
vectors?" becomes "how related are these texts?". That is semantic search: it
matches meaning, not keywords.
- `embed_documents(list[str])` : embed many chunks (for indexing)
- `embed_query(str)`           : embed one question (at search time)
Always use the SAME embedding model for indexing and querying.
Claude has no embedding endpoint, so we use OpenRouter (see llm_setup.py).

Vector stores (langchain_chroma.Chroma)
---------------------------------------
A vector store saves (vector, text, metadata) and finds the nearest vectors to a
query vector. `Chroma.from_documents(chunks, embeddings)` embeds and stores in
one call; `similarity_search(query, k=4)` embeds the query and returns the top-k
Documents. Without `persist_directory`, Chroma lives in memory and is gone when
the script ends.

Retrievers
----------
A retriever is anything that takes a string and returns `list[Document]`. It's a
Runnable, so it can sit inside an LCEL chain. A vector store becomes one with
`vectorstore.as_retriever(search_type=..., search_kwargs={"k": 3})`:
- search_type="similarity" : top-k most similar chunks (default)
- search_type="mmr"        : Maximal Marginal Relevance, relevant but DIVERSE
                             chunks (fewer near-duplicates)

Parent document retrieval
-------------------------
The chunk-size dilemma: SMALL chunks give precise embeddings, LARGE chunks give
the LLM enough context. The fix: search over small "child" chunks, but return the
large "parent" chunk each child came from. We build it by hand (about 15 lines)
so you can see exactly how it works.

RAG chain (replaces RetrievalQA)
--------------------------------
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
The dict runs both branches in parallel on the same input question. That's
the "stuff" strategy: concatenate the retrieved chunks into the prompt.

Changes from the notebook (LangChain 0.2 -> 1.x)
------------------------------------------------
| Notebook                                   | Now                                              |
|--------------------------------------------|--------------------------------------------------|
| WatsonxEmbeddings (IBM slate)              | OpenAIEmbeddings -> OpenRouter (qwen3-embedding) |
| from langchain.vectorstores import Chroma  | from langchain_chroma import Chroma              |
| retriever.get_relevant_documents(q)        | retriever.invoke(q)                              |
| ParentDocumentRetriever + langchain.storage| hand-built with InMemoryStore (langchain_core)   |
| RetrievalQA.from_chain_type(...)           | LCEL RAG chain                                   |
(The legacy classes still exist in the separate `langchain-classic` package.)

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/04_embeddings_vectorstores_rag.py
"""

import math
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from llm_setup import PAPER_URL, WEB_URL, get_chat_model, get_embeddings, print_header

embeddings = get_embeddings()
llm = get_chat_model(max_tokens=1024)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ---------------------------------------------------------------------------
# 1. Embedding models
# ---------------------------------------------------------------------------
def embeddings_demo(chunks: list[Document]) -> None:
    print_header("1. EMBEDDINGS")
    texts = [chunk.page_content for chunk in chunks[:5]]
    vectors = embeddings.embed_documents(texts)
    print(f"Embedded {len(vectors)} chunks; each vector has {len(vectors[0])} dimensions")
    print(f"First 5 numbers of chunk 0: {vectors[0][:5]}")

    # See semantic similarity at work: same meaning scores high, even with no shared words.
    query = embeddings.embed_query("How do I cook pasta?")
    for sentence in ["Boil noodles in salted water for ten minutes.", "LangChain connects LLMs to external data."]:
        score = cosine_similarity(query, embeddings.embed_query(sentence))
        print(f"  similarity('How do I cook pasta?', {sentence!r}) = {score:.3f}")


# ---------------------------------------------------------------------------
# 2. Vector store + similarity search
# ---------------------------------------------------------------------------
def vector_store_demo(chunks: list[Document]) -> Chroma:
    print_header("2. VECTOR STORE (Chroma)")
    # Each Chroma store in this script gets its own collection name; in-memory
    # stores with the same name would share (and mix) their contents.
    docsearch = Chroma.from_documents(chunks, embeddings, collection_name="paper_chunks")
    docs = docsearch.similarity_search("Langchain")
    print(f"Top match out of {len(chunks)} chunks:\n{docs[0].page_content}")
    print(f"metadata: page={docs[0].metadata.get('page')}")
    return docsearch


# ---------------------------------------------------------------------------
# 3. Vector store-backed retriever
# ---------------------------------------------------------------------------
def retriever_demo(docsearch: Chroma) -> None:
    print_header("3. VECTOR STORE-BACKED RETRIEVER")
    retriever = docsearch.as_retriever()
    docs = retriever.invoke("Langchain")
    print(f"similarity retriever -> {len(docs)} docs; first one matches the similarity_search result:")
    print(docs[0].page_content[:200])

    mmr_retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    print("\nMMR retriever (relevant AND diverse), first 80 chars of each:")
    for doc in mmr_retriever.invoke("Langchain"):
        print(f"  - {doc.page_content[:80]!r}")


# ---------------------------------------------------------------------------
# 4. Parent document retrieval (built by hand)
# ---------------------------------------------------------------------------
def parent_document_retriever_demo(document: list[Document]) -> None:
    """Search with SMALL chunks, but hand the LLM the BIGGER chunk they came from.

    The problem this solves (a chunk-size trade-off):
      * Small chunks (~400 chars) embed well: one chunk ~ one idea, so the vector
        is "sharp" and matches a query precisely.
      * But a small chunk is poor context for an LLM: the surrounding sentences
        that explain it are cut off.
      * Large chunks (~2000 chars) give the LLM great context, but they mix many
        ideas into one vector, so similarity search gets blurry.
    Parent document retrieval gives us both: embed small "child" chunks for
    searching, then return the large "parent" chunk each matching child belongs to.
    Note the child text is a slice of the parent text; we are not storing the same
    thing twice for no reason. Each copy has a different job (find vs. read).
    """
    print_header("4. PARENT DOCUMENT RETRIEVAL")
    # Parents: big chunks, returned to the LLM as context (never embedded).
    parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator="\n")
    # Children: small chunks, embedded and searched (never returned directly).
    child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator="\n")

    # Two different stores, because the two kinds of chunk do different jobs:
    # vector store = good at "find similar" but only for things that were embedded.
    vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings)  # children: embedded
    # key-value store = good at "give me item with this id"; no embeddings needed.
    store = InMemoryStore()  # parents: stored by id, NOT embedded

    child_docs = []
    # Step 1: cut the document into big parents.
    for parent in parent_splitter.split_documents(document):
        # Give each parent a unique id so children can point back to it.
        parent_id = str(uuid4())
        # Step 2: save the parent in the key-value store under that id.
        store.mset([(parent_id, parent)])
        # Step 3: cut THIS parent into small children (so every child lies inside exactly one parent).
        for child in child_splitter.split_documents([parent]):
            # Step 4: stamp the child with its parent's id. This metadata is the
            # only link between the two stores; without it we could not go from
            # a matching child back to its parent.
            child.metadata["parent_id"] = parent_id  # the link back to the parent
            child_docs.append(child)
    # Step 5: embed ONLY the children into the vector store (parents are not embedded).
    vectorstore.add_documents(child_docs)
    print(f"{len(list(store.yield_keys()))} parent chunks, {len(child_docs)} child chunks")

    def retrieve_parents(query: str) -> list[Document]:
        # Search happens on the precise small chunks...
        children = vectorstore.similarity_search(query, k=4)
        # ...then we translate children -> parents. Several children can share a
        # parent; dict.fromkeys de-duplicates while keeping rank order, so the
        # LLM does not receive the same big chunk twice.
        parent_ids = list(dict.fromkeys(child.metadata["parent_id"] for child in children))
        # Fetch the big parents by id (mget returns None for unknown ids, so filter those).
        return [doc for doc in store.mget(parent_ids) if doc is not None]

    # Wrapping it in RunnableLambda makes it a Runnable, usable in chains like any retriever.
    retriever = RunnableLambda(retrieve_parents)

    # Compare what plain search returns (child) with what our retriever returns (parent).
    sub_docs = vectorstore.similarity_search("Langchain")
    print(f"\nsmall child chunk ({len(sub_docs[0].page_content)} chars):\n{sub_docs[0].page_content}")
    retrieved_docs = retriever.invoke("Langchain")
    print(f"\nits large parent chunk ({len(retrieved_docs[0].page_content)} chars), first 600:\n{retrieved_docs[0].page_content[:600]}")


# ---------------------------------------------------------------------------
# 5. Question answering over the paper (RetrievalQA -> LCEL RAG chain)
# ---------------------------------------------------------------------------
# The prompt is a template with two blanks, {context} and {question}.
# It is not filled here; the chain below supplies both values at run time.
# "Use only the context" keeps the LLM grounded in the retrieved text, and the
# "say you don't know" instruction reduces made-up answers (hallucination).
RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

<context>
{context}
</context>

Question: {question}"""
)


def format_docs(docs: list[Document]) -> str:
    # A retriever returns a LIST of Documents, but the prompt's {context} blank
    # needs ONE string. Join each chunk's text, separated by blank lines.
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """Build: question (str) -> retrieve -> fill prompt -> LLM -> answer (str).

    The chain is called with just the question string, e.g. chain.invoke("What is X?").
    `|` pipes the output of each step into the next step.

    Step 1 - the dict `{"context": ..., "question": ...}`:
        A dict inside a chain is auto-converted to a RunnableParallel. It sends
        the SAME input (the question string) to each value, runs them, and
        returns a dict with the same keys:
            input "What is X?"  ->  {"context": <string>, "question": "What is X?"}
        * "context": retriever | format_docs
              retriever.invoke(question) -> [Document, Document, ...]  (similarity search)
              format_docs(...)           -> one big string of their text
        * "question": RunnablePassthrough()
              Returns its input unchanged, so the original question is kept
              and can fill the {question} blank.
    Step 2 - RAG_PROMPT:
        Takes that dict and fills {context} and {question} by key name (so the
        dict keys must match the blanks in the template). The result is the
        full prompt (chat messages) with the retrieved text embedded in it.
    Step 3 - llm:
        Receives the filled prompt and generates a reply (an AIMessage).
    Step 4 - StrOutputParser():
        Pulls the plain text out of the AIMessage so the caller gets a str.
    """
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def rag_demo(docsearch: Chroma) -> None:
    print_header("5. RETRIEVAL QA (LCEL RAG CHAIN)")
    qa = build_rag_chain(docsearch.as_retriever())
    query = "what is this paper discussing?"
    print(f"Q: {query}\nA: {qa.invoke(query)}")


# ---------------------------------------------------------------------------
# 6. Exercise 4 - a simple retrieval system over a web page
# ---------------------------------------------------------------------------
def exercise_4_retrieval_system() -> None:
    print_header("6. EXERCISE 4: SIMPLE RETRIEVAL SYSTEM")
    # 1. Load  2. Split  3. Embed + store  4. Retriever
    documents = WebBaseLoader(WEB_URL).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(chunks, embeddings, collection_name="langchain_docs")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(f"Indexed {len(chunks)} chunks from {WEB_URL}")

    def search_documents(query: str, top_k: int = 3) -> list[Document]:
        """Search for documents relevant to a query."""
        return retriever.invoke(query)[:top_k]

    qa = build_rag_chain(retriever)
    for query in ["What is LangChain?", "How do retrievers work?", "Why is document splitting important?"]:
        print(f"\nQuery: {query}")
        for i, doc in enumerate(search_documents(query), start=1):
            print(f"  Result {i}: {' '.join(doc.page_content.split())[:150]}...")
        # Bonus: turn retrieval into an answer. Note the "don't know" path: the page
        # may not cover a question, and the prompt tells Claude not to make things up.
        print(f"  Answer: {qa.invoke(query)}")


def main() -> None:
    document = PyPDFLoader(PAPER_URL).load()
    chunks = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n").split_documents(document)

    embeddings_demo(chunks)
    docsearch = vector_store_demo(chunks)
    retriever_demo(docsearch)
    parent_document_retriever_demo(document)
    rag_demo(docsearch)
    exercise_4_retrieval_system()


if __name__ == "__main__":
    main()
