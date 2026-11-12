"""
04_embeddings_vectorstores_rag.py - Embeddings, vector stores, retrievers and RAG (+ Exercise 4)
===============================================================================================

WHAT THIS SCRIPT DOES
---------------------
An LLM (like Claude) only knows what it was trained on. It has never seen your
private PDF or last week's web page, and it can't read a 100-page document in one
go. RAG (Retrieval-Augmented Generation) solves this in two moves:
    1. RETRIEVE: find the few passages of your documents most relevant to the question.
    2. GENERATE: paste those passages into the prompt and ask the LLM to answer from them.

This script builds that whole pipeline step by step, on a research paper (PDF) and
on a web page, and prints the result of every step so you can see what happens:

    PDF --load--> pages --split--> chunks --embed--> vectors --store--> Chroma (vector database)
                                                                             |
    question --embed--> query vector --nearest-neighbour search--------------+
                                                                             v
                      top-k relevant chunks --pasted into a prompt--> Claude --> answer

(Loading and splitting documents are done at the top of main(); everything after
that is what this file is about.)

HOW TO READ THIS FILE
---------------------
Read it top to bottom; main() at the very bottom is the table of contents, and
each numbered section is one idea that builds on the previous one:
    1. Embeddings           - turn text into numbers that capture meaning
    2. Vector store         - save those numbers and search them
    3. Retrievers           - wrap search in a reusable "question in, documents out" component
    4. Parent-doc retrieval - a trick to get precise search AND rich context
    5. RAG chain            - connect retrieval to the LLM to answer questions
    6. Exercise 4           - the full pipeline again, end to end, on a web page

CONCEPTS YOU WILL LEARN
-----------------------
* Embedding         : a list of numbers (a vector) that represents the MEANING of a text.
* Cosine similarity : a score from about -1 to 1 for how closely two vectors point the
                      same way; higher means more similar meaning.
* Semantic search   : searching by meaning instead of exact keywords.
* Vector store      : a database specialised in "find the vectors closest to this one".
* Retriever         : a component that takes a question string and returns relevant Documents.
* Chunking trade-off: small chunks search precisely, big chunks give the LLM more context.
* Prompt grounding  : telling the LLM to answer ONLY from supplied text to reduce made-up answers.
* LCEL chains       : building a pipeline with the `|` operator (like Unix pipes).

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

Modern LangChain choices used here (you may see older tutorials do it differently)
-------------------------------------------------------------------------------
* Embeddings : OpenAIEmbeddings pointed at OpenRouter (qwen3-embedding model).
* Chroma     : imported from the dedicated `langchain_chroma` package.
* Retrieval  : `retriever.invoke(question)` (older code used `get_relevant_documents`).
* Parent-doc : built by hand with `InMemoryStore` so every step is visible.
* QA chain   : an LCEL chain (`prompt | llm | parser`) instead of the old `RetrievalQA` class.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/04_embeddings_vectorstores_rag.py
"""

import math
from uuid import uuid4  # uuid4() makes a random unique id string, used to label parent chunks

from langchain_chroma import Chroma  # Chroma: an open-source vector database
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader  # read a PDF / a web page into Documents
from langchain_core.documents import Document  # LangChain's basic unit of text: page_content (str) + metadata (dict)
from langchain_core.output_parsers import StrOutputParser  # turns the LLM's reply message into a plain str
from langchain_core.prompts import ChatPromptTemplate  # a prompt with {blanks} to fill in later
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # building blocks for chains
from langchain_core.stores import InMemoryStore  # a simple id -> object dictionary kept in RAM
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter  # cut long text into chunks

# Shared helpers from llm_setup.py in this folder: example URLs, model factories, and a banner printer.
from llm_setup import PAPER_URL, WEB_URL, get_chat_model, get_embeddings, print_header

# Created once at module level and reused by every function below.
# `embeddings`: the model that converts text -> vector (used for both indexing and searching).
embeddings = get_embeddings()
# `llm`: Claude, used only to write answers. max_tokens caps the reply length.
llm = get_chat_model(max_tokens=1024)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar are two vectors in direction? 1 = same meaning, ~0 = unrelated.

    Formula: (a . b) / (|a| * |b|)
      - numerator   : the "dot product" (multiply matching positions, then add up)
      - denominator : the lengths of the vectors, so only direction matters, not size
    Vector stores do this kind of maths for you; we write it out so it isn't magic.
    """
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ---------------------------------------------------------------------------
# 1. Embedding models
# ---------------------------------------------------------------------------
def embeddings_demo(chunks: list[Document]) -> None:
    print_header("1. EMBEDDINGS")
    # Goal: see that an embedding is just a long list of numbers, and that closeness = similar meaning.
    # A Document holds its text in `.page_content`; the embedding model wants plain strings.
    # We only embed 5 chunks to keep the demo fast (embedding calls cost time/money).
    texts = [chunk.page_content for chunk in chunks[:5]]
    # One API call turns all 5 texts into 5 vectors (a list of lists of floats).
    vectors = embeddings.embed_documents(texts)
    # "Dimensions" = how many numbers describe each text. Every vector has the same length.
    print(f"Embedded {len(vectors)} chunks; each vector has {len(vectors[0])} dimensions")
    print(f"First 5 numbers of chunk 0: {vectors[0][:5]}")  # individual numbers mean nothing to a human; only comparisons do

    # See semantic similarity at work: same meaning scores high, even with no shared words.
    # embed_query is for ONE search question (embed_documents is for many texts to index).
    query = embeddings.embed_query("How do I cook pasta?")
    # Expect the cooking sentence to score much higher than the LangChain sentence, even though
    # it shares almost no words with the question ("noodles" ~ "pasta", "boil" ~ "cook").
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
    # from_documents does three things at once: embed every chunk, then store (vector + text + metadata).
    # "collection" is like a table name inside the database.
    docsearch = Chroma.from_documents(chunks, embeddings, collection_name="paper_chunks")
    # similarity_search: embed the query, find the k nearest chunk vectors (default k=4),
    # and return them as Documents ordered best-first. Note it matches meaning, not just the word.
    docs = docsearch.similarity_search("Langchain")
    print(f"Top match out of {len(chunks)} chunks:\n{docs[0].page_content}")
    # Metadata travels with each chunk from the loader (e.g. which PDF page it came from),
    # which lets an app cite its sources.
    print(f"metadata: page={docs[0].metadata.get('page')}")
    return docsearch  # returned so later sections reuse it instead of paying to embed again


# ---------------------------------------------------------------------------
# 3. Vector store-backed retriever
# ---------------------------------------------------------------------------
def retriever_demo(docsearch: Chroma) -> None:
    print_header("3. VECTOR STORE-BACKED RETRIEVER")
    # A retriever is a thin wrapper with one job: question string in, list of Documents out.
    # Why bother, when similarity_search exists? Because a retriever is a standard "Runnable",
    # so it plugs straight into chains (see section 5), and other retrievers can be swapped in.
    retriever = docsearch.as_retriever()
    docs = retriever.invoke("Langchain")  # .invoke(input) is how every Runnable is called
    print(f"similarity retriever -> {len(docs)} docs; first one matches the similarity_search result:")
    print(docs[0].page_content[:200])

    # Plain similarity can return 4 near-identical chunks (wasteful for the LLM's limited prompt space).
    # MMR (Maximal Marginal Relevance) picks each next chunk to be relevant to the query
    # but different from the ones already chosen. k=3 means "return 3 chunks".
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
    # Retriever (from the vector store) feeds the chain; the chain is then called like a function.
    qa = build_rag_chain(docsearch.as_retriever())
    # Behind this one invoke: embed question -> fetch top chunks -> fill prompt -> Claude answers.
    query = "what is this paper discussing?"
    print(f"Q: {query}\nA: {qa.invoke(query)}")


# ---------------------------------------------------------------------------
# 6. Exercise 4 - a simple retrieval system over a web page
# ---------------------------------------------------------------------------
def exercise_4_retrieval_system() -> None:
    print_header("6. EXERCISE 4: SIMPLE RETRIEVAL SYSTEM")
    # 1. Load  2. Split  3. Embed + store  4. Retriever
    # Load: download the page and extract its text into Document(s).
    documents = WebBaseLoader(WEB_URL).load()
    # Split: the Recursive splitter tries to break at paragraph, then line, then sentence, then word
    # boundaries, so chunks stay readable. chunk_size is max characters per chunk; chunk_overlap
    # repeats 50 characters between neighbours so a sentence cut at a boundary isn't lost.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    # Embed + store, then expose as a retriever that returns the 3 best chunks per question.
    vector_store = Chroma.from_documents(chunks, embeddings, collection_name="langchain_docs")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(f"Indexed {len(chunks)} chunks from {WEB_URL}")

    def search_documents(query: str, top_k: int = 3) -> list[Document]:
        """Search for documents relevant to a query."""
        # The retriever already returns k=3; [:top_k] lets a caller ask for fewer.
        return retriever.invoke(query)[:top_k]

    # Reuse the same RAG chain from section 5; only the retriever (the knowledge source) changed.
    qa = build_rag_chain(retriever)
    for query in ["What is LangChain?", "How do retrievers work?", "Why is document splitting important?"]:
        print(f"\nQuery: {query}")
        for i, doc in enumerate(search_documents(query), start=1):
            # ' '.join(text.split()) collapses newlines/extra spaces so each result prints on one line.
            print(f"  Result {i}: {' '.join(doc.page_content.split())[:150]}...")
        # Bonus: turn retrieval into an answer. Note the "don't know" path: the page
        # may not cover a question, and the prompt tells Claude not to make things up.
        print(f"  Answer: {qa.invoke(query)}")


def main() -> None:
    # Load: download the PDF; yields one Document per page.
    document = PyPDFLoader(PAPER_URL).load()
    # Split: pages are too big to embed/search precisely, so cut them into ~200-character chunks
    # at line breaks, with 20 characters of overlap between neighbours to keep context at the seams.
    chunks = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n").split_documents(document)

    # Run the sections in order; each one builds on the idea before it.
    embeddings_demo(chunks)
    docsearch = vector_store_demo(chunks)  # index once...
    retriever_demo(docsearch)              # ...then reuse the same index here
    parent_document_retriever_demo(document)  # gets the unsplit pages: it makes its own big + small chunks
    rag_demo(docsearch)                    # ...and here, now with an LLM answering
    exercise_4_retrieval_system()          # finally the whole pipeline again on a web page


# This guard means main() runs only when you execute this file directly
# (not when another file imports it).
if __name__ == "__main__":
    main()
