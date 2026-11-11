"""
03_documents_loaders_splitters.py - Documents, loaders and text splitters (+ Exercise 3)
=======================================================================================

LEARNING NOTES
--------------
This script covers the first half of every RAG (Retrieval-Augmented Generation)
pipeline. No LLM is called here, so no API key is needed.

    source (PDF, web page, ...) --load--> Documents --split--> chunks --embed/store--> (script 04)

Document (langchain_core.documents.Document)
--------------------------------------------
LangChain's universal container for text:
- page_content : str   - the text itself
- metadata     : dict  - anything describing it (source, page number, id, timestamp ...)
Metadata matters later: it tells you WHERE an answer came from (citations), and
vector stores can filter on it.

Document loaders (langchain_community.document_loaders)
-------------------------------------------------------
LangChain has 100+ loaders that read a source and return `list[Document]`.
- PyPDFLoader(path_or_url) : one Document PER PAGE, with metadata such as
  source, page and total_pages. Needs the `pypdf` package.
- WebBaseLoader(url)       : downloads HTML and extracts the visible text with
  BeautifulSoup (`beautifulsoup4` package). One Document per URL.

Text splitters (langchain_text_splitters)
-----------------------------------------
Why split? Models have context limits, and small focused chunks embed and
retrieve far more precisely than whole pages.
How it works: split the text into small pieces at a separator, then merge the
pieces back together until the next one would exceed `chunk_size`. Start a new
chunk that repeats the last `chunk_overlap` characters, so a sentence cut at a
boundary keeps its context.

| Splitter                        | Splits on                                   | Notes                           |
|---------------------------------|---------------------------------------------|---------------------------------|
| CharacterTextSplitter           | ONE separator (default "\\n\\n")              | simplest; chunks can overshoot  |
|                                 |                                             | chunk_size if no separator fits |
| RecursiveCharacterTextSplitter  | a LIST, tried in order:                     | recommended default; keeps      |
|                                 | "\\n\\n" -> "\\n" -> " " -> ""                  | paragraphs/sentences together   |

`split_documents()` copies each parent Document's metadata onto every chunk,
so a chunk still knows which page it came from.

Changes from the notebook (LangChain 0.2 -> 1.x)
------------------------------------------------
- `from langchain.text_splitter import ...` -> `from langchain_text_splitters import ...`
- The v0.2 docs URL now redirects; we load the current LangChain overview page.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/03_documents_loaders_splitters.py
"""

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from llm_setup import PAPER_URL, WEB_URL, print_header


# ---------------------------------------------------------------------------
# 1. Document objects
# ---------------------------------------------------------------------------
def document_object_demo() -> None:
    print_header("1. DOCUMENT OBJECT")
    doc = Document(
        page_content="""Python is an interpreted high-level general-purpose programming language.
Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
        metadata={
            "my_document_id": 234234,  # unique identifier
            "my_document_source": "About Python",  # source / title
            "my_document_create_time": 1680013019,  # Unix timestamp (March 28, 2023)
        },
    )
    print(doc)

    # Metadata is optional - it defaults to an empty dict.
    print(Document(page_content="Python is an interpreted high-level general-purpose programming language."))


# ---------------------------------------------------------------------------
# 2. Document loaders: PDF and web
# ---------------------------------------------------------------------------
def loaders_demo() -> list[Document]:
    print_header("2. DOCUMENT LOADERS")

    # PyPDFLoader downloads the PDF and returns one Document per page.
    document = PyPDFLoader(PAPER_URL).load()
    print(f"PDF pages loaded: {len(document)}")
    print(f"page 2 metadata : {document[2].metadata}")
    print(f"page 1 (first 500 chars):\n{document[1].page_content[:500]}\n")

    web_data = WebBaseLoader(WEB_URL).load()
    print(f"Web documents loaded: {len(web_data)}  metadata keys: {list(web_data[0].metadata)}")
    # Web text is full of blank lines from the HTML layout; collapse them for display.
    print(f"web page (first 500 chars):\n{' '.join(web_data[0].page_content.split())[:500]}")
    return document


# ---------------------------------------------------------------------------
# 3. Text splitters
# ---------------------------------------------------------------------------
def splitter_demo(document: list[Document]) -> None:
    print_header("3. TEXT SPLITTER")
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
    chunks = text_splitter.split_documents(document)
    print(f"{len(document)} pages -> {len(chunks)} chunks")
    print(f"chunk 5 content : {chunks[5].page_content!r}")
    print(f"chunk 5 metadata: {chunks[5].metadata}  <- inherited from its page")


# ---------------------------------------------------------------------------
# 4. Exercise 3 - compare two splitters on PDF and web content
# ---------------------------------------------------------------------------
def display_document_stats(docs: list[Document], name: str) -> None:
    """Display statistics about a list of document chunks."""
    total_chunks = len(docs)
    total_chars = sum(len(doc.page_content) for doc in docs)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0

    all_metadata_keys = set()
    for doc in docs:
        all_metadata_keys.update(doc.metadata.keys())

    print(f"\n=== {name} Statistics ===")
    print(f"Total number of chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size:.2f} characters")
    print(f"Metadata keys preserved: {', '.join(sorted(all_metadata_keys))}")

    if docs:
        example_doc = docs[min(5, total_chunks - 1)]
        print(f"Example chunk (first 150 chars): {example_doc.page_content[:150]!r}...")
        print(f"Example metadata: {example_doc.metadata}")
        lengths = [len(doc.page_content) for doc in docs]
        print(f"Min chunk size: {min(lengths)} characters")
        print(f"Max chunk size: {max(lengths)} characters")


def exercise_3_compare_splitters() -> None:
    print_header("4. EXERCISE 3: LOADERS + TWO SPLITTERS")
    pdf_document = PyPDFLoader(PAPER_URL).load()
    web_document = WebBaseLoader(WEB_URL).load()

    # Splitter 1: one separator, smaller chunks.
    splitter_1 = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
    # Splitter 2: recursive, larger chunks, falls back to smaller separators.
    splitter_2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])

    display_document_stats(splitter_1.split_documents(pdf_document), "PDF / Splitter 1 (Character, 300)")
    display_document_stats(splitter_2.split_documents(pdf_document), "PDF / Splitter 2 (Recursive, 500)")
    # Web pages are one big Document; splitting matters even more there.
    display_document_stats(splitter_2.split_documents(web_document), "Web / Splitter 2 (Recursive, 500)")

    print(
        "\nWhat to notice: the Recursive splitter's max chunk size stays at or under 500,"
        "\nwhile CharacterTextSplitter can overshoot when a line has no '\\n' to break on."
        "\nEvery chunk keeps its source metadata (source, page ...)."
    )


def main() -> None:
    document_object_demo()
    document = loaders_demo()
    splitter_demo(document)
    exercise_3_compare_splitters()


if __name__ == "__main__":
    main()
