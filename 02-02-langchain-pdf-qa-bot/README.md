# Lab 02 - Build a PDF Question-Answering Bot with LangChain, ChromaDB and Local Ollama

**Time:** 60 minutes | **Level:** beginner-intermediate | **Verified:** 2026-11-14 with langchain-core 1.8.0, langchain-ollama 1.1.0, langchain-chroma 1.1.0, chromadb 1.5.9, Gradio 6.32.0, Python 3.13 (27 tests against the Ollama simulator)

---

## 1. Problem statement

Acme Broadband has a library of PDFs: customer handbooks, policies, technical manuals. Support agents waste minutes per call searching them, and a plain chatbot (Lab 01) simply invents answers because it has never read those files.

We need a bot that reads the PDFs and answers questions **only from what they say**, showing which page each answer came from.

## 2. What you will learn

By the end you can:

1. Combine a document loader, text splitter, embedding model, vector database and retriever into a working QA bot.
2. Use LangChain and a local LLM to answer questions from PDF content (retrieval-augmented generation, RAG).
3. Explain what each component does and what breaks when you tune it badly.
4. Put a Gradio front end on it, test it without a model, containerise it and reason about production.

## 3. What we will build

```
INGEST (once per PDF)
  PDF --> Loader --> Splitter --> Embedding model --> ChromaDB (vectors on disk)
          pages      chunks       numbers                 |
                                                          +--> SQLite (which files are loaded)
ASK (every question)
  Question --> Embedding model --> Retriever (top-k chunks) --> Prompt --> LLM --> Answer + sources
```

Five steps, each adding one component:

| Step | File | Component | Needs Ollama |
|---|---|---|---|
| 1 | `step1_load_split.py` | Document loader + text splitter | No |
| 2 | `step2_embed_store.py` | Embedding model + vector database | Yes |
| 3 | `step3_retrieve.py` | Retriever | Yes |
| 4 | `step4_qa_chain.py` | Prompt + LLM = RAG chain | Yes |
| 5 | `app.py` | Gradio front end | Yes |

Folder map:

```
02-langchain-pdf-qa-bot/
  README.md              this guide
  PRODUCTION.md          taking it to production
  step1..step4_*.py      one component per step
  app.py                 step 5: the finished bot
  qabot/                 shared code: config, loader, ingest, store, chain, db
  sample_docs/           acme_broadband_handbook.pdf (fictional, 6 pages)
  tests/                 pytest suite + fake Ollama server
  Dockerfile, docker-compose.yml, .env.example
  requirements.txt, requirements-dev.txt
```

## 4. Tech stack and why

| Choice | Why | Alternatives considered |
|---|---|---|
| **LangChain 1.x** (MIT): `langchain-core`, `langchain-text-splitters`, `langchain-ollama`, `langchain-chroma` | Standard interfaces (`Document`, `Retriever`, `Runnable`) so every part is swappable; we install only the four small packages we use | LlamaIndex (more batteries, more magic), plain Python (fine here, but you re-invent the interfaces) |
| **pypdf** (BSD) behind our own 15-line loader | `langchain-community`, home of the old `PyPDFLoader`, is being sunset; a loader is tiny and worth seeing | Docling / PyMuPDF4LLM (better on tables and scans, heavier; see PRODUCTION.md) |
| **Ollama** (MIT) with **llama3.2** + **nomic-embed-text** (Apache-2.0) | Free, offline, no API key; documents never leave the laptop | Claude API (better answers, costs money, data leaves the machine) |
| **ChromaDB** (Apache-2.0), embedded | Team default vector DB; runs in-process and saves to a folder, zero setup | FAISS (no metadata filtering or persistence built in), pgvector / Qdrant (for scale) |
| **SQLite** | Team default; records which files are loaded so re-uploads cost nothing | Chroma metadata only (awkward to list and audit) |
| **Gradio 6** (Apache-2.0) | Chat UI + file upload in ~60 lines of Python | FastAPI + React (team default for production; too much for 60 min) |
| **pytest** + fake Ollama | 27 tests in ~20 s with no model or GPU | Mocking LangChain objects (hides HTTP and streaming bugs) |
| **Docker Compose** | Same environment on every laptop | Bare venv (fine for the lab, drifts over time) |

> **No FastAPI or React here?** Same reasoning as Lab 01: Gradio replaces both for internal tools and prototypes. `qabot/` has no Gradio imports, so a FastAPI service can reuse it unchanged. PRODUCTION.md shows when to switch.

> **Old tutorials look different.** `RetrievalQA`, `from langchain.chains import ...` and `langchain_community` loaders are legacy (now in `langchain-classic` or sunset). The current style is LCEL: components joined with `|`.

> **Licence flag:** everything above is commercial-friendly except the *model* llama3.2 (Meta's Llama Community License, not MIT/Apache). Fine for learning; check with legal before shipping, or use an Apache/MIT model such as `qwen2.5`, `mistral` or `phi4-mini` via `OLLAMA_MODEL`.

## 5. Concepts, with analogies

The whole thing is an **open-book exam**: the model may not answer from memory; it must find the page first.

| Concept | Analogy | In one line |
|---|---|---|
| **RAG** | Open-book exam | Retrieve relevant text first, then let the LLM answer using only that text |
| **Document loader** | Mailroom clerk | Opens the file and hands over plain text plus a stamp saying where it came from (`metadata`) |
| **`Document`** | An index card | `page_content` (the text) + `metadata` (source, page) |
| **Text splitter** | Cutting a manual into index cards | Small pieces are easier to match precisely and cheap to fit in a prompt |
| **Chunk overlap** | Repeating the last line at the top of the next card | Stops a sentence cut at a boundary from being lost |
| **Embedding** | GPS coordinates for meaning | Text becomes a list of numbers; similar meanings land close together |
| **Vector database** | A map with "find what is nearby" | Stores the coordinates and returns the nearest neighbours fast |
| **Retriever** | The librarian | Question in, top-k most relevant cards out |
| **top-k** | How many books the librarian brings back | Too few: answer missing. Too many: noise, slower, costlier |
| **Prompt template** | An exam paper with blanks | Fixed rules + `{context}` + `{question}` filled in per request |
| **LCEL chain (`a \| b \| c`)** | An assembly line | Output of each station feeds the next; any station can be swapped |
| **Grounding / citations** | "Show your working" | The answer points at pages so a human can verify it |
| **Idempotent ingest** | A coat-check ticket | Same file (same fingerprint) is never processed twice |

## 6. Setup (8 min)

Prerequisites: Python 3.10+, [Ollama](https://ollama.com/download) installed and running.

```bash
cd 02-langchain-pdf-qa-bot

ollama pull llama3.2                 # chat model, ~2 GB (already there from Lab 01)
ollama pull nomic-embed-text         # embedding model, ~270 MB

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**No model yet / slow download?** Use the simulator so you are not blocked:

```bash
python tests/fake_ollama.py &                   # fake server on :11435
export OLLAMA_HOST=http://127.0.0.1:11435       # crude word-matching answers, but every step runs
```

## 7. Exercises

### Exercise 1 - Load and split (8 min)

```bash
python step1_load_split.py
```

1. 6 pages became 14 chunks. Open `qabot/loader.py`: a loader is just "yield one `Document` per page".
2. Compare the end of chunk 0 with the start of chunk 1 - that repeat is the overlap.
3. Run with `QABOT_CHUNK_SIZE=200 QABOT_CHUNK_OVERLAP=0 python step1_load_split.py`. Chunks now end mid-thought.

**Try it:** point it at any PDF of your own: `python step1_load_split.py ~/Downloads/some.pdf`. Is the extracted text clean? Tables and scans usually are not.

### Exercise 2 - Embed and store (8 min)

```bash
python step2_embed_store.py
```

1. One sentence became 768 numbers (256 with the simulator). The billing question scores closer to the billing sentence than to the router sentence, although they share few exact words.
2. The handbook's 14 chunks are now in `data/chroma/`. Run the script again: "already in the library". Find the two lines in `qabot/ingest.py` that make this happen (`file_id`, `db.find_document`).

**Try it:** in the script, change `far` to a sentence of your own. How low can you get the similarity?

### Exercise 3 - Retrieve (8 min)

```bash
python step3_retrieve.py "What does a red light on the router mean?"
```

1. Read the 4 hits and their distances (0 = same meaning, 1 = unrelated). Is the right passage ranked first?
2. Ask something the handbook does not cover: `python step3_retrieve.py "Who won the world cup?"`. You still get 4 hits - a retriever always returns its nearest neighbours, relevant or not. Note the distances.
3. Ask with different words than the handbook uses: "my internet is dead". With the real embedding model the red-light passage should still rank near the top; a keyword search would miss it.

**Try it:** `QABOT_TOP_K=1 python step3_retrieve.py "How much is the late fee and when is the bill due?"`. One card cannot answer a two-part question.

### Exercise 4 - The RAG chain (10 min)

```bash
python step4_qa_chain.py "Can I get my money back?"
```

1. The script prints the exact prompt sent to the model: rules, numbered passages, question. That is all RAG is.
2. Open `qabot/chain.py` and read `build_rag_chain`. Trace the assembly line: `retriever | format_docs` fills `{context}`, then `PROMPT | llm | parser`.
3. Ask `"What is the Wi-Fi password for the office?"`. The handbook does not say; a well-behaved bot replies that it could not find it. Which sentence in `qabot/config.py` causes that?

**Try it:** delete the "Use ONLY the numbered context" sentence from `SYSTEM_PROMPT` and ask `"Who founded Acme Broadband?"`. Watch it start guessing. Put the sentence back.

### Exercise 5 - The Gradio app (10 min)

```bash
python app.py                 # open http://localhost:7860
```

1. **Documents** tab: the handbook from step 2 is listed. Upload a PDF of your own and click **Add to library**.
2. **Ask** tab: click an example. The answer streams, then the sources appear.
3. Open **Retrieval settings**, set k to 1, ask a two-part question, then set k to 8. Compare quality and speed.
4. **Question log** tab: every question is saved in SQLite. In production this log becomes your evaluation set.
5. Read `chat()` in `app.py`: retrieve, stream, append sources, log. About a dozen lines.

**Try it:** add a **Remove** feature: a `gr.Dropdown` of file names and a button that calls `qabot.ingest.remove_document(doc_id)`. The function is already written and tested.

### Exercise 6 - Tests (4 min)

```bash
pytest -v
```

27 tests, about 20 seconds, no model needed. Open `tests/fake_ollama.py`: a flight simulator for the Ollama API whose embeddings are simple word counts - enough for retrieval tests to be meaningful. `test_end_to_end_over_http` launches the real app, uploads the PDF and asks a question like a browser would.

### Exercise 7 - Docker (4 min)

```bash
docker compose up --build     # open http://localhost:7860, click "Load sample handbook"
```

The app runs in a container; Ollama stays on your host. ChromaDB and SQLite live in the `qabot_data` volume, so the library survives restarts.

> Linux only: Ollama listens on 127.0.0.1 by default, which containers cannot reach. Start it with `OLLAMA_HOST=0.0.0.0 ollama serve`.

### Stretch goals

- **Sentence-aware chunks:** in `qabot/ingest.py` pass `separators=["\n\n", ". ", " ", ""]` and `keep_separator="end"` to the splitter, re-run step 1 and compare.
- **Refuse off-topic questions early:** use `similarity_search_with_score` and skip the LLM when the best distance is above a threshold you pick from Exercise 3.
- **Follow-up questions:** "And how much is it?" fails because each question is retrieved on its own. Use the LLM to rewrite the question using `history` before retrieving.
- **Swap the LLM:** change `get_llm()` to `ChatAnthropic` from `langchain-anthropic`. Nothing else changes - that is what the standard interface buys you.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Cannot reach Ollama at ..." | Ollama not running | Start the Ollama app or `ollama serve` |
| "A model is not installed" | Model not pulled | `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| "The library is empty" | Step 2 not run | `python step2_embed_store.py`, or use the Documents tab |
| "no text found. Scanned PDFs need OCR" | PDF is images of pages | Use a text PDF for the lab; see PRODUCTION.md for OCR |
| Answers ignore a PDF you just changed | Old chunks still stored | Edited file = new fingerprint; remove the old one (`remove_document`) |
| Wrong or mixed-up answers after changing `OLLAMA_EMBED_MODEL` | Old vectors came from a different model | Delete `data/` and ingest again - vectors from two models are not comparable |
| First answer takes 20+ s | Model loading into memory | Normal; later answers are fast |
| Port 7860 in use | Lab 01 or another run still up | Stop it, or `GRADIO_SERVER_PORT=7861 python app.py` |
| Works locally, fails in Docker on Linux | Ollama bound to 127.0.0.1 | `OLLAMA_HOST=0.0.0.0 ollama serve` |
| `ImportError: langchain.chains` / `langchain_community` from copied code | Pre-1.0 LangChain tutorial | Use the imports in `qabot/`; legacy chains live in `langchain-classic` |

## 9. Conclusion

You built a QA bot from six replaceable parts: loader, splitter, embedding model, vector database, retriever and LLM, with Gradio on top. The ideas that carry to any RAG system:

- The LLM is the last and easiest step. Answer quality is decided earlier, by parsing, chunking and retrieval.
- A retriever always returns something. Grounding rules in the prompt, and later a distance threshold, stop the model bluffing.
- Show sources. An answer nobody can verify is not worth much in support or legal work.
- Ingestion must be safe to repeat, and embeddings are tied to the model that made them.
- Standard interfaces let you swap any part - PDF parser, vector DB or LLM - without touching the rest.

Next: read `PRODUCTION.md` to see what breaks when 6 pages become 60,000.
