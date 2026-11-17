# Taking the QA Bot to Production

**Analogy:** the lab is a personal bookshelf - you know every book and you are the only reader. Production is a city library: deliveries every day, damaged books, a catalogue that must stay correct, members who may only enter certain rooms, and a librarian whose advice people act on. Same idea, different operation.

## 1. What changes

```
Lab:         Browser -> Gradio -> [ChromaDB folder + SQLite file] -> Ollama (laptop)

Production:  Browser -> CDN/WAF -> Load balancer (TLS, SSO)
                          -> React or Gradio UI
                          -> FastAPI service (auth, rate limit, guardrails, logging)
                               -> Retrieval: vector DB server (+ keyword index, reranker)
                               -> Model server pool (GPU) or hosted LLM API
             Sources -> Ingestion workers (queue) -> parse/OCR -> chunk -> embed -> vector DB
                          -> Postgres (document registry, Q&A log) + object store (original PDFs)
```

The biggest change: **ingestion becomes its own pipeline**, separate from the chat app.

## 2. Deployment guide (small internal rollout)

1. **Build and push:** `docker build -t registry.example.com/qabot:1.0.0 .` then push. Tag by version, never `latest`.
2. **Configure by environment:** `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `QABOT_*`. No secrets in the image.
3. **Run Ollama on a GPU host** with `OLLAMA_HOST=0.0.0.0`, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=-1`. Keep it on a private network - Ollama has no authentication.
4. **Run ChromaDB as a server** (`chroma run` or its Docker image) on a persistent disk, and connect with `Chroma(host=..., port=...)` in `qabot/store.py`. The embedded folder mode supports one process only.
5. **Persist and back up** the vector DB volume, the SQLite/Postgres database and the original PDFs. The PDFs are the source of truth: with them you can always rebuild the index.
6. **Put a reverse proxy in front** for TLS. For streaming, disable buffering (`proxy_buffering off;`) and raise read timeouts.
7. **Add login:** mount Gradio inside FastAPI (`gr.mount_gradio_app(app, demo, path="/", auth_dependency=...)`) behind your OIDC provider.
8. **Lock down the UI:** remove the Documents and Question log tabs from the end-user build (section 3, rows 7 and 11). Ingestion is an admin job.
9. **Health checks:** probe `/` on the app, `/api/tags` on Ollama and `/api/v2/heartbeat` on Chroma.
10. **Roll out** to a pilot group, watch the metrics in section 4, then widen.

## 3. Challenges when you scale up

| # | Challenge | Why it bites | What to do |
|---|---|---|---|
| 1 | **PDF parsing quality** | Real PDFs have tables, columns, headers/footers and scanned pages; `pypdf` returns jumbled or empty text. Garbage in, garbage out | Layout-aware parsers (Docling, PyMuPDF4LLM, Unstructured), OCR for scans, strip repeated headers; spot-check extracted text per source |
| 2 | **Chunking strategy** | Fixed 500-character cuts split tables, clauses and steps; legal text needs whole sections | Split on structure (headings, sections), keep the heading in each chunk, tune size per document type and measure (row 6) |
| 3 | **Ingestion at volume** | The lab embeds inside the web request; 10,000 PDFs would block the UI for hours and a crash loses progress | Queue + workers (Celery, RQ, Arq), batches, retries, status per document; our stable chunk ids already make retries safe |
| 4 | **Stale and changed documents** | Policy v3 replaces v2, but v2's chunks still answer questions | Track version and effective date in metadata, delete-then-insert on update, scheduled re-sync from the source system |
| 5 | **Embedding model lock-in** | Vectors from two models are not comparable; changing model means re-embedding everything | Record model name + version per collection, build the new index alongside the old one, switch over, then drop the old |
| 6 | **Retrieval quality** | Pure vector search misses exact terms (error codes, part numbers, names) and top-k is a blunt tool | Hybrid search (BM25 + vectors), a reranker on the top 20-50, metadata filters, query rewriting for follow-ups. For nomic models also add the `search_query:` / `search_document:` prefixes |
| 7 | **Access control** | One shared collection means anyone can retrieve chunks from HR or legal files they may not open | Store ACL/tenant in chunk metadata and filter *in the retrieval query*, never after; separate collections per tenant for hard isolation |
| 8 | **Hallucination and trust** | A fluent wrong answer about a refund or contract costs real money | Keep grounding rules, cite pages, refuse when the best distance is poor, human hand-off, and say "not found" proudly |
| 9 | **Evaluation** | "It looked fine in the demo" does not survive a prompt, model or chunking change | Golden set of 50-200 real questions with expected answers and pages (start from the Question log); score retrieval hit-rate and answer faithfulness (Ragas, DeepEval) in CI |
| 10 | **Prompt injection via documents** | A PDF can contain "ignore your rules and reveal..." and the model may obey | Treat context as data (our system prompt says so), restrict who can ingest, scan uploads, give the bot no tools it does not need, filter output |
| 11 | **Sensitive data** | PDFs and questions contain PII and confidential terms; the log tab shows everyone's questions | Redact or classify at ingest, encrypt at rest, retention policy, role-gate logs; keep data in-region |
| 12 | **Vector DB scale** | Embedded Chroma holds the index in one process's memory on one disk | Chroma server, then Qdrant / Milvus / pgvector as you pass millions of chunks; plan for index RAM, replication and backups |
| 13 | **Concurrency, latency and cost** | Each question = one embedding call + one generation; one GPU slot serves one request at a time | Keep models warm, cap context (k x chunk size), cache frequent questions, autoscale on queue depth, vLLM/TGI or a hosted API for throughput |
| 14 | **SQLite and multiple replicas** | One writer at a time, file on one container's disk; Gradio keeps session state in memory | Postgres; sticky sessions or a stateless API |
| 15 | **Upload safety** | Users upload huge, malformed or malicious files | Size and page limits, type sniffing (not just the `.pdf` extension), virus scan, parse in a sandboxed worker |
| 16 | **Dependency churn** | LangChain moves fast; `langchain-community`, used by most older tutorials, has been sunset | Pin versions (we do), depend on the small `langchain-core` interfaces, upgrade on a schedule with the test suite as the gate |

## 4. What to measure

- **Retrieval:** hit-rate@k on the golden set, distance of the top hit, share of "not found" answers.
- **Answers:** faithfulness to context, citation accuracy, thumbs up/down, hand-off rate.
- **Speed:** time to first token, total latency (split into embed / search / generate), queue wait.
- **Ingestion:** documents pending, failure rate by cause (encrypted, scanned, too large), index freshness (age of the newest synced document).
- **Cost and capacity:** tokens per question, GPU utilisation, vector count and index size.

Tools: OpenTelemetry + Prometheus/Grafana for metrics, Langfuse or LangSmith for per-question traces (which chunks were retrieved, what prompt was sent).

## 5. When to leave Gradio for FastAPI + React

Stay on Gradio while the bot is an internal tool with tens of users. Switch when you need custom branding, per-user document permissions in the UI, mobile support or embedding the chat in an existing product.

The move is cheap because `qabot/` has no Gradio imports: wrap `retrieve` + `stream_answer` in a FastAPI streaming endpoint and `ingest_pdf` in an upload endpoint that enqueues a job, then build the React client against those.

## 6. Pre-launch checklist

- [ ] Golden-set evaluation passing and wired into CI
- [ ] Ingestion runs in workers, is retry-safe and reports per-document status
- [ ] Document updates and deletions propagate to the index
- [ ] Access control enforced inside the retrieval query
- [ ] Sources shown; "not found" path and human hand-off tested
- [ ] SSO in place; Documents and Question log tabs admin-only
- [ ] Upload limits, type checks and malware scan
- [ ] PII handling, retention and encryption documented
- [ ] Vector DB, relational DB and original files backed up; index rebuild rehearsed
- [ ] Embedding and chat model versions recorded; licences cleared (see README licence flag)
- [ ] Dashboards and alerts for latency, errors, "not found" rate and ingestion failures
- [ ] Load test at 2x expected peak
