"""Step 5 - The finished QA bot: Gradio front end over the RAG pipeline.

    python app.py        then open http://localhost:7860
"""
from pathlib import Path

import gradio as gr

from qabot import config, db
from qabot.chain import format_sources, retrieve, stream_answer
from qabot.ingest import ingest_pdf
from qabot.store import explain_error

SAMPLE_PDF = Path(__file__).parent / "sample_docs" / "acme_broadband_handbook.pdf"
EXAMPLES = [
    "How much is the late payment fee?",
    "What does a solid red light on the router mean?",
    "Can I get a refund if I cancel in the first month?",
    "What is the Wi-Fi password for the office?",   # not in the handbook: the bot should say so
]


def chat(message: str, history: list[dict], k: int):
    """Called by gr.ChatInterface on every question. `yield` = streaming."""
    try:
        docs = retrieve(message, int(k))
        answer = ""
        for piece in stream_answer(message, docs):
            answer += piece
            yield answer
        sources = format_sources(docs)
        if sources:
            yield f"{answer}\n\n**Sources**\n{sources}"
        db.log_qa(message, answer, sources)
    except Exception as exc:  # noqa: BLE001 - show a toast instead of a stack trace
        raise gr.Error(explain_error(exc)) from exc


def library_rows() -> list[list]:
    return [[d["filename"], d["pages"], d["chunks"], d["ingested_at"]] for d in db.list_documents()]


def log_rows() -> list[list]:
    return [[r["asked_at"], r["question"], r["answer"], r["sources"]] for r in db.recent_qa()]


def ingest(paths: list[str] | None):
    """Called by the Add button: load, split, embed and store each uploaded PDF."""
    if not paths:
        raise gr.Error("Choose at least one PDF first.")
    lines = []
    for path in paths:
        try:
            r = ingest_pdf(path)
            state = "already in the library" if r.skipped else "added"
            lines.append(f"- **{r.filename}**: {state} ({r.pages} pages, {r.chunks} chunks)")
        except ValueError as exc:         # a bad PDF should not stop the other files
            lines.append(f"- {exc}")
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(explain_error(exc)) from exc
    return "\n".join(lines), library_rows()


def ingest_sample():
    return ingest([str(SAMPLE_PDF)])


with gr.Blocks(title="Document QA Bot") as demo:
    gr.Markdown(
        "# Document QA Bot\n"
        f"Answers come only from the PDFs you load. Chat model `{config.chat_model()}`, "
        f"embeddings `{config.embed_model()}`, both running locally in Ollama."
    )

    with gr.Tab("Ask"):
        top_k = gr.Slider(1, 8, value=config.top_k(), step=1, label="Passages to retrieve (k)", render=False)
        gr.ChatInterface(
            fn=chat,
            additional_inputs=[top_k],
            additional_inputs_accordion="Retrieval settings",
            examples=[[q] for q in EXAMPLES],
            cache_examples=False,
            api_name="chat",
        )

    with gr.Tab("Documents"):
        gr.Markdown("Add PDFs to the library. Each file is split into chunks, embedded and stored in ChromaDB.")
        uploads = gr.File(label="PDF files", file_count="multiple", file_types=[".pdf"], type="filepath")
        with gr.Row():
            add_btn = gr.Button("Add to library", variant="primary")
            sample_btn = gr.Button("Load sample handbook")
        status = gr.Markdown()
        library = gr.Dataframe(headers=["File", "Pages", "Chunks", "Added (UTC)"], value=library_rows, interactive=False)
        add_btn.click(ingest, inputs=uploads, outputs=[status, library], api_name="ingest")
        sample_btn.click(ingest_sample, outputs=[status, library], api_name="ingest_sample")

    with gr.Tab("Question log"):
        refresh = gr.Button("Refresh")
        log = gr.Dataframe(headers=["Asked (UTC)", "Question", "Answer", "Sources"], value=log_rows, interactive=False, wrap=True)
        refresh.click(log_rows, outputs=log, api_name="recent_questions")

if __name__ == "__main__":
    demo.launch()
