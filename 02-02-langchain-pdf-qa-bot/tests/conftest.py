import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import qabot` works
sys.path.insert(0, str(ROOT / "tests"))

import fake_ollama  # noqa: E402

SAMPLE_PDF = ROOT / "sample_docs" / "acme_broadband_handbook.pdf"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets a fake Ollama, an empty ChromaDB folder and a throwaway SQLite file."""
    server = fake_ollama.start()
    fake_ollama.Handler.requests_seen.clear()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("QABOT_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("QABOT_DB", str(tmp_path / "test.db"))
    for name in ("QABOT_CHUNK_SIZE", "QABOT_CHUNK_OVERLAP", "QABOT_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def sample_pdf() -> Path:
    return SAMPLE_PDF


@pytest.fixture
def loaded_library(sample_pdf):
    """The sample handbook, already ingested."""
    from qabot.ingest import ingest_pdf

    return ingest_pdf(sample_pdf)
