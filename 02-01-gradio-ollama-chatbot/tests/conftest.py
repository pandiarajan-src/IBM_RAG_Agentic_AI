import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import supportbot` works
sys.path.insert(0, str(ROOT / "tests"))

import fake_ollama  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets a fake Ollama and a throwaway SQLite file."""
    server = fake_ollama.start()
    fake_ollama.Handler.requests_seen.clear()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("SUPPORTBOT_DB", str(tmp_path / "test.db"))
    yield server
    server.shutdown()
    server.server_close()
