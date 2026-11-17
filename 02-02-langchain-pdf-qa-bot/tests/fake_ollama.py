"""A tiny stand-in for the Ollama server so tests need no model and no GPU.

Analogy: a flight simulator. Same cockpit (HTTP API), no real plane.
Also runnable by hand:  python tests/fake_ollama.py   (listens on :11435)

- /api/embed  returns a "bag of words" vector: texts sharing words land close together.
- /api/chat   replies with the context sentence that best matches the question.
Good enough to exercise the whole pipeline; nowhere near as smart as a real model.
"""
import json
import math
import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 256
MODELS = ["llama3.2:latest", "nomic-embed-text:latest"]
STOPWORDS = set("a an and are at be by can do does for from get how i if in is it my of on or the to what when with".split())


def words(text: str) -> list[str]:
    """Lower-case, drop filler words, keep the first 5 letters (so refund ~ refunded)."""
    return [w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for w in words(text):
        vec[zlib.crc32(w.encode()) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def best_sentence(prompt: str) -> str:
    """Pick the context sentence sharing the most words with the question."""
    context, _, question = prompt.rpartition("Question:")
    wanted = set(words(question))
    passages = re.split(r"^\[\d+\] \(.*\)$", context, flags=re.MULTILINE)   # drop "[1] (file, page 3)" headers
    sentences = [s for p in passages for s in re.split(r"(?<=[.!?])\s+", p.replace("\n", " ").strip())]
    scored = [(len(wanted & set(words(s))), s.strip()) for s in sentences]
    score, sentence = max(scored, default=(0, ""))
    return sentence if score else "I could not find that in the loaded documents."


class Handler(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"model": m, "name": m} for m in MODELS]})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests_seen.append({"path": self.path, **body})
        model = body.get("model", "")
        if model == "missing-model":
            return self._send(404, {"error": "model 'missing-model' not found"})
        if self.path == "/api/embed":
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._send(200, {"model": model, "embeddings": [embed(t) for t in texts]})
        if self.path == "/api/chat":
            reply = best_sentence(body["messages"][-1]["content"])
            pieces = re.findall(r"\S+\s*", reply)
            lines = [{"model": model, "message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
            lines.append({"model": model, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"})
            return self._send(200, "".join(json.dumps(l) + "\n" for l in lines).encode(), "application/x-ndjson")
        self._send(404, {})


def start(port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    srv = start(11435)
    print("Fake Ollama on http://127.0.0.1:11435  (Ctrl+C to stop)")
    threading.Event().wait()
