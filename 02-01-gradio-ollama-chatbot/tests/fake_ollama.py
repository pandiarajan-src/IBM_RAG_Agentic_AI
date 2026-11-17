"""A tiny stand-in for the Ollama server so tests need no model and no GPU.

Analogy: a flight simulator. Same cockpit (HTTP API), no real plane.
Also runnable by hand:  python tests/fake_ollama.py   (listens on :11435)
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ["Please ", "restart ", "your ", "router."]
FULL_REPLY = "".join(WORDS)


class Handler(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, body: bytes, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, json.dumps({"models": [{"model": "llama3.2:latest", "name": "llama3.2:latest"}]}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests_seen.append(body)
        if self.path != "/api/chat":
            return self._send(404, b"{}")
        if body["model"] == "missing-model":
            return self._send(404, json.dumps({"error": "model 'missing-model' not found"}).encode())
        lines = [
            {"model": body["model"], "message": {"role": "assistant", "content": w}, "done": False}
            for w in WORDS
        ] + [{"model": body["model"], "message": {"role": "assistant", "content": ""}, "done": True}]
        self._send(200, "".join(json.dumps(l) + "\n" for l in lines).encode(), "application/x-ndjson")


def start(port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    srv = start(11435)
    print("Fake Ollama on http://127.0.0.1:11435  (Ctrl+C to stop)")
    threading.Event().wait()
