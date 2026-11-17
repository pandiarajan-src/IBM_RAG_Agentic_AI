"""Thin wrapper around the Ollama Python client.

Analogy: this file is the phone line to the kitchen. The UI (waiter) never
talks to the chef (model) directly - it always goes through this line, so we
can swap the chef later without retraining the waiter.
"""
from collections.abc import Iterator

import httpx  # installed with ollama; its errors are what we see when the server is down
import ollama

from . import config


class LLMUnavailable(RuntimeError):
    """Raised with a human-readable message when Ollama cannot answer."""


def _client() -> ollama.Client:
    return ollama.Client(host=config.ollama_host(), timeout=120)


def _text(content) -> str:
    """Gradio may hand us content as a plain string or as a list of
    {"type": "text", "text": ...} blocks. Ollama wants a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content or "")


def build_messages(system_prompt: str, history: list[dict], user_message: str) -> list[dict]:
    """Turn (system prompt + past turns + new question) into Ollama's message list."""
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for turn in history or []:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": _text(turn.get("content"))})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_chat(messages: list[dict], model: str | None = None, temperature: float = 0.3) -> Iterator[str]:
    """Yield the answer-so-far each time a new chunk arrives.

    Gradio replaces the output with whatever we yield, so we yield the
    growing text, not the individual chunk.
    """
    model = model or config.default_model()
    answer = ""
    try:
        for chunk in _client().chat(
            model=model,
            messages=messages,
            stream=True,
            options={"temperature": temperature},
        ):
            answer += chunk["message"]["content"] or ""
            yield answer
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise LLMUnavailable(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise LLMUnavailable(f"Ollama returned an error: {exc.error}") from exc
    except (httpx.TransportError, ConnectionError) as exc:
        raise LLMUnavailable(
            f"Cannot reach Ollama at {config.ollama_host()}. Is 'ollama serve' running?"
        ) from exc


def ask(question: str, system_prompt: str = config.DEFAULT_SYSTEM_PROMPT,
        model: str | None = None, temperature: float = 0.3) -> str:
    """One question in, one full answer out (no streaming, no history)."""
    answer = ""
    for answer in stream_chat(build_messages(system_prompt, [], question), model, temperature):
        pass
    return answer


def list_models() -> list[str]:
    """Names of locally installed models; falls back to the default if Ollama is down."""
    try:
        names = sorted(m.model for m in _client().list().models)
    except Exception:
        names = []
    default = config.default_model()
    if not any(n == default or n.split(":")[0] == default for n in names):
        names.insert(0, default)
    return names
