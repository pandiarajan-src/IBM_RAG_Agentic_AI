"""All settings in one place, read from environment variables (12-factor style)."""
import os

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly customer support agent for Acme Broadband. "
    "Answer in at most 4 short sentences. "
    "If you do not know the answer, say so and offer to connect the customer to a human."
)


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


def default_model() -> str:
    return os.getenv("OLLAMA_MODEL", "gemma3:4b")


def db_path() -> str:
    return os.getenv("SUPPORTBOT_DB", "data/chat_log.db")
