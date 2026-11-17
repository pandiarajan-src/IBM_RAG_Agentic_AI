"""
llm_setup.py - Shared helpers used by every script in this folder.
==================================================================

LEARNING NOTES
--------------
The original IBM lab used watsonx.ai for two things:

| Notebook (IBM)                                 | Here                                           |
|------------------------------------------------|------------------------------------------------|
| `ModelInference(...).generate()`  (raw model)  | Anthropic SDK `client.messages.create()`        |
| `WatsonxLLM(model=...)`  (LangChain wrapper)   | `ChatAnthropic` from `langchain-anthropic`      |
| `WatsonxEmbeddings(slate-125m-english-rtrvr)`  | `OpenAIEmbeddings` pointed at **OpenRouter**    |

Why OpenRouter for embeddings?
    Claude generates text but has no embedding endpoint. OpenRouter exposes
    many embedding models behind ONE OpenAI-compatible API, so LangChain's
    `OpenAIEmbeddings` class can talk to it by just changing `base_url`.
    Default: `qwen/qwen3-embedding-8b`, a top-ranked embedding model that costs
    about $0.01 per million tokens. Cheaper/free alternatives are listed below.

Generation parameters: watsonx vs Claude
---------------------------------------
| watsonx (notebook)  | Claude equivalent      | Notes                                       |
|---------------------|------------------------|---------------------------------------------|
| MAX_NEW_TOKENS      | max_tokens             | Hard cap on output length.                  |
| TEMPERATURE         | (removed)              | Current Claude models reject it (HTTP 400). |
| (none)              | effort                 | low | medium | high | xhigh | max.          |
| (none)              | thinking               | Let the model reason before answering.      |
| (none)              | system                 | Role / behaviour instructions.              |

Exercise 1 in the notebook compared temperature 0.1 vs 0.8. With Claude you
compare *models* (Haiku vs Sonnet/Opus) and *effort* levels instead.

Setup
-----
1. Packages (already added from the repo root):
       uv add anthropic langchain langchain-core langchain-anthropic python-dotenv \
              langchain-openai langchain-chroma langchain-community \
              langchain-text-splitters pypdf beautifulsoup4
2. Create `<repo root>/.env` containing:
       ANTHROPIC_API_KEY=sk-ant-...
       OPENROUTER_API_KEY=sk-or-...      # only needed for script 04
3. Run any script from the repo root, e.g.:
       uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/01_models_and_messages.py
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

# Load API keys from the .env file at the repo root (one level up).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# WebBaseLoader warns if no user agent is set; give it a polite one.
os.environ.setdefault("USER_AGENT", "langchain-learning-scripts")
# Chroma sends anonymous telemetry unless told not to (the notebook did this too).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Change the chat model in ONE place. Haiku is the fastest/cheapest - ideal for
# learning. Swap to "claude-sonnet-5-5" or "claude-opus-5-5" to compare quality.
MODEL = os.environ.get("MODEL", "claude-haiku-5-5")

# Models that accept `thinking={"type": "disabled"}`. Claude Opus 5.5 and
# Claude Sonnet 5.5 always think; for them we leave thinking on and keep effort low.
MODELS_THAT_CAN_DISABLE_THINKING = {"claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5"}

# Embedding model served through OpenRouter. Other good options:
#   "voyageai/voyage-4-lite"          - Anthropic's recommended embedding partner
#   "openai/text-embedding-3-small"   - the classic OpenAI choice
#   "baai/bge-base-en-v1.5"           - tiny & ultra cheap (512-token limit)
#   "nvidia/nemotron-3.5-embed-1b:free" - free, but rate-limited
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Source material used by the notebook (scripts 03 and 04).
PAPER_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
# The notebook used https://python.langchain.com/v0.2/docs/introduction/ which
# now redirects here.
WEB_URL = "https://docs.langchain.com/oss/python/langchain/overview"

_client = None


def get_client() -> anthropic.Anthropic:
    """Return a single shared Anthropic client (created on first use)."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def thinking_config(thinking: bool, model: str = MODEL) -> dict:
    if not thinking and model in MODELS_THAT_CAN_DISABLE_THINKING:
        return {"type": "disabled"}
    return {"type": "adaptive"}


def get_chat_model(
    model: str = MODEL,
    max_tokens: int = 1024,
    effort: str = "low",
    thinking: bool = False,
) -> ChatAnthropic:
    """Return a LangChain chat model backed by Claude.

    Replaces the notebook's `WatsonxLLM(model=ModelInference(...))`.

    Note: WatsonxLLM was a *text-completion* LLM (string in, string out).
    ChatAnthropic is a *chat model* (messages in, AIMessage out). LangChain
    converts a plain string into a HumanMessage for you, and
    `StrOutputParser` turns the AIMessage back into a string.
    """
    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        effort=effort,
        thinking=thinking_config(thinking, model),
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Return an embedding model served by OpenRouter.

    Replaces the notebook's `WatsonxEmbeddings(model_id="ibm/slate-125m-english-rtrvr-v2")`.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Add it to the .env file at the repo root.")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        # By default OpenAIEmbeddings tokenizes text with OpenAI's tiktoken and
        # sends token IDs. Non-OpenAI models need the raw text, so turn that off.
        check_embedding_ctx_length=False,
    )


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
