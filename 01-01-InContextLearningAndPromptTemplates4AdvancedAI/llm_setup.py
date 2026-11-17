"""
llm_setup.py - Shared LLM helpers used by every script in this folder.
=====================================================================

LEARNING NOTES
--------------
The original IBM lab used `ibm-watsonx-ai` + `langchain-ibm` (WatsonxLLM) with a
Granite / Llama model. Here we use Anthropic's Claude instead, in two flavours:

1. The raw **Anthropic Python SDK** (`anthropic`)  -> `llm_model()`
   Used in scripts 01 and 02 so you can see *exactly* what goes over the wire:
   a list of messages in, a list of content blocks out. No framework magic.

2. **LangChain's `ChatAnthropic`** (`langchain-anthropic`) -> `get_chat_model()`
   Used in scripts 03 and 04 so you can compose prompt templates, models and
   output parsers into chains with the LCEL pipe operator `|`.

Generation parameters: watsonx vs Claude
---------------------------------------
| watsonx (notebook)  | Claude equivalent         | Notes                                   |
|---------------------|---------------------------|-----------------------------------------|
| max_new_tokens      | max_tokens                | Hard cap on output length.              |
| min_new_tokens      | (none)                    | Ask for length in the prompt instead.   |
| temperature         | (removed)                 | Current Claude models reject it (400).  |
| top_p / top_k       | (removed)                 | Same - sampling knobs are not exposed.  |
| stop_sequences      | stop_sequences            | Stop generating when a string appears.  |
| (none)              | system                    | Role / behaviour instructions.          |
| (none)              | output_config.effort      | low | medium | high | xhigh | max.      |
| (none)              | thinking                  | Let the model reason before answering.  |

Why no temperature? Modern Claude models tune sampling internally. You steer
the output with *the prompt itself* (that's what this whole lab is about!),
with `effort` (how much work the model puts in), and with `max_tokens`.

Thinking and max_tokens
-----------------------
Claude Haiku 5.5 "thinks" (reasons privately) by default, and thinking tokens
count against `max_tokens`. The notebook used tiny limits such as
`max_new_tokens=10`; with thinking on, that budget could be used up before any
visible answer appears. So `llm_model()` turns thinking OFF by default, which
behaves like the notebook's Granite model. Script 02 turns it ON to compare.

Setup
-----
1. Add the packages from the repo root (already done for this repo):
       uv add anthropic langchain langchain-core langchain-anthropic python-dotenv
2. Create `<repo root>/.env` containing:
       ANTHROPIC_API_KEY=sk-ant-...
3. Run any script from the repo root, e.g.:
       uv run python InContextLearningAndPromptTemplates4AdvancedAI/01_prompting_basics.py
"""

from pathlib import Path
import os
import anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load ANTHROPIC_API_KEY from the .env file at the repo root (one level up).
# Both the Anthropic SDK and ChatAnthropic read the key from the environment.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Change the model in ONE place. Haiku is the fastest/cheapest - ideal for
# learning. Swap to "claude-sonnet-5-5" or "claude-opus-5-5" to compare quality.
MODEL = os.environ.get("MODEL", "claude-haiku-5-5")

# Models that accept `thinking={"type": "disabled"}`. Claude Opus 5.5 and
# Claude Sonnet 5.5 always think; for them we leave thinking on and keep effort low.
MODELS_THAT_CAN_DISABLE_THINKING = {"claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5"}

_client = None


def get_client() -> anthropic.Anthropic:
    """Return a single shared Anthropic client (created on first use)."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def thinking_config(thinking: bool) -> dict:
    if not thinking and MODEL in MODELS_THAT_CAN_DISABLE_THINKING:
        return {"type": "disabled"}
    return {"type": "adaptive"}


def llm_model(
    prompt_txt: str,
    max_tokens: int = 512,
    system: str | None = None,
    effort: str = "low",
    thinking: bool = False,
    stop_sequences: list[str] | None = None,
) -> str:
    """Send one prompt to Claude and return the text answer.

    This replaces the notebook's `llm_model(prompt_txt, params)` helper.

    Args:
        prompt_txt: The user prompt (the thing you're "engineering").
        max_tokens: Max output tokens (watsonx: max_new_tokens).
        system: Optional system prompt - sets role, tone and rules.
        effort: "low" | "medium" | "high" | "xhigh" | "max". Higher means more
            careful (and slower, pricier) answers.
        thinking: Let the model reason privately before it answers.
        stop_sequences: Stop as soon as any of these strings is produced.
    """
    request = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt_txt}],
        "thinking": thinking_config(thinking),
        "output_config": {"effort": effort},
    }
    if system:
        request["system"] = system
    if stop_sequences:
        request["stop_sequences"] = stop_sequences

    response = get_client().messages.create(**request)

    if response.stop_reason == "refusal":
        return f"[Claude declined this request: {response.stop_details}]"

    # response.content is a list of blocks (thinking, text, ...). Keep only text.
    text = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        text += "  [...truncated: hit max_tokens]"
    return text.strip()


def get_chat_model(max_tokens: int = 1024, effort: str = "low", thinking: bool = False) -> ChatAnthropic:
    """Return a LangChain chat model backed by Claude.

    Replaces the notebook's `WatsonxLLM(model_id=..., params=...)`.

    Note: WatsonxLLM was a *text-completion* LLM (string in, string out).
    ChatAnthropic is a *chat model* (messages in, AIMessage out). LangChain
    converts a plain string prompt into a HumanMessage for you, and
    `StrOutputParser` turns the AIMessage back into a string, so chains
    still read `prompt | llm | StrOutputParser()`.
    """
    return ChatAnthropic(
        model=MODEL,
        max_tokens=max_tokens,
        effort=effort,
        thinking=thinking_config(thinking),
    )


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(prompt: str, response: str) -> None:
    """Print a prompt/response pair the way the notebook did."""
    print(f"prompt: {prompt.strip()}\n")
    print(f"response : {response}\n")
