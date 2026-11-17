"""
01_models_and_messages.py - Models, chat models and chat messages (+ Exercise 1)
================================================================================

LEARNING NOTES
--------------
LangChain is a framework that gives LLM apps standard building blocks (models,
prompts, parsers, retrievers, tools ...) and one standard way to connect them.
The first building block is the **model** itself.

1. Raw model (no framework)
   The notebook called watsonx's `ModelInference.generate()` directly. Here we
   call the Anthropic SDK: `client.messages.create(model=..., messages=[...])`.
   Useful to see what LangChain wraps: a list of messages goes in, a response
   with a list of content blocks comes out.

   Completion vs chat: Llama in the notebook *continued* the text
   "In today's sales meeting, we ...". Claude is a chat model: it answers a
   message rather than autocompleting raw text. To get the same effect, you
   *ask* it to continue the sentence.

2. Chat model (LangChain wrapper)
   `ChatAnthropic` (replaces `WatsonxLLM`) implements LangChain's standard
   chat-model interface. Every LangChain model is a *Runnable*, so they all
   share the same methods: `.invoke()`, `.batch()`, `.stream()` and async
   versions. Swap Claude for another provider and the rest of your code
   stays the same - that's the main value of the abstraction.

3. Chat messages - every message has a role and content:
   - SystemMessage : primes the AI's behaviour. Usually first.
   - HumanMessage  : what the user says.
   - AIMessage     : what the model said (text, or a request to call a tool).
   A chat model takes a LIST of messages and returns ONE new AIMessage.
   Models are *stateless*: to "remember" earlier turns you pass the whole
   history back in each time (see script 05 for doing this automatically).

   AIMessage fields worth knowing:
   - .content           : the answer text (or a list of blocks)
   - .usage_metadata    : input/output token counts, i.e. what you pay for
   - .response_metadata : provider details such as stop_reason and model

4. Exercise 1 - compare responses with different settings
   The notebook compared Granite vs Llama at temperature 0.1 vs 0.8. Claude
   doesn't accept `temperature`. Instead you compare:
   - the MODEL (Haiku = fast/cheap, Sonnet = balanced, Opus = most capable)
   - the EFFORT level (low ... max = how much work the model puts in)

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/01_models_and_messages.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm_setup import MODEL, get_chat_model, get_client, print_header, thinking_config


# ---------------------------------------------------------------------------
# 1. Raw model call with the Anthropic SDK (notebook: ModelInference.generate)
# ---------------------------------------------------------------------------
def raw_model_demo() -> None:
    print_header("1. RAW MODEL CALL (Anthropic SDK, no LangChain)")
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=256,  # watsonx: MAX_NEW_TOKENS
        thinking=thinking_config(False),
        messages=[
            {
                "role": "user",
                "content": "Continue this sentence with one or two more sentences: In today's sales meeting, we ",
            }
        ],
    )
    # watsonx returned msg['results'][0]['generated_text'].
    # Claude returns a list of content blocks; the text lives in the "text" blocks.
    print("".join(block.text for block in response.content if block.type == "text"))
    print(f"\nstop_reason={response.stop_reason}  usage={response.usage.input_tokens} in / {response.usage.output_tokens} out")


# ---------------------------------------------------------------------------
# 2. Chat model via LangChain (notebook: WatsonxLLM(model=model))
# ---------------------------------------------------------------------------
def chat_model_demo() -> None:
    print_header("2. CHAT MODEL (LangChain ChatAnthropic)")
    llm = get_chat_model(max_tokens=256)

    # A plain string is converted to [HumanMessage(content=...)] for you.
    msg = llm.invoke("Who is man's best friend?")
    print(msg.content)


# ---------------------------------------------------------------------------
# 3. Chat messages: System / Human / AI
# ---------------------------------------------------------------------------
def chat_messages_demo() -> None:
    print_header("3. CHAT MESSAGES")
    llm = get_chat_model(max_tokens=256)

    # (a) System message sets the behaviour; Human message asks the question.
    msg = llm.invoke(
        [
            SystemMessage(content="You are a helpful AI bot that assists a user in choosing the perfect book to read in one short sentence"),
            HumanMessage(content="I enjoy mystery novels, what should I read?"),
        ]
    )
    print("--- (a) system + human ---")
    print(f"type: {type(msg).__name__}")  # the model replies with an AIMessage
    print(f"content: {msg.content}")
    print(f"usage_metadata: {msg.usage_metadata}")
    print(f"stop_reason: {msg.response_metadata.get('stop_reason')}\n")

    # (b) Pass a whole conversation, including a previous AI turn. The model
    # reads the history and answers the LAST human message in context:
    # "How often should I attend?" only makes sense because of the CrossFit turn.
    msg = llm.invoke(
        [
            SystemMessage(content="You are a supportive AI bot that suggests fitness activities to a user in one short sentence"),
            HumanMessage(content="I like high-intensity workouts, what should I do?"),
            AIMessage(content="You should try a CrossFit class"),
            HumanMessage(content="How often should I attend?"),
        ]
    )
    print("--- (b) with chat history ---")
    print(f"content: {msg.content}\n")

    # (c) A system message is optional.
    msg = llm.invoke([HumanMessage(content="What month follows June?")])
    print("--- (c) human only ---")
    print(f"content: {msg.content}")


# ---------------------------------------------------------------------------
# 4. Exercise 1 - compare responses across models and effort levels
# ---------------------------------------------------------------------------
def exercise_1_compare_models() -> None:
    print_header("4. EXERCISE 1: COMPARE MODEL RESPONSES")

    # Each config is (label, model, effort). Same idea as the notebook's
    # granite/llama x precise/creative grid, using the knobs Claude exposes.
    configs = [
        ("haiku  / low effort ", "claude-haiku-5-5", "low"),
        ("haiku  / high effort", "claude-haiku-5-5", "high"),
        ("sonnet / low effort ", "claude-sonnet-5-5", "low"),
    ]
    # Three different kinds of prompt, as the exercise asks.
    prompts = [
        "Write a two-line poem about the ocean.",  # creative
        "Explain what a vector database is in one sentence.",  # factual
        "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball? Answer in one line.",  # reasoning
    ]

    for prompt in prompts:
        print(f"\nPROMPT: {prompt}")
        for label, model, effort in configs:
            # Let thinking happen at high effort so the effort setting has room to matter.
            llm = get_chat_model(model=model, max_tokens=2048, effort=effort, thinking=effort != "low")
            # .text gives just the answer text, skipping any thinking blocks.
            print(f"  [{label}] {llm.invoke(prompt).text.strip()}")


def main() -> None:
    raw_model_demo()
    chat_model_demo()
    chat_messages_demo()
    exercise_1_compare_models()


if __name__ == "__main__":
    main()
