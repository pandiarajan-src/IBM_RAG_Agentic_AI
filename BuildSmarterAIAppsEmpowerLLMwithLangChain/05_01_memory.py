"""
05_01_memory.py - Conversation memory in LangChain (+ Exercise 5)
=================================================================

WHO THIS IS FOR
---------------
Students who know basic Python and have heard of LLMs, but are new to LangChain.
Read the file top to bottom: every concept is explained right where it is used,
so you should not need to look anything up elsewhere.

THE BIG IDEA: LLMs HAVE NO MEMORY
---------------------------------
A chat model (Claude, GPT, ...) is STATELESS. Every call is independent: the
model only sees the messages you send in THAT call, and forgets them right after.

So how does ChatGPT "remember" what you said 5 messages ago? It doesn't. The
application re-sends the whole conversation every turn:

    Turn 1 sends: [Human: "My name is Alice"]
    Turn 2 sends: [Human: "My name is Alice", AI: "Hi Alice!", Human: "What's my name?"]
                   ^ the app pasted the old messages back in, so the model "knows"

"Memory" in an LLM app therefore just means:
    (1) STORE the conversation so far, and
    (2) INJECT it into the next prompt.

KEY VOCABULARY
--------------
- Message types: LangChain wraps text in typed messages.
    SystemMessage = instructions for the model ("You are a helpful assistant")
    HumanMessage  = what the user said
    AIMessage     = what the model replied
  A "chat history" is just a list of these.
- Session id: a label (like "alice") that identifies ONE conversation. One app
  can serve many users at once; the session id keeps their histories separate.
- LCEL (LangChain Expression Language): the `|` pipe syntax. `a | b | c` means
  "feed the output of a into b, then the output of b into c" (like Unix pipes).
- Runnable: anything you can `.invoke(input)` on - prompts, models, parsers and
  whole `a | b | c` pipelines are all Runnables. That's why they snap together.

WHAT THIS FILE COVERS
---------------------
1. Chat message history (manual)   - store messages yourself in a list wrapper.
2. Conversation buffer (automatic) - let `RunnableWithMessageHistory` do it.
3. Exercise 5                      - a chatbot with memory, plus a bonus that
                                     compares buffer memory vs window memory.

MEMORY STRATEGIES (a design choice every chatbot makes)
-------------------------------------------------------
- Buffer : keep EVERYTHING. Simple and accurate, but the prompt grows every
           turn, which costs more money and eventually exceeds the model's
           "context window" (the maximum text it can read in one call).
- Window : keep only the last N messages. Cheap and bounded, but the bot
           FORGETS older facts. The Exercise 5 bonus demonstrates this.
- Summary: replace old turns with an LLM-written summary. (Not implemented
           here; you'd add a step that asks the LLM to condense old messages.)

OLD vs NEW LANGCHAIN (useful if you see older tutorials online)
---------------------------------------------------------------
| Notebook (LangChain 0.2)               | LangChain 1.x (used in this file)                      |
|----------------------------------------|--------------------------------------------------------|
| langchain.memory.ChatMessageHistory    | langchain_core.chat_history.InMemoryChatMessageHistory |
| ConversationBufferMemory +             | RunnableWithMessageHistory around a prompt|llm chain   |
|   ConversationChain(verbose=True)      |   (print the history yourself instead of verbose=True) |
| ConversationSummaryMemory              | trim_messages / your own summarization step            |

For production agent apps, LangChain 1.x recommends LangGraph persistence:
`create_agent(..., checkpointer=InMemorySaver())` plus a `thread_id`. Script 06
covers agents. RunnableWithMessageHistory is the simplest way to see the idea.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/05_01_memory.py
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# BaseChatMessageHistory   : the abstract "type" for any history store (used
#                            only as a return-type hint below).
# InMemoryChatMessageHistory: a concrete store that keeps messages in a Python
#                            list in RAM. Lost when the program exits.
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# trim_messages: a helper that cuts a message list down to a size limit
#                (we use it to build "window" memory in the bonus section).
from langchain_core.messages import trim_messages

# StrOutputParser: the model returns an AIMessage object; this parser pulls out
#                  just the text (`.content`) so we get a plain string.
from langchain_core.output_parsers import StrOutputParser

# ChatPromptTemplate : a prompt made of several messages with {placeholders}.
# MessagesPlaceholder: a slot in the prompt that gets filled with a LIST of
#                      messages (this is where the chat history goes).
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# RunnablePassthrough: passes its input along unchanged; `.assign(...)` lets us
#                      modify one key of the input dict on the way through.
from langchain_core.runnables import RunnablePassthrough

# RunnableWithMessageHistory: the wrapper that automates load-history ->
#                      run-chain -> save-history for us.
from langchain_core.runnables.history import RunnableWithMessageHistory

# Project helpers (defined in llm_setup.py): get_chat_model builds the LLM client
# from your environment/API key; print_header prints a section banner.
from llm_setup import get_chat_model, print_header

# Create the model once and reuse it everywhere.
# max_tokens=512 caps the reply length (a "token" is roughly 3/4 of a word),
# which keeps answers short and cheap.
llm = get_chat_model(max_tokens=512)


# ===========================================================================
# 1. CHAT MESSAGE HISTORY (manual)
# ===========================================================================
# Goal: see with your own eyes that "memory" is just a list of messages that
# YOU maintain and pass to the model.
def chat_message_history_demo() -> None:
    print_header("1. CHAT MESSAGE HISTORY")

    # InMemoryChatMessageHistory is a thin wrapper around a Python list.
    history = InMemoryChatMessageHistory()

    # add_ai_message / add_user_message append an AIMessage / HumanMessage.
    # (The order here is deliberately odd - an AI greeting first, then the
    #  user's question - to show you control the list freely.)
    history.add_ai_message("hi!")
    history.add_user_message("what is the capital of France?")

    # `.messages` is the underlying list of message objects.
    print("history:", history.messages)

    # A chat model accepts a list of messages directly. The model reads the
    # whole list as "the conversation so far" and writes the next AI reply.
    ai_response = llm.invoke(history.messages)
    print(f"\nAI: {ai_response.content}")  # `.content` is the reply text

    # IMPORTANT: the model does NOT save its own reply anywhere. If we skip this
    # line, the next turn would have no idea the AI already answered.
    history.add_ai_message(ai_response)

    print(f"\nhistory now has {len(history.messages)} messages:")
    for message in history.messages:
        # `message.type` is "human" or "ai"; `:>5` right-aligns it in 5 chars.
        print(f"  {message.type:>5}: {message.content}")


# ===========================================================================
# 2. CONVERSATION BUFFER MEMORY (automatic)
# ===========================================================================
# Doing the add/append dance by hand gets tedious. RunnableWithMessageHistory
# automates it. On EVERY call it will:
#     a. look up the history for the given session id
#     b. put that history into the prompt's MessagesPlaceholder
#     c. run the chain
#     d. save the new human message + AI reply back into that history
#
# It needs a function that maps session_id -> history object. We keep all
# histories in a dictionary (a "store"), one entry per conversation.
#
# In a real product you'd swap this dict for Redis or a SQL database so
# conversations survive restarts; the rest of the code would not change.
session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return the history for `session_id`, creating an empty one on first use."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def build_conversation(system_prompt: str) -> RunnableWithMessageHistory:
    """Build a chatbot that automatically remembers the whole conversation (buffer memory)."""
    # The prompt has three parts, in order:
    #   1. system message  - the bot's instructions/personality
    #   2. history slot    - all earlier turns get inserted here
    #   3. human message   - the new user input; {input} is a template variable
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),  # past turns are injected here
            ("human", "{input}"),
        ]
    )

    # `prompt | llm | StrOutputParser()` is the base chain:
    #   fill the template -> call the model -> extract the text.
    # RunnableWithMessageHistory wraps it and adds the memory behaviour.
    return RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,  # the function that finds a session's history
        input_messages_key="input",  # which input key holds the NEW user message
        history_messages_key="history",  # which prompt slot receives the history
    )


def conversation_buffer_demo() -> None:
    print_header("2. CONVERSATION BUFFER (RunnableWithMessageHistory)")
    conversation = build_conversation("You are a friendly assistant. Keep answers to two sentences.")

    # The session id travels in `config` under "configurable". The wrapper hands
    # it to get_session_history() to pick the right conversation.
    config = {"configurable": {"session_id": "little-cat"}}

    for user_input in ["Hello, I am a little cat. Who are you?", "What can you do?", "Who am I?"]:
        print(f"\nHuman: {user_input}")
        # We only pass the new message. The wrapper adds the earlier turns for us,
        # which is why the 3rd question ("Who am I?") can be answered: "a little cat".
        print(f"AI: {conversation.invoke({'input': user_input}, config=config)}")

    # A different session id means a fresh, empty memory, so the bot has no idea
    # who "I" is. This is how one app keeps thousands of users' chats separate.
    other = conversation.invoke({"input": "Who am I?"}, config={"configurable": {"session_id": "stranger"}})
    print(f"\n[new session] Human: Who am I?\nAI: {other}")


# ===========================================================================
# 3. EXERCISE 5 - CHATBOT WITH MEMORY
# ===========================================================================
def chat_simulation(conversation, inputs: list[str], session_id: str) -> None:
    """Feed a list of user messages to the conversation one by one and print each reply."""
    print("\n=== Beginning Chat Simulation ===")
    config = {"configurable": {"session_id": session_id}}
    # enumerate(..., start=1) gives us turn numbers starting at 1 instead of 0.
    for i, user_input in enumerate(inputs, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {user_input}")
        print(f"AI: {conversation.invoke({'input': user_input}, config=config)}")
    print("\n=== End of Chat Simulation ===")


def exercise_5_chatbot_with_memory() -> None:
    print_header("3. EXERCISE 5: CHATBOT WITH MEMORY")

    # ---- Part 1: a normal chat that remembers facts ----------------------
    # Pre-load ("seed") the history with an opening exchange. Because the chain
    # reads history by session id, the bot will already "know" Alice's name.
    history = get_session_history("alice")
    history.add_user_message("Hello, my name is Alice.")
    history.add_ai_message("Hello Alice! It's nice to meet you. How can I help you today?")
    print("Initial history:", [f"{m.type}: {m.content}" for m in history.messages])

    # The last two questions test recall: the answers exist only in earlier turns.
    test_inputs = [
        "My favorite color is blue.",
        "I enjoy hiking in the mountains.",
        "What activities would you recommend for me?",
        "What was my favorite color again?",
        "Can you remember both my name and my favorite color?",
    ]
    conversation = build_conversation("You are a helpful assistant. Keep answers brief.")
    chat_simulation(conversation, test_inputs, session_id="alice")

    # Inspect what was stored. [:90] truncates long messages for readable output.
    print("\nFinal Memory Contents:")
    for message in get_session_history("alice").messages:
        print(f"  {message.type:>5}: {message.content[:90]}")

    # ---- Part 2 (bonus): buffer memory vs WINDOW memory ------------------
    # Experiment: give two bots the SAME 6-message history, then ask a question
    # whose answer is in the OLDEST messages.
    #   - buffer bot sees all 6 messages -> can answer
    #   - window bot sees only the last 4 -> the name has fallen out of view
    #
    # We use hard-coded AI replies (instead of calling the model) so the setup
    # is identical and predictable. None of them repeats the name "Alice", so the
    # only place the name exists is the very first human message.
    for session_id in ["buffer-demo", "window-demo"]:
        history = get_session_history(session_id)
        history.add_user_message("Hello, my name is Alice.")  # message 1  <- outside the 4-message window
        history.add_ai_message("Nice to meet you!")  # message 2  <- outside the window
        history.add_user_message("My favorite color is blue.")  # message 3  <- inside the window
        history.add_ai_message("Blue is a lovely color.")  # message 4
        history.add_user_message("I enjoy hiking in the mountains.")  # message 5
        history.add_ai_message("Hiking is great exercise.")  # message 6

    system_prompt = "You are a helpful assistant. Keep answers brief. If you don't know something, say so."
    windowed_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("history"), ("human", "{input}")]
    )

    # trim_messages cuts a message list down to a limit.
    #   max_tokens=4      : keep at most 4 "units"...
    #   token_counter=len : ...where a unit is ONE MESSAGE (len = list length),
    #                       not a real token. Normally you'd count real tokens.
    #   strategy="last"   : keep the most RECENT messages, drop the oldest.
    #   start_on="human"  : the kept list must begin with a human message
    #                       (chat models expect conversations to start with the user).
    trimmer = trim_messages(max_tokens=4, token_counter=len, strategy="last", start_on="human")

    # Build the windowed chain. The first step uses RunnablePassthrough.assign to
    # REPLACE the "history" key with a trimmed copy before the prompt sees it:
    #     input dict {"input": ..., "history": [6 msgs]}
    #       -> assign(history=trim)  -> {"input": ..., "history": [last 4 msgs]}
    #       -> prompt -> llm -> parser
    # `lambda x: ...` is a tiny inline function; x is the input dict.
    windowed_chain = (
        RunnablePassthrough.assign(history=lambda x: trimmer.invoke(x["history"]))
        | windowed_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap it with the same memory machinery. The STORED history stays complete
    # (6 messages); only the copy shown to the model is trimmed.
    window_conversation = RunnableWithMessageHistory(
        windowed_chain, get_session_history, input_messages_key="input", history_messages_key="history"
    )
    buffer_conversation = build_conversation(system_prompt)

    # Ask both bots the same question and compare.
    question = {"input": "What is my name and my favorite color?"}
    print(f"\nHuman: {question['input']}")
    print(
        "[buffer memory, all 6 messages] AI: "
        f"{buffer_conversation.invoke(question, config={'configurable': {'session_id': 'buffer-demo'}})}"
    )
    print(
        "[window memory, last 4 only]    AI: "
        f"{window_conversation.invoke(question, config={'configurable': {'session_id': 'window-demo'}})}"
    )
    # Expected: the buffer bot knows "Alice" and "blue"; the window bot knows only
    # "blue" and admits it doesn't know the name. That's the cost of cheap memory.


def main() -> None:
    chat_message_history_demo()
    conversation_buffer_demo()
    exercise_5_chatbot_with_memory()


# This guard means main() runs only when you execute this file directly,
# not when another file imports it.
if __name__ == "__main__":
    main()
