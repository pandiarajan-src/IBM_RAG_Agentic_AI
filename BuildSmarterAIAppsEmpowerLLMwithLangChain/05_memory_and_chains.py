"""
05_memory_and_chains.py - Conversation memory and multi-step chains (+ Exercises 5 and 6)
=========================================================================================

LEARNING NOTES
--------------
PART A - MEMORY
---------------
Chat models are STATELESS: each call only sees the messages you send in it. A
chatbot "remembers" only because the app re-sends the conversation each turn.
Memory = storing that history and injecting it into the next prompt.

1. Chat message history (langchain_core.chat_history.InMemoryChatMessageHistory)
   A thin list wrapper: `add_user_message()`, `add_ai_message()`, `.messages`.
   You can pass `.messages` straight to the model.

2. Automatic memory with `RunnableWithMessageHistory`
   Wraps any chain and, on every call:
     a. loads the history for a session id  ->  b. puts it in the prompt's
        MessagesPlaceholder  ->  c. runs the chain  ->  d. saves the new
        human + AI messages back.
   The session id (passed in `config`) lets one app keep separate conversations
   per user.

3. Memory strategies
   - Buffer: keep everything. Simple, but the prompt grows every turn
     (cost and context-window limits).
   - Window: keep only the last N messages (`trim_messages`). Cheap, but it
     forgets older facts. Exercise 5's bonus shows this happening.
   - Summary: replace old turns with an LLM-written summary.

| Notebook (LangChain 0.2)               | LangChain 1.x                                            |
|----------------------------------------|----------------------------------------------------------|
| langchain.memory.ChatMessageHistory    | langchain_core.chat_history.InMemoryChatMessageHistory   |
| ConversationBufferMemory +             | RunnableWithMessageHistory around a prompt|llm chain     |
|   ConversationChain(verbose=True)      |   (print history yourself instead of verbose=True)       |
| ConversationSummaryMemory              | trim_messages / your own summarization step              |

For full agent apps, the LangChain 1.x recommendation is LangGraph persistence:
`create_agent(..., checkpointer=InMemorySaver())` + a `thread_id`. Script 06
covers agents. RunnableWithMessageHistory is the simplest way to see the idea.

PART B - CHAINS
---------------
A chain is a pipeline of steps. Breaking one big task into focused steps (each
with its own prompt) gives more reliable output than one giant prompt.

- Simple chain: `prompt | llm | StrOutputParser()`
- Sequential chain: the output of step 1 feeds step 2, and so on. In LCEL use
  `RunnablePassthrough.assign(key=chain)`: it runs `chain` on the current dict
  and ADDS the result under `key`, keeping every earlier key. So the final dict
  holds every intermediate result, like SequentialChain's `output_variables`.

      {"location"} -> assign(meal) -> {"location","meal"} -> assign(recipe) -> ... -> assign(time)

| Notebook (LangChain 0.2)                 | LangChain 1.x (LCEL)                              |
|------------------------------------------|---------------------------------------------------|
| LLMChain(llm, prompt, output_key="meal") | prompt | llm | StrOutputParser()                  |
| SequentialChain(chains=[...],            | RunnablePassthrough.assign(meal=...)              |
|   input_variables, output_variables)     |   | RunnablePassthrough.assign(recipe=...) | ...   |
| verbose=True                             | print intermediate keys / use LangSmith tracing   |

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/05_memory_and_chains.py
"""

from pprint import pprint

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm_setup import get_chat_model, print_header

llm = get_chat_model(max_tokens=512)


# ===========================================================================
# PART A - MEMORY
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Chat message history (manual)
# ---------------------------------------------------------------------------
def chat_message_history_demo() -> None:
    print_header("1. CHAT MESSAGE HISTORY")
    history = InMemoryChatMessageHistory()
    history.add_ai_message("hi!")
    history.add_user_message("what is the capital of France?")
    print("history:", history.messages)

    ai_response = llm.invoke(history.messages)
    print(f"\nAI: {ai_response.content}")

    # You must save the reply yourself, or the next turn won't know about it.
    history.add_ai_message(ai_response)
    print(f"\nhistory now has {len(history.messages)} messages:")
    for message in history.messages:
        print(f"  {message.type:>5}: {message.content}")


# ---------------------------------------------------------------------------
# 2. Conversation buffer memory (automatic, via RunnableWithMessageHistory)
# ---------------------------------------------------------------------------
# One history per session id. Swap this dict for Redis/SQL to persist across runs.
session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def build_conversation(system_prompt: str) -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),  # past turns are injected here
            ("human", "{input}"),
        ]
    )
    return RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",  # which input key is the new user message
        history_messages_key="history",  # which prompt slot receives the history
    )


def conversation_buffer_demo() -> None:
    print_header("2. CONVERSATION BUFFER (RunnableWithMessageHistory)")
    conversation = build_conversation("You are a friendly assistant. Keep answers to two sentences.")
    config = {"configurable": {"session_id": "little-cat"}}

    for user_input in ["Hello, I am a little cat. Who are you?", "What can you do?", "Who am I?"]:
        print(f"\nHuman: {user_input}")
        print(f"AI: {conversation.invoke({'input': user_input}, config=config)}")

    # A different session id means a fresh, empty memory.
    other = conversation.invoke({"input": "Who am I?"}, config={"configurable": {"session_id": "stranger"}})
    print(f"\n[new session] Human: Who am I?\nAI: {other}")


# ---------------------------------------------------------------------------
# 3. Exercise 5 - chatbot with memory
# ---------------------------------------------------------------------------
def chat_simulation(conversation, inputs: list[str], session_id: str) -> None:
    """Run a series of inputs through the conversation and display responses."""
    print("\n=== Beginning Chat Simulation ===")
    config = {"configurable": {"session_id": session_id}}
    for i, user_input in enumerate(inputs, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {user_input}")
        print(f"AI: {conversation.invoke({'input': user_input}, config=config)}")
    print("\n=== End of Chat Simulation ===")


def exercise_5_chatbot_with_memory() -> None:
    print_header("3. EXERCISE 5: CHATBOT WITH MEMORY")
    # Seed the history with an opening exchange, as the notebook does.
    history = get_session_history("alice")
    history.add_user_message("Hello, my name is Alice.")
    history.add_ai_message("Hello Alice! It's nice to meet you. How can I help you today?")
    print("Initial history:", [f"{m.type}: {m.content}" for m in history.messages])

    test_inputs = [
        "My favorite color is blue.",
        "I enjoy hiking in the mountains.",
        "What activities would you recommend for me?",
        "What was my favorite color again?",
        "Can you remember both my name and my favorite color?",
    ]
    conversation = build_conversation("You are a helpful assistant. Keep answers brief.")
    chat_simulation(conversation, test_inputs, session_id="alice")

    print("\nFinal Memory Contents:")
    for message in get_session_history("alice").messages:
        print(f"  {message.type:>5}: {message.content[:90]}")

    # Bonus (the notebook's optional step 8): buffer memory vs WINDOW memory.
    # Same stored history, but the window only shows the model the last 4 messages.
    # (Fixed AI replies keep the demo predictable: none of them repeats the name.)
    for session_id in ["buffer-demo", "window-demo"]:
        history = get_session_history(session_id)
        history.add_user_message("Hello, my name is Alice.")
        history.add_ai_message("Nice to meet you!")
        history.add_user_message("My favorite color is blue.")
        history.add_ai_message("Blue is a lovely color.")
        history.add_user_message("I enjoy hiking in the mountains.")
        history.add_ai_message("Hiking is great exercise.")

    system_prompt = "You are a helpful assistant. Keep answers brief. If you don't know something, say so."
    windowed_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("history"), ("human", "{input}")]
    )
    # token_counter=len means "count messages, not tokens".
    trimmer = trim_messages(max_tokens=4, token_counter=len, strategy="last", start_on="human")
    windowed_chain = (
        RunnablePassthrough.assign(history=lambda x: trimmer.invoke(x["history"]))
        | windowed_prompt
        | llm
        | StrOutputParser()
    )
    window_conversation = RunnableWithMessageHistory(
        windowed_chain, get_session_history, input_messages_key="input", history_messages_key="history"
    )
    buffer_conversation = build_conversation(system_prompt)

    question = {"input": "What is my name and my favorite color?"}
    print(f"\nHuman: {question['input']}")
    print(f"[buffer memory, all 6 messages] AI: {buffer_conversation.invoke(question, config={'configurable': {'session_id': 'buffer-demo'}})}")
    print(f"[window memory, last 4 only]    AI: {window_conversation.invoke(question, config={'configurable': {'session_id': 'window-demo'}})}")


# ===========================================================================
# PART B - CHAINS
# ===========================================================================
location_template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}

YOUR RESPONSE (just the dish name and one sentence):
"""
dish_template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

YOUR RESPONSE:
"""
time_template = """Given the recipe {recipe}, estimate how much time I need to cook it. Answer in one or two sentences.

YOUR RESPONSE:
"""


# ---------------------------------------------------------------------------
# 4. Simple chain
# ---------------------------------------------------------------------------
def simple_chain_demo() -> None:
    print_header("4. SIMPLE CHAIN (LCEL)")
    location_chain = PromptTemplate.from_template(location_template) | llm | StrOutputParser()
    print(location_chain.invoke({"location": "China"}))


# ---------------------------------------------------------------------------
# 5. Sequential chain: location -> meal -> recipe -> time
# ---------------------------------------------------------------------------
def sequential_chain_demo() -> None:
    print_header("5. SEQUENTIAL CHAIN (RunnablePassthrough.assign)")
    location_chain = PromptTemplate.from_template(location_template) | llm | StrOutputParser()
    dish_chain = PromptTemplate.from_template(dish_template) | llm | StrOutputParser()
    time_chain = PromptTemplate.from_template(time_template) | llm | StrOutputParser()

    # Each sub-chain gets the WHOLE dict; its prompt picks out the key it needs.
    # (The notebook wrapped each step in a lambda; that isn't necessary.)
    overall_chain = (
        RunnablePassthrough.assign(meal=location_chain)
        | RunnablePassthrough.assign(recipe=dish_chain)
        | RunnablePassthrough.assign(time=time_chain)
    )
    pprint(overall_chain.invoke({"location": "China"}), width=100)


# ---------------------------------------------------------------------------
# 6. Exercise 6 - multi-step product review processing
# ---------------------------------------------------------------------------
positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing.
The built-in grinder saves me so much time in the morning, and the programmable timer means
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use,
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours.
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

response_template = """Write a short, helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""


def exercise_6_review_pipeline() -> None:
    print_header("6. EXERCISE 6: REVIEW ANALYSIS PIPELINE")
    sentiment_chain = PromptTemplate.from_template(sentiment_template) | llm | StrOutputParser()
    summary_chain = PromptTemplate.from_template(summary_template) | llm | StrOutputParser()
    response_chain = PromptTemplate.from_template(response_template) | llm | StrOutputParser()

    review_chain = (
        RunnablePassthrough.assign(sentiment=sentiment_chain)  # needs: review
        | RunnablePassthrough.assign(summary=summary_chain)  # needs: review, sentiment
        | RunnablePassthrough.assign(response=response_chain)  # needs: review, sentiment, summary
    )

    # .batch() runs the whole pipeline for several inputs concurrently.
    results = review_chain.batch([{"review": positive_review}, {"review": negative_review}])
    for result in results:
        print("\n" + "=" * 50)
        print(f"REVIEW:\n{result['review'][:100]}...\n")
        print(result["sentiment"].strip())
        print(f"\nSUMMARY:\n{result['summary'].strip()}")
        print(f"\nRESPONSE:\n{result['response'].strip()}")


def main() -> None:
    chat_message_history_demo()
    conversation_buffer_demo()
    exercise_5_chatbot_with_memory()
    simple_chain_demo()
    sequential_chain_demo()
    exercise_6_review_pipeline()


if __name__ == "__main__":
    main()
