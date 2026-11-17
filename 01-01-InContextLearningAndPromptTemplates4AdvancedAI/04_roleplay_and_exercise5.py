"""
04_roleplay_and_exercise5.py - Role-playing chatbot + product review analyzer
=============================================================================

LEARNING NOTES
--------------
PART A - Role playing
---------------------
You can give the LLM a *persona* (role), a *tone* and rules, and it will act
as a task-focused chatbot. Making these template variables means you can
switch the bot's personality by changing one value instead of rewriting the
prompt.

Upgrades over the notebook's version:
- **ChatPromptTemplate** instead of PromptTemplate. Chat models understand
  message *roles*. The persona goes in the **system** message (lasting
  instructions) and the user's question goes in a **human** message. Models
  follow system instructions more consistently than text mixed into the question.
- **Conversation memory**. The notebook sent each question on its own, so the
  bot forgot everything between turns. The API is stateless: to "remember",
  you resend the history every turn. `MessagesPlaceholder("history")` is the
  slot in the template where that history goes.

PART B - Exercise 5: product review analyzer
--------------------------------------------
B1. The notebook's solution: a template that asks for a fixed text format,
    then prompt | llm | StrOutputParser(). Here every review is processed
    in one `.batch()` call, which runs them in parallel.
B2. Structured output: `llm.with_structured_output(PydanticModel)` makes
    the model return a validated Python object instead of free text. No
    fragile string parsing, so this is how you'd build it for production.

Run (from the repo root):
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/04_roleplay_and_exercise5.py           # interactive chat, then Exercise 5
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/04_roleplay_and_exercise5.py --demo    # canned questions, no typing
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/04_roleplay_and_exercise5.py --only-exercise
"""

import argparse
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field

from llm_setup import get_chat_model, print_header


# ---------------------------------------------------------------------------
# PART A - Role-playing chatbot
# ---------------------------------------------------------------------------
roleplay_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert {role}. Keep the conversation {tone}. "
     "Stay in character and keep answers under 150 words."),
    MessagesPlaceholder("history"),   # earlier turns are inserted here
    ("human", "{question}"),
])

DEMO_QUESTIONS = [
    "Who are you?",
    "What are the basic rules of Dungeons & Dragons?",
    "Can you describe a mysterious forest setting for my adventure?",
    "Based on that forest, what's a good puzzle I could use?",  # needs memory of the previous answer
]


def roleplay_chat(demo: bool) -> None:
    print_header("PART A - ROLE-PLAYING CHATBOT (D&D game master)")
    role = "Dungeons & Dragons game master"
    tone = "engaging and immersive"

    roleplay_chain = roleplay_prompt | get_chat_model(max_tokens=512) | StrOutputParser()
    history = []  # list of HumanMessage / AIMessage - the bot's "memory"

    def ask(query: str) -> None:
        response = roleplay_chain.invoke(
            {"role": role, "tone": tone, "history": history, "question": query}
        )
        print(f"Answer: {response}\n")
        history.extend([HumanMessage(query), AIMessage(response)])

    if demo:
        for query in DEMO_QUESTIONS:
            print(f"Question: {query}")
            ask(query)
        return

    print('Ask the game master anything. Type "quit", "exit" or "bye" to stop.\n')
    while True:
        query = input("Question: ").strip()
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
        if query:
            ask(query)


# ---------------------------------------------------------------------------
# PART B - Exercise 5: product review analyzer
# ---------------------------------------------------------------------------
REVIEWS = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful.",
]


def review_analyzer_text() -> None:
    """B1 - Exercise 5 as in the notebook: template -> LLM -> string."""
    print_header("PART B1 - EXERCISE 5: review analyzer (text output)")

    # Notebook: llama-3-3-70b with MAX_NEW_TOKENS=512 and TEMPERATURE=0.2.
    # A low temperature asked for consistent output. Claude has no temperature,
    # so the fixed output format in the template does that job.
    llm = get_chat_model(max_tokens=512)

    product_review_prompt = PromptTemplate.from_template(
        """Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)"""
    )
    review_analysis_chain = product_review_prompt | llm | StrOutputParser()

    # .batch() runs the chain on many inputs at once (in parallel threads).
    results = review_analysis_chain.batch([{"review": r} for r in REVIEWS])
    for i, result in enumerate(results, start=1):
        print(f"==== Review #{i} ====")
        print(result)
        print()


class ReviewAnalysis(BaseModel):
    """Structured analysis of a single product review."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of the review")
    features: list[str] = Field(description="Product features mentioned, e.g. 'camera', 'battery'")
    summary: str = Field(description="One-sentence summary of the review")


def review_analyzer_structured() -> None:
    """B2 - 'Try modifying the chain to change the output format' -> typed objects.

    with_structured_output() sends the Pydantic schema to Claude, which must
    return JSON matching it (method="json_schema" uses Claude's native
    structured outputs). LangChain then validates that JSON into a
    ReviewAnalysis instance.
    """
    print_header("PART B2 - EXERCISE 5: review analyzer (structured output)")

    structured_llm = get_chat_model(max_tokens=512).with_structured_output(
        ReviewAnalysis, method="json_schema"
    )
    prompt = PromptTemplate.from_template("Analyze this product review:\n\n{review}")
    chain = prompt | structured_llm  # no StrOutputParser: output is a ReviewAnalysis

    analyses: list[ReviewAnalysis] = chain.batch([{"review": r} for r in REVIEWS])
    for i, analysis in enumerate(analyses, start=1):
        print(f"==== Review #{i} ====")
        print(f"sentiment: {analysis.sentiment}")
        print(f"features : {', '.join(analysis.features)}")
        print(f"summary  : {analysis.summary}")
        print()

    # Because the result is data, you can work with it directly in code:
    negatives = [a for a in analyses if a.sentiment == "negative"]
    print(f"{len(negatives)} of {len(analyses)} reviews are negative.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--demo", action="store_true", help="run the chatbot with canned questions instead of input()")
    parser.add_argument("--only-exercise", action="store_true", help="skip the chatbot, run only Exercise 5")
    args = parser.parse_args()

    if not args.only_exercise:
        roleplay_chat(demo=args.demo)

    review_analyzer_text()
    review_analyzer_structured()


if __name__ == "__main__":
    main()
