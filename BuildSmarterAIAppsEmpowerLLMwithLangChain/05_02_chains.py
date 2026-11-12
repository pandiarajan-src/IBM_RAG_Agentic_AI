"""
05_02_chains.py - Multi-step chains in LangChain (+ Exercise 6)
===============================================================

WHO THIS IS FOR
---------------
Students who know basic Python and have heard of LLMs, but are new to LangChain.
Read the file top to bottom: every concept is explained right where it is used,
so you should not need to look anything up elsewhere.

THE BIG IDEA: A CHAIN IS A PIPELINE OF STEPS
--------------------------------------------
Asking an LLM to do one huge task in a single prompt ("find a dish, write the
recipe, estimate cooking time, ...") often gives sloppy results. It is usually
more reliable to split the work into small, focused steps, each with its own
prompt, and pass results from one step to the next - like an assembly line:

    location --> [LLM: pick a dish] --> meal --> [LLM: write recipe] --> recipe --> [LLM: estimate time] --> time

Benefits: each prompt is simple, you can inspect/debug every intermediate
result, and you can reuse the individual steps elsewhere.

KEY VOCABULARY
--------------
- Prompt template: a string with {placeholders}. Calling it with a dict like
  {"location": "China"} fills the placeholders and produces the final prompt.
- LCEL (LangChain Expression Language): the `|` pipe syntax. `a | b | c` means
  "feed a's output into b, then b's output into c" (like Unix pipes).
- Runnable: anything you can `.invoke(input)` on. Prompts, models, parsers and
  whole `a | b | c` pipelines are all Runnables, which is why they snap together.
- Output parser: converts the model's raw reply object into something handy.
  `StrOutputParser` turns the reply into a plain Python string.

WHAT THIS FILE COVERS
---------------------
4. Simple chain      - one step:  prompt | llm | parser
5. Sequential chain  - several steps where each uses the previous results.
6. Exercise 6        - a 3-step product-review pipeline (sentiment -> summary
                       -> customer response), run on two reviews at once.

HOW STEPS SHARE DATA: RunnablePassthrough.assign
------------------------------------------------
Our data travels through the pipeline as a DICTIONARY.
`RunnablePassthrough.assign(key=some_chain)` does three things:
    1. runs `some_chain` on the current dict,
    2. ADDS the result to the dict under `key`,
    3. keeps every key that was already there.
So the dictionary grows at each step and the final output contains EVERY
intermediate result:

    {"location"}
      -> assign(meal)    -> {"location", "meal"}
      -> assign(recipe)  -> {"location", "meal", "recipe"}
      -> assign(time)    -> {"location", "meal", "recipe", "time"}

OLD vs NEW LANGCHAIN (useful if you see older tutorials online)
---------------------------------------------------------------
| Notebook (LangChain 0.2)                 | LangChain 1.x (LCEL, used in this file)           |
|------------------------------------------|---------------------------------------------------|
| LLMChain(llm, prompt, output_key="meal") | prompt | llm | StrOutputParser()                  |
| SequentialChain(chains=[...],            | RunnablePassthrough.assign(meal=...)              |
|   input_variables, output_variables)     |   | RunnablePassthrough.assign(recipe=...) | ...   |
| verbose=True                             | print intermediate keys / use LangSmith tracing   |

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/05_02_chains.py
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# pprint: "pretty print" - prints dicts/lists in a readable, wrapped layout.
from pprint import pprint

# StrOutputParser: pulls the plain text out of the model's reply object.
from langchain_core.output_parsers import StrOutputParser

# PromptTemplate: a single text prompt with {placeholders} to fill in.
from langchain_core.prompts import PromptTemplate

# RunnablePassthrough: passes its input along unchanged; `.assign(...)` adds
# new keys to the dict (explained in the module docstring above).
from langchain_core.runnables import RunnablePassthrough

# Project helpers (defined in llm_setup.py): get_chat_model builds the LLM client
# from your environment/API key; print_header prints a section banner.
from llm_setup import get_chat_model, print_header

# Create the model once and reuse it in every chain.
# max_tokens=512 caps the reply length (a "token" is roughly 3/4 of a word),
# which keeps answers short and cheap.
llm = get_chat_model(max_tokens=512)


# ===========================================================================
# PROMPT TEMPLATES for the food example (sections 4 and 5)
# ===========================================================================
# Each {name} is a placeholder that is filled from the input dictionary's key
# of the same name. Notice how each template needs the OUTPUT of the previous step:
#   location_template needs {location}  (the user's input)
#   dish_template     needs {meal}      (produced by step 1)
#   time_template     needs {recipe}    (produced by step 2)
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


# ===========================================================================
# 4. SIMPLE CHAIN
# ===========================================================================
def simple_chain_demo() -> None:
    print_header("4. SIMPLE CHAIN (LCEL)")

    # Read the pipe left to right:
    #   PromptTemplate : fills {location} -> produces the final prompt text
    #   llm            : sends the prompt to the model -> returns an AIMessage
    #   StrOutputParser: extracts the text from the AIMessage -> a plain string
    location_chain = PromptTemplate.from_template(location_template) | llm | StrOutputParser()

    # .invoke() runs the whole pipeline once. The dict's keys must match the
    # template's placeholders ("location").
    print(location_chain.invoke({"location": "China"}))


# ===========================================================================
# 5. SEQUENTIAL CHAIN: location -> meal -> recipe -> time
# ===========================================================================
def sequential_chain_demo() -> None:
    print_header("5. SEQUENTIAL CHAIN (RunnablePassthrough.assign)")

    # Three independent single-step chains (each = prompt | llm | parser).
    location_chain = PromptTemplate.from_template(location_template) | llm | StrOutputParser()
    dish_chain = PromptTemplate.from_template(dish_template) | llm | StrOutputParser()
    time_chain = PromptTemplate.from_template(time_template) | llm | StrOutputParser()

    # Link them together. Each `assign` step:
    #   - hands the WHOLE current dict to its sub-chain,
    #   - the sub-chain's prompt picks out only the placeholder it needs
    #     (extra keys in the dict are simply ignored),
    #   - the reply is stored under the new key (meal / recipe / time).
    # The original notebook wrapped each step in a lambda; that isn't necessary.
    overall_chain = (
        RunnablePassthrough.assign(meal=location_chain)  # uses {location}
        | RunnablePassthrough.assign(recipe=dish_chain)  # uses {meal}, made one step earlier
        | RunnablePassthrough.assign(time=time_chain)  # uses {recipe}, made one step earlier
    )

    # The result is a dict holding the input AND all three intermediate outputs.
    # width=100 tells pprint it may use lines up to 100 characters wide.
    pprint(overall_chain.invoke({"location": "China"}), width=100)


# ===========================================================================
# 6. EXERCISE 6 - MULTI-STEP PRODUCT REVIEW PROCESSING
# ===========================================================================
# Scenario: a company receives customer reviews and wants to automatically
#   (1) classify the sentiment, (2) summarise the key points, and
#   (3) draft a reply to the customer.
# Each step builds on the previous ones, so this is a natural sequential chain.

# Two sample inputs: one clearly happy customer, one clearly unhappy.
# Triple-quoted strings (""" ... """) let us write multi-line text directly.
positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing.
The built-in grinder saves me so much time in the morning, and the programmable timer means
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use,
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours.
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

# Step 1 prompt. Needs: {review}. Asking for a fixed output format
# ("SENTIMENT: [...]") makes the answer predictable and easy to parse.
sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

# Step 2 prompt. Needs: {review} (original input) and {sentiment} (from step 1).
summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

# Step 3 prompt. Needs: {review}, {sentiment} (step 1) and {summary} (step 2).
# The instructions branch on sentiment, so the reply tone adapts automatically.
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

    # One single-step chain per prompt template.
    sentiment_chain = PromptTemplate.from_template(sentiment_template) | llm | StrOutputParser()
    summary_chain = PromptTemplate.from_template(summary_template) | llm | StrOutputParser()
    response_chain = PromptTemplate.from_template(response_template) | llm | StrOutputParser()

    # Chain them. The dict grows: {review} -> +sentiment -> +summary -> +response.
    # The comments show which keys each step can read at that moment.
    review_chain = (
        RunnablePassthrough.assign(sentiment=sentiment_chain)  # needs: review
        | RunnablePassthrough.assign(summary=summary_chain)  # needs: review, sentiment
        | RunnablePassthrough.assign(response=response_chain)  # needs: review, sentiment, summary
    )

    # .batch() runs the whole pipeline on a LIST of inputs and returns a list of
    # results. The inputs are processed concurrently (in parallel), which is
    # faster than calling .invoke() on each review one after another.
    results = review_chain.batch([{"review": positive_review}, {"review": negative_review}])

    for result in results:
        print("\n" + "=" * 50)
        # [:100] shows only the first 100 characters so the output stays short.
        print(f"REVIEW:\n{result['review'][:100]}...\n")
        # .strip() removes stray leading/trailing whitespace/newlines from the LLM text.
        print(result["sentiment"].strip())
        print(f"\nSUMMARY:\n{result['summary'].strip()}")
        print(f"\nRESPONSE:\n{result['response'].strip()}")


def main() -> None:
    simple_chain_demo()
    sequential_chain_demo()
    exercise_6_review_pipeline()


# This guard means main() runs only when you execute this file directly,
# not when another file imports it.
if __name__ == "__main__":
    main()
