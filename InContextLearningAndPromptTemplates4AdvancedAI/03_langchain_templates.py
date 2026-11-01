"""
03_langchain_templates.py - LangChain PromptTemplates + LCEL applications
=========================================================================

LEARNING NOTES
--------------
**LangChain** is a framework that gives LLM apps standard building blocks
(prompts, models, output parsers, retrievers, tools ...) and a standard way to
connect them.

**PromptTemplate** is a reusable prompt with `{placeholders}`. You write the
prompt once and fill in different values each time:
    PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")

**LCEL (LangChain Expression Language)** chains components with the pipe `|`:

    chain = prompt | llm | StrOutputParser()
    chain.invoke({"adjective": "funny", "content": "chickens"})

    dict --> PromptTemplate --> PromptValue --> ChatAnthropic --> AIMessage --> StrOutputParser --> str

Every component is a *Runnable*, so every chain gets the same methods:
`.invoke()` (one input), `.batch()` (many inputs), `.stream()` (token by
token), plus async versions (`.ainvoke()` ...).

Changes from the notebook (LangChain 0.2 -> 1.x):
- `from langchain.chains import LLMChain` is gone. LCEL replaces it.
- The notebook wrapped formatting in `RunnableLambda(format_prompt)`. That
  isn't needed: a PromptTemplate IS a Runnable and can be piped directly.
  The notebook's helper also had a hidden bug risk: `format_prompt` read the
  *global* `prompt` variable, so every chain silently used whichever template
  was assigned last. Piping the template object makes each chain own its prompt.
- `WatsonxLLM` (text LLM) -> `ChatAnthropic` (chat model). A string prompt
  is turned into a single HumanMessage for you automatically.

The application pattern used by every section below:
  1. Define the content/problem.
  2. Write a template with {variables}.
  3. PromptTemplate.from_template(template).
  4. chain = prompt | llm | StrOutputParser()
  5. chain.invoke({...variables...})

Run (from the repo root):
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/03_langchain_templates.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from llm_setup import get_chat_model, print_header

# One chat model shared by every chain (notebook: max_new_tokens=256).
llm = get_chat_model(max_tokens=512)


# ---------------------------------------------------------------------------
# 1. PromptTemplate basics
# ---------------------------------------------------------------------------
def prompt_template_basics() -> None:
    print_header("1. PROMPT TEMPLATE BASICS")
    prompt = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")

    # from_template() finds the variables for you.
    print(f"input_variables: {prompt.input_variables}")

    # .format() -> plain string. Handy for checking what will be sent.
    print(f"format():  {prompt.format(adjective='funny', content='chickens')!r}")

    # .invoke() -> PromptValue. This is what flows through a chain. It can
    # become either a string (for text LLMs) or messages (for chat models).
    value = prompt.invoke({"adjective": "funny", "content": "chickens"})
    print(f"invoke():  {value!r}")
    print(f"as messages: {value.to_messages()}\n")

    # .partial() pre-fills some variables -> a more specialised template.
    chicken_jokes = prompt.partial(content="chickens")
    print(f"partial -> still needs: {chicken_jokes.input_variables}\n")


# ---------------------------------------------------------------------------
# 2. First LCEL chain: jokes
# ---------------------------------------------------------------------------
def joke_chain_demo() -> None:
    print_header("2. LCEL CHAIN - JOKES")
    prompt = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
    joke_chain = prompt | llm | StrOutputParser()

    print(joke_chain.invoke({"adjective": "funny", "content": "chickens"}))
    print()
    # Same chain, new context - just change the variables.
    print(joke_chain.invoke({"adjective": "sad", "content": "fish"}))
    print()


# ---------------------------------------------------------------------------
# 3. Text summarization
# ---------------------------------------------------------------------------
def summarization_demo() -> None:
    print_header("3. TEXT SUMMARIZATION")
    content = """
The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation.
Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems.
For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion.
Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers.
These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""
    # Tip: delimit the input (here with <text> tags) so the model can tell
    # your instructions apart from the content it should work on.
    prompt = PromptTemplate.from_template(
        "Summarize the text below in one sentence.\n\n<text>\n{content}\n</text>"
    )
    summarize_chain = prompt | llm | StrOutputParser()
    print(summarize_chain.invoke({"content": content}))
    print()


# ---------------------------------------------------------------------------
# 4. Question answering (grounded in provided context)
# ---------------------------------------------------------------------------
# This is the core idea behind RAG (retrieval-augmented generation): give the
# model the context, ask it to answer ONLY from that context, and give it a
# way out ("Unsure about answer") so it doesn't make things up.
def question_answering_demo() -> None:
    print_header("4. QUESTION ANSWERING")
    content = """
The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets.
The inner planets-Mercury, Venus, Earth, and Mars-are rocky and solid.
The outer planets-Jupiter, Saturn, Uranus, and Neptune-are much larger and gaseous.
"""
    prompt = PromptTemplate.from_template(
        """Answer the question based only on the content below.
Respond "Unsure about answer" if the content does not contain the answer.

<content>
{content}
</content>

Question: {question}
Answer:"""
    )
    qa_chain = prompt | llm | StrOutputParser()

    for question in [
        "Which planets in the solar system are rocky and solid?",
        "How many moons does Jupiter have?",  # not in the content -> should be "Unsure"
    ]:
        print(f"Q: {question}")
        print(f"A: {qa_chain.invoke({'question': question, 'content': content})}\n")


# ---------------------------------------------------------------------------
# 5. Text classification (zero-shot) + few-shot via FewShotPromptTemplate
# ---------------------------------------------------------------------------
def classification_demo() -> None:
    print_header("5. TEXT CLASSIFICATION")
    text = "The concert last night was an exhilarating experience with outstanding performances by all artists."
    categories = "Entertainment, Food and Dining, Technology, Literature, Music"

    # Zero-shot: no examples. "Reply with only the category" constrains the format.
    prompt = PromptTemplate.from_template(
        """Classify the text into one of these categories: {categories}.
Reply with only the category name.

Text: {text}
Category:"""
    )
    classification_chain = prompt | llm | StrOutputParser()
    print(f"zero-shot: {classification_chain.invoke({'text': text, 'categories': categories})}\n")

    # Few-shot with a template: in-context learning (script 01) + reusable
    # templates (this script). FewShotPromptTemplate formats each example with
    # `example_prompt` and places them between `prefix` and `suffix`.
    examples = [
        {"text": "The new GPU doubles ray-tracing performance.", "category": "Technology"},
        {"text": "This ramen shop has the richest broth in town.", "category": "Food and Dining"},
        {"text": "Her latest novel explores memory and loss.", "category": "Literature"},
    ]
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=PromptTemplate.from_template("Text: {text}\nCategory: {category}"),
        prefix="Classify each text into one of these categories: {categories}.\nReply with only the category name.",
        suffix="Text: {text}\nCategory:",
        input_variables=["text", "categories"],
    )
    print("--- Few-shot prompt as rendered ---")
    print(few_shot_prompt.format(text=text, categories=categories))
    few_shot_chain = few_shot_prompt | llm | StrOutputParser()
    print(f"\nfew-shot: {few_shot_chain.invoke({'text': text, 'categories': categories})}\n")


# ---------------------------------------------------------------------------
# 6. Code generation (SQL)
# ---------------------------------------------------------------------------
def code_generation_demo() -> None:
    print_header("6. CODE GENERATION (SQL)")
    description = """
Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days.
The table 'purchases' contains a column 'purchase_date'.
"""
    prompt = PromptTemplate.from_template(
        """Generate a {dialect} SQL query for the following request.
Return only the SQL, without explanation.

Request: {description}"""
    )
    sql_generation_chain = prompt | llm | StrOutputParser()
    # An extra {dialect} variable makes the same template reusable across databases.
    for dialect in ["PostgreSQL", "SQLite"]:
        print(f"--- {dialect} ---")
        print(sql_generation_chain.invoke({"description": description, "dialect": dialect}))
        print()


def main() -> None:
    prompt_template_basics()
    joke_chain_demo()
    summarization_demo()
    question_answering_demo()
    classification_demo()
    code_generation_demo()


if __name__ == "__main__":
    main()
