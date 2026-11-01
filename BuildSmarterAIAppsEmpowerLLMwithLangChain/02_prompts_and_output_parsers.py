"""
02_prompts_and_output_parsers.py - Prompt templates and output parsers (+ Exercise 2)
=====================================================================================

LEARNING NOTES
--------------
Prompt templates turn user input + parameters into the exact instructions a
model receives. Output parsers do the opposite: turn the model's raw text into
something your program can use (a string, a list, a dict ...).

Prompt templates (langchain_core.prompts)
-----------------------------------------
| Template                          | Produces                     | Use for                       |
|-----------------------------------|------------------------------|-------------------------------|
| PromptTemplate                    | one string                   | simple single-string prompts  |
| ChatPromptTemplate.from_messages  | a list of messages           | system + user role prompts    |
| MessagesPlaceholder("msgs")       | slot for a LIST of messages  | chat history, multi-turn      |

Calling `.invoke(dict)` on a template returns a *PromptValue*, which can become
a string (`.to_string()`) or messages (`.to_messages()`). That's why the same
template can feed both text models and chat models.

LCEL (LangChain Expression Language)
-------------------------------------
Templates, models and parsers are all *Runnables*, so you connect them with `|`:

    chain = prompt | llm | parser
    dict --> PromptTemplate --> PromptValue --> ChatAnthropic --> AIMessage --> parser --> result

Output parsers (langchain_core.output_parsers)
-----------------------------------------------
- StrOutputParser               : AIMessage -> str
- JsonOutputParser               : AIMessage -> dict. With `pydantic_object=`, its
                                   `get_format_instructions()` describes the schema
                                   so you can paste it into the prompt.
- CommaSeparatedListOutputParser : "a, b, c" -> ["a", "b", "c"]

`partial_variables` fills a template variable ONCE when the template is created
(perfect for format instructions), while `input_variables` are filled on every call.

Modern alternative: `llm.with_structured_output(PydanticModel)`
---------------------------------------------------------------
Parsers *ask* the model to emit JSON and hope it complies. Chat models that
support tool calling / structured outputs (Claude does) can be *constrained*
to a schema instead. `with_structured_output` returns a validated Pydantic
object, so there's no format-instructions text and no parsing step. Prefer it for
new code; parsers remain useful for models without that feature.

Changes from the notebook (LangChain 0.2 -> 1.x)
------------------------------------------------
- `from langchain_core.pydantic_v1 import BaseModel` is gone -> use `pydantic` directly.
- `from langchain.output_parsers import CommaSeparatedListOutputParser`
  -> `from langchain_core.output_parsers import CommaSeparatedListOutputParser`.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/02_prompts_and_output_parsers.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field

from llm_setup import get_chat_model, print_header

llm = get_chat_model(max_tokens=512)


# ---------------------------------------------------------------------------
# 1. String prompt template
# ---------------------------------------------------------------------------
def string_prompt_template_demo() -> None:
    print_header("1. STRING PROMPT TEMPLATE")
    prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
    input_ = {"adjective": "funny", "topic": "cats"}  # keys match the {placeholders}

    prompt_value = prompt.invoke(input_)
    print(f"PromptValue : {prompt_value!r}")
    print(f"as string   : {prompt_value.to_string()}")
    print(f"as messages : {prompt_value.to_messages()}")


# ---------------------------------------------------------------------------
# 2. Chat prompt template
# ---------------------------------------------------------------------------
def chat_prompt_template_demo() -> None:
    print_header("2. CHAT PROMPT TEMPLATE")
    # Each tuple is (role, content). "user" and "human" are interchangeable.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("user", "Tell me a joke about {topic}"),
        ]
    )
    for message in prompt.invoke({"topic": "cats"}).to_messages():
        print(f"{type(message).__name__:>14}: {message.content}")


# ---------------------------------------------------------------------------
# 3. MessagesPlaceholder + first chain
# ---------------------------------------------------------------------------
def messages_placeholder_demo() -> None:
    print_header("3. MESSAGES PLACEHOLDER")
    # MessagesPlaceholder("msgs") is replaced by however many messages you pass
    # under the "msgs" key. Script 05 uses this exact slot for chat history.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("msgs"),
        ]
    )
    input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}
    print("formatted:", prompt.invoke(input_).to_messages())

    # Pipe the template into the model -> a chain. Output is an AIMessage.
    chain = prompt | llm
    response = chain.invoke(input_)
    print(f"\nresponse ({type(response).__name__}): {response.content}")


# ---------------------------------------------------------------------------
# 4. JSON output parser (with a Pydantic schema)
# ---------------------------------------------------------------------------
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


def json_parser_demo() -> None:
    print_header("4. JSON OUTPUT PARSER")
    output_parser = JsonOutputParser(pydantic_object=Joke)
    format_instructions = output_parser.get_format_instructions()
    print("--- format instructions injected into the prompt ---")
    print(format_instructions)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],  # filled on every call
        partial_variables={"format_instructions": format_instructions},  # filled once, now
    )
    chain = prompt | llm | output_parser
    result = chain.invoke({"query": "Tell me a joke."})
    print(f"\nparsed ({type(result).__name__}): {result}")

    # Modern alternative: no instructions, no parser. Claude is constrained to the
    # schema and LangChain hands back a validated Joke object.
    structured_llm = llm.with_structured_output(Joke)
    joke = structured_llm.invoke("Tell me a joke about programmers.")
    print(f"\nwith_structured_output ({type(joke).__name__}): setup={joke.setup!r} punchline={joke.punchline!r}")


# ---------------------------------------------------------------------------
# 5. Comma-separated list parser
# ---------------------------------------------------------------------------
def csv_parser_demo() -> None:
    print_header("5. COMMA-SEPARATED LIST PARSER")
    output_parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate(
        template="Answer the user query. {format_instructions}\nList five {subject}.",
        input_variables=["subject"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    chain = prompt | llm | output_parser
    flavors = chain.invoke({"subject": "ice cream flavors"})
    print(f"parsed ({type(flavors).__name__}): {flavors}")


# ---------------------------------------------------------------------------
# 6. Exercise 2 - JSON output parser for movie info
# ---------------------------------------------------------------------------
def exercise_2_movie_json() -> None:
    print_header("6. EXERCISE 2: JSON OUTPUT PARSER FOR MOVIES")
    json_parser = JsonOutputParser()  # no schema: we spell the shape out ourselves

    format_instructions = """RESPONSE FORMAT: Return ONLY a single JSON object - no markdown, no examples, no extra keys. It must look exactly like:
{
  "title": "movie title",
  "director": "director name",
  "year": 2000,
  "genre": "movie genre"
}

IMPORTANT: Your response must be *only* that JSON. Do NOT include any illustrative or example JSON."""
    # The braces above are safe: a partial variable's VALUE is inserted as-is,
    # not parsed as a template. Braces typed directly INTO a template string
    # would need doubling ({{ }}) because single braces mean "variable".

    prompt_template = PromptTemplate(
        template="""You are a JSON-only assistant.

Task: Generate info about the movie "{movie_name}" in JSON format.

{format_instructions}
""",
        input_variables=["movie_name"],
        partial_variables={"format_instructions": format_instructions},
    )
    movie_chain = prompt_template | llm | json_parser

    for movie_name in ["The Matrix", "Spirited Away", "Inception"]:
        result = movie_chain.invoke({"movie_name": movie_name})
        # The parser returned a real dict, so we can index its fields.
        print(f"\nParsed result for {movie_name!r}:")
        print(f"  Title:    {result['title']}")
        print(f"  Director: {result['director']}")
        print(f"  Year:     {result['year']} ({type(result['year']).__name__})")
        print(f"  Genre:    {result['genre']}")


def main() -> None:
    string_prompt_template_demo()
    chat_prompt_template_demo()
    messages_placeholder_demo()
    json_parser_demo()
    csv_parser_demo()
    exercise_2_movie_json()


if __name__ == "__main__":
    main()
