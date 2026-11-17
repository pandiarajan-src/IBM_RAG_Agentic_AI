"""
06_02_agents.py - Agents (+ Exercise 7)
=======================================

LEARNING NOTES
--------------
Prerequisite: read 06_01_tools.py first. It explains what tools are and how
tool calling works for a single step. This script automates that step into a loop.

What is an agent?
-----------------
An agent is the tool-calling loop, automated. The LLM acts as the "reasoning
engine": it decides which tool to call, with what input, looks at the result,
and decides whether it needs another tool or can answer now.

    user question -> LLM -> tool call? --yes--> run tool -> ToolMessage -> LLM -> ...
                              \\--no--> final answer

Why a loop? Some questions need several steps ("list the primes below 50 AND add
them up"), and the model can't know in advance how many. It keeps going until
it has enough information to answer.

This is the ReAct (Reason + Act) pattern from the notebook. The difference:
the notebook's `create_react_agent` made the model write "Thought: / Action: /
Action Input:" as plain TEXT and parsed it with regular expressions. That breaks
whenever the model formats the text slightly wrong (hence
`handle_parsing_errors=True`). Modern agents use native tool calling: structured,
validated requests with no custom ReAct prompt and no text parsing.

| Notebook (LangChain 0.2)                       | LangChain 1.x                                      |
|------------------------------------------------|----------------------------------------------------|
| create_react_agent(llm, tools, prompt)         | create_agent(model, tools, system_prompt=...)      |
|   + AgentExecutor(agent, tools, verbose=True)  |   (returns a runnable LangGraph graph: no executor)|
| agent_executor.invoke({"input": q})["output"]  | agent.invoke({"messages": [...]})["messages"][-1]  |
| verbose=True                                   | inspect result["messages"] (see print_trace)       |
| langchain_experimental PythonREPL              | small run_python tool below (same idea)            |
| Tool(name=..., func=..., description=...)      | @tool or StructuredTool.from_function(...)         |

Messages: the agent's "memory" of one run
-----------------------------------------
An agent's state is a list of messages. Each kind has a role:
- HumanMessage : the user's question.
- AIMessage    : the model's turn. It either contains the final answer text or
                 a `.tool_calls` list (a request to run tools).
- ToolMessage  : the result of a tool, sent back to the model.
After `agent.invoke(...)`, `result["messages"]` holds the full conversation
including every tool call, which is how we "see inside" the agent.

Remembering across questions
----------------------------
Each `agent.invoke` above starts from scratch. To give an agent memory across
calls, add `checkpointer=InMemorySaver()` (from langgraph.checkpoint.memory)
and pass `config={"configurable": {"thread_id": "1"}}`. Same idea as the
session ids in script 05.

SAFETY: run_python and the calculator execute model-written code with
exec/eval. That is fine for a local learning script. NEVER do this in a real
app without a sandbox: the model's input can come from untrusted users.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/06_02_agents.py
"""

import contextlib  # tools for "with" blocks; used to capture print() output
import io  # StringIO: an in-memory text buffer that behaves like a file
import math

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool

from llm_setup import get_chat_model, print_header

llm = get_chat_model(max_tokens=2048)


# ---------------------------------------------------------------------------
# 1. Tools for the agent (same ones as 06_01_tools.py, repeated here so this
#    file runs on its own)
# ---------------------------------------------------------------------------
def run_python(code: str) -> str:
    """Execute Python code and return whatever it prints."""
    # redirect_stdout sends print() output into this buffer so we can return it
    # as text the model can read.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})  # fresh globals each time; see SAFETY note above
    except Exception as e:
        # Report errors as text so the model can read them and retry.
        return f"Error: {e!r}"
    return buffer.getvalue() or "(no output - use print() to see results)"


# Wrap an existing function as a tool, choosing its name and description.
python_calculator = StructuredTool.from_function(
    func=run_python,
    name="python_calculator",
    description="Useful for when you need to perform calculations or execute Python code. "
    "Input should be valid Python code that PRINTS the result.",
)


# The decorator form: function name = tool name, docstring = description.
@tool
def search_weather(location: str) -> str:
    """Search for the current weather in the specified location."""
    # In a real application, this would call a weather API.
    return f"The weather in {location} is currently sunny and 72°F."


# A toolkit is simply a list of tools that work together.
tools = [python_calculator, search_weather]


# ---------------------------------------------------------------------------
# 2. Helpers to run an agent and look inside it
# ---------------------------------------------------------------------------
def print_trace(messages) -> None:
    """Show the agent's reasoning loop (the modern stand-in for verbose=True)."""
    for message in messages[1:]:  # skip index 0: it is the user's own question
        if isinstance(message, AIMessage) and message.tool_calls:
            # The model asked for one or more tools ("Action" in ReAct terms).
            for call in message.tool_calls:
                print(f"  -> Action: {call['name']}  Input: {call['args']}")
        elif isinstance(message, ToolMessage):
            # What the tool returned ("Observation" in ReAct terms). We cut it
            # to 200 characters so long outputs don't flood the terminal.
            print(f"  <- Observation: {str(message.content).strip()[:200]}")


def ask(agent, question: str) -> str:
    """Send one question to an agent, print its trace, and return the answer."""
    # Agents take a dict with a "messages" list. A {"role": "user", ...} dict is
    # a shortcut for a HumanMessage.
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print_trace(result["messages"])
    # The last message is the model's final reply (no more tool calls);
    # .text extracts its plain-text content.
    return result["messages"][-1].text


# ---------------------------------------------------------------------------
# 3. Agent demo
# ---------------------------------------------------------------------------
def agent_demo() -> None:
    print_header("3. AGENT (create_agent)")

    # create_agent wires everything together: the model, the list of tools, and
    # a system prompt (standing instructions that apply to every question).
    # It returns a runnable graph that performs the whole tool-calling loop.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the python_calculator tool for any math "
        "instead of computing in your head, and remember its code must print() the result.",
    )

    # Watch the trace for each query. Notice the agent chooses the tool by
    # itself: math -> calculator, weather -> weather tool, and for the Chicago
    # question it may need no tool at all, so it just answers directly.
    queries = [
        "What is the square root of 256?",
        "What's 345 * 789?",
        "What's the weather in Miami?",
        "If it's sunny in Chicago, what would be a good outdoor activity?",
        # Multi-step: the model may write one program, or call the tool several times.
        "Generate a list of prime numbers below 50 and calculate their sum",
    ]
    for query in queries:
        print(f"\n{'=' * 60}\nQUERY: {query}\n{'=' * 60}")
        print(f"FINAL ANSWER: {ask(agent, query)}")


# ---------------------------------------------------------------------------
# 4. Exercise 7 - your first agent with a calculator and a text formatter
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        # eval() computes a string like "2 + 2" and returns the value.
        # The second argument {"__builtins__": {}} removes Python's built-in
        # functions (open, __import__, ...), so the expression can't touch files
        # or import modules. The third argument exposes only the math module.
        # This is safer than run_python, though not a full sandbox.
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Result: {result}"
    except Exception as e:
        # Return the error as text so the model can fix its input and retry.
        return f"Error calculating: {e}"


@tool
def format_text(format_type: str, text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    format_type must be one of: 'uppercase', 'lowercase', 'titlecase'."""
    # This tool has TWO arguments. The notebook packed both values into one
    # string ("uppercase: hello") and parsed it by hand. With tool calling,
    # each value is its own typed argument, so no parsing is needed.
    # Dictionary of "format name -> function that does it". str.upper etc. are
    # existing string methods, passed around as values.
    formatters = {"uppercase": str.upper, "lowercase": str.lower, "titlecase": str.title}
    # strip() + lower() make the lookup forgiving ("  Uppercase " still works).
    # .get returns None when the key is missing instead of raising an error.
    formatter = formatters.get(format_type.strip().lower())
    if formatter is None:
        return f"Error formatting text: unknown format_type {format_type!r}"
    return f"Formatted text: {formatter(text)}"


def exercise_7_first_agent() -> None:
    print_header("4. EXERCISE 7: CALCULATOR + TEXT FORMATTER AGENT")

    # A new agent with a different toolkit and a different system prompt.
    # The system prompt steers behaviour: "Always use a tool" stops the model
    # from answering simple questions from memory.
    agent = create_agent(
        model=llm,
        tools=[calculator, format_text],
        system_prompt="You are a helpful assistant who uses tools to help with simple tasks. "
        "Always use a tool for arithmetic or text formatting, then give a short final answer.",
    )
    # Note the last question is phrased oddly ("titlecase: ..."). The model has
    # to work out that this means format_type="titlecase" and the text after it.
    test_questions = [
        "What is 25 + 63?",
        "Can you convert 'hello world' to uppercase?",
        "Calculate 15 * 7",
        "titlecase: langchain is awesome",
    ]
    for question in test_questions:
        print(f"\n===== Testing: {question} =====")
        print(f"Answer: {ask(agent, question)}")


def main() -> None:
    agent_demo()
    exercise_7_first_agent()


# This guard means main() only runs when you execute this file directly, not
# when another file imports it.
if __name__ == "__main__":
    main()
