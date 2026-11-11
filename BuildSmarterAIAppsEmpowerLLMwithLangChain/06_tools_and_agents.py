"""
06_tools_and_agents.py - Tools, toolkits and agents (+ Exercise 7)
==================================================================

LEARNING NOTES
--------------
Tools
-----
On its own an LLM can only output text. A TOOL is a function the model can ask
your program to run: a calculator, a search API, a database query, a vector
store (script 04)... Every LangChain tool has:
- name        : how the model refers to it. Claude requires letters, digits,
                "_" or "-" (no spaces: the notebook's "Python Calculator" would
                be rejected).
- description : tells the model WHEN to use it. Write it for the model!
- args schema : the JSON schema of its inputs, inferred from type hints.

Two ways to define one:
- `@tool` decorator    : name = function name, description = docstring.
- `StructuredTool.from_function(func, name=..., description=...)`: wrap an
  existing function (the notebook used the older `Tool(...)` class for this).

A TOOLKIT is just a list of tools meant to be used together.

How tool calling works (native, no text parsing)
-------------------------------------------------
`llm.bind_tools(tools)` sends the tool schemas along with the prompt. Instead of
answering, Claude can reply with an AIMessage whose `.tool_calls` holds
structured requests: [{"name": "search_weather", "args": {"location": "Miami"}}].
Your code runs the tool and sends the result back as a ToolMessage.

Agents
------
An agent is that loop, automated: the LLM is the reasoning engine and decides
which tool to call, with what input, and when it has enough to answer.

    user question -> LLM -> tool call? --yes--> run tool -> ToolMessage -> LLM -> ...
                              \\--no--> final answer

This is the ReAct (Reason + Act) pattern from the notebook. The difference: the
notebook's `create_react_agent` made the model write "Thought: / Action: /
Action Input:" TEXT and parsed it with regexes, which breaks when the model
formats it slightly wrong (hence `handle_parsing_errors=True`). Modern agents use
native tool calling: structured and validated, with no custom ReAct prompt.

| Notebook (LangChain 0.2)                       | LangChain 1.x                                      |
|------------------------------------------------|----------------------------------------------------|
| create_react_agent(llm, tools, prompt)         | create_agent(model, tools, system_prompt=...)      |
|   + AgentExecutor(agent, tools, verbose=True)  |   (returns a runnable LangGraph graph: no executor)|
| agent_executor.invoke({"input": q})["output"]  | agent.invoke({"messages": [...]})["messages"][-1]  |
| verbose=True                                   | inspect result["messages"] (see print_trace)       |
| langchain_experimental PythonREPL              | small run_python tool below (same idea)            |
| Tool(name=..., func=..., description=...)      | @tool or StructuredTool.from_function(...)         |

To give an agent memory, add `checkpointer=InMemorySaver()` (from
langgraph.checkpoint.memory) and pass `config={"configurable": {"thread_id": "1"}}`.
Same idea as the session ids in script 05.

SAFETY: run_python and the calculator execute model-written code with exec/eval.
That is fine for a local learning script. NEVER do this in a real app
without a sandbox: the model's input can come from untrusted users.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/06_tools_and_agents.py
"""

import contextlib
import io
import math

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool

from llm_setup import get_chat_model, print_header

llm = get_chat_model(max_tokens=2048)


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------
def run_python(code: str) -> str:
    """Execute Python code and return whatever it prints."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})  # fresh globals each time; see SAFETY note above
    except Exception as e:
        return f"Error: {e!r}"
    return buffer.getvalue() or "(no output - use print() to see results)"


# Class-based definition: wrap an existing function and give it a name + description.
python_calculator = StructuredTool.from_function(
    func=run_python,
    name="python_calculator",
    description="Useful for when you need to perform calculations or execute Python code. "
    "Input should be valid Python code that PRINTS the result.",
)


# Decorator definition: the function name becomes the tool name and the
# docstring becomes the description.
@tool
def search_weather(location: str) -> str:
    """Search for the current weather in the specified location."""
    # In a real application, this would call a weather API.
    return f"The weather in {location} is currently sunny and 72°F."


# A toolkit is simply a list of tools that work together.
tools = [python_calculator, search_weather]


def tools_demo() -> None:
    print_header("1. TOOLS")
    # Tools are Runnables too: you can call them directly, no LLM involved.
    print("python_calculator.invoke ->", python_calculator.invoke("a = 3; b = 1; print(a+b)").strip())
    print("search_weather.invoke    ->", search_weather.invoke("Paris"))

    # This is what the model sees about each tool:
    for t in tools:
        print(f"\nname={t.name!r}\ndescription={t.description!r}\nargs={t.args}")


# ---------------------------------------------------------------------------
# 2. Tool calling: what an agent does under the hood (one step)
# ---------------------------------------------------------------------------
def tool_calling_demo() -> None:
    print_header("2. TOOL CALLING (one step, by hand)")
    llm_with_tools = llm.bind_tools(tools)
    ai_msg = llm_with_tools.invoke("What's the weather in Miami?")
    print(f"The model did not answer; it asked for a tool:\n  tool_calls = {ai_msg.tool_calls}")

    # We run the tool ourselves. An agent automates this, then calls the LLM again.
    call = ai_msg.tool_calls[0]
    tool_by_name = {t.name: t for t in tools}
    print(f"  running {call['name']}({call['args']}) -> {tool_by_name[call['name']].invoke(call['args'])}")


# ---------------------------------------------------------------------------
# 3. Agents
# ---------------------------------------------------------------------------
def print_trace(messages) -> None:
    """Show the agent's reasoning loop (the modern stand-in for verbose=True)."""
    for message in messages[1:]:  # skip the user's question
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  -> Action: {call['name']}  Input: {call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"  <- Observation: {str(message.content).strip()[:200]}")


def ask(agent, question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print_trace(result["messages"])
    return result["messages"][-1].text


def agent_demo() -> None:
    print_header("3. AGENT (create_agent)")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the python_calculator tool for any math "
        "instead of computing in your head, and remember its code must print() the result.",
    )

    queries = [
        "What is the square root of 256?",
        "What's 345 * 789?",
        "What's the weather in Miami?",
        "If it's sunny in Chicago, what would be a good outdoor activity?",
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
        # No builtins: only arithmetic and the math module are reachable.
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def format_text(format_type: str, text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    format_type must be one of: 'uppercase', 'lowercase', 'titlecase'."""
    # The notebook packed both values into one string ("uppercase: hello") and
    # parsed it. With tool calling, each value is its own typed argument.
    formatters = {"uppercase": str.upper, "lowercase": str.lower, "titlecase": str.title}
    formatter = formatters.get(format_type.strip().lower())
    if formatter is None:
        return f"Error formatting text: unknown format_type {format_type!r}"
    return f"Formatted text: {formatter(text)}"


def exercise_7_first_agent() -> None:
    print_header("4. EXERCISE 7: CALCULATOR + TEXT FORMATTER AGENT")
    agent = create_agent(
        model=llm,
        tools=[calculator, format_text],
        system_prompt="You are a helpful assistant who uses tools to help with simple tasks. "
        "Always use a tool for arithmetic or text formatting, then give a short final answer.",
    )
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
    tools_demo()
    tool_calling_demo()
    agent_demo()
    exercise_7_first_agent()


if __name__ == "__main__":
    main()
