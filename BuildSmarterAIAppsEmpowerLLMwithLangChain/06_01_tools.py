"""
06_01_tools.py - Tools, toolkits and native tool calling
========================================================

LEARNING NOTES
--------------
Why do we need tools?
---------------------
On its own, an LLM can only do one thing: take text in and produce text out.
It cannot browse the web, run code, check today's weather, or reliably multiply
two big numbers (it "guesses" the digits instead of calculating them).

A TOOL fixes this. A tool is an ordinary function that the model can ASK your
program to run. The model never runs the function itself; it only says
"please call X with these arguments", your code runs it, and the result is
handed back to the model.

Examples of tools: a calculator, a web-search API, a database query, a vector
store lookup (script 04)...

Every LangChain tool has three parts:
- name        : how the model refers to it. Claude requires letters, digits,
                "_" or "-" (no spaces: the notebook's "Python Calculator" would
                be rejected).
- description : tells the model WHEN to use the tool. The model reads this text
                to decide, so write it for the model, not for humans!
- args schema : the JSON schema (names + types) of the inputs. LangChain builds
                it automatically from the function's type hints
                (e.g. `location: str`).

Two ways to define a tool
-------------------------
1. `@tool` decorator
   A "decorator" is a function that wraps another function to add behaviour.
   Putting `@tool` above a function turns it into a tool:
   name = the function name, description = the docstring (the text in triple
   quotes right under `def`).
2. `StructuredTool.from_function(func, name=..., description=...)`
   Wraps a function that ALREADY exists, so you can choose the name and
   description yourself. (The course notebook used the older `Tool(...)` class
   for the same job.)

A TOOLKIT is simply a Python list of tools meant to be used together.

Three kinds of tools used in this script
-----------------------------------------
1. LOCAL tool      : our own Python function (python_calculator). It runs on
                      YOUR machine, in YOUR code.
2. DEFAULT tools   : ready-made tools that ship with LangChain
                      (read_file / write_file from FileManagementToolkit).
                      No function to write; we only choose which ones to enable.
3. CLAUDE (server) tool : web_search. This one is a plain dict, not a function.
                      Anthropic runs the search on ITS servers and Claude uses
                      the results in the same reply, so our code never has to
                      execute anything or send back a ToolMessage for it.

Tool calling (native, no text parsing)
--------------------------------------
`llm.bind_tools(tools)` returns a copy of the model that sends the tool schemas
along with every prompt. Now, instead of always answering in prose, Claude may
reply with an AIMessage whose `.tool_calls` attribute holds a structured request:

    [{"name": "python_calculator", "args": {"code": "print(2 * 21)"}, "id": "..."}]

Your code runs the tool, then sends the result back to the model as a
ToolMessage so it can write the final answer. Doing that loop by hand is what
this script shows for ONE step; script 06_02_agents.py automates the whole loop.

SAFETY: run_python executes model-written code with exec(). That is fine for a
local learning script. NEVER do this in a real app without a sandbox: the
model's input can come from untrusted users. The file tools are confined to one
folder (`workspace/`) via root_dir, so the model cannot touch the rest of disk.

Run:
    uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/06_01_tools.py
"""

import contextlib  # tools for managing "with" blocks; used here to capture print() output
import io  # gives us StringIO: an in-memory text buffer that behaves like a file
from pathlib import Path

from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import StructuredTool

# Shared helpers from llm_setup.py: get_chat_model builds the Claude chat model,
# print_header prints a nicely formatted section title.
from llm_setup import get_chat_model, print_header

# max_tokens caps how long the model's reply can be.
llm = get_chat_model(max_tokens=2048)


# ---------------------------------------------------------------------------
# 1. Defining tools
# ---------------------------------------------------------------------------
def run_python(code: str) -> str:
    """Execute Python code and return whatever it prints."""
    # The model can only read TEXT back, so we need the code's printed output
    # as a string. redirect_stdout temporarily sends everything print() writes
    # into this in-memory buffer instead of the terminal.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            # exec() runs a string as Python code. The empty dict {} is the
            # set of global variables the code starts with, so each call begins
            # "fresh" and cannot see this script's variables. See SAFETY note above.
            exec(code, {})
    except Exception as e:
        # Return the error as TEXT instead of crashing. The model can read it
        # and try again with corrected code. Letting the model recover from
        # its own mistakes is a key trick when building tools.
        return f"Error: {e!r}"
    # If the code printed nothing, give a hint; otherwise the model would
    # receive an empty string and not know what went wrong.
    return buffer.getvalue() or "(no output - use print() to see results)"


# Way 1 - wrap an EXISTING function with StructuredTool.from_function.
# We choose the tool's name and description ourselves. The description doubles
# as instructions: it tells the model the code must PRINT its result, because
# run_python only returns what is printed.
python_calculator = StructuredTool.from_function(
    func=run_python,
    name="python_calculator",
    description="Useful for when you need to perform calculations or execute Python code. "
    "Input should be valid Python code that PRINTS the result.",
)


# Kind 2 - DEFAULT tools. FileManagementToolkit builds ready-made file tools.
# root_dir confines them to one folder, and selected_tools keeps only read/write
# (the toolkit also offers copy, delete, list_directory, ...).
WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)
file_tools = FileManagementToolkit(root_dir=str(WORKSPACE), selected_tools=["read_file", "write_file"]).get_tools()

# Kind 3 - CLAUDE web tool. A server-side tool is described by a dict, not a
# function: Anthropic runs it. max_uses caps how many searches one reply may make.
web_search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}

# A toolkit is simply a list of tools that work together. Only the LOCAL and
# DEFAULT tools are runnable by us; the dict is just passed through to Claude.
local_tools = [python_calculator, *file_tools]
tools = [*local_tools, web_search]


# ---------------------------------------------------------------------------
# 2. Using tools directly (no LLM involved)
# ---------------------------------------------------------------------------
def tools_demo() -> None:
    print_header("1. TOOLS")
    # Tools are "Runnables" (LangChain's common interface, see script 05), so
    # they have .invoke() just like models and chains. Here we call them
    # ourselves, with no LLM, to prove that a tool is just a function.
    print("python_calculator.invoke ->", python_calculator.invoke("a = 3; b = 1; print(a+b)").strip())
    # Default tools work the same way: write a file, then read it back.
    file_tools[1].invoke({"file_path": "hello.txt", "text": "Hello from a tool!"})
    print("read_file.invoke         ->", file_tools[0].invoke({"file_path": "hello.txt"}))

    # This is exactly what the model gets to see about each tool. The model
    # chooses a tool using only these three fields: name, description, args.
    for t in local_tools:
        print(f"\nname={t.name!r}\ndescription={t.description!r}\nargs={t.args}")


# ---------------------------------------------------------------------------
# 3. Tool calling: what an agent does under the hood (one step)
# ---------------------------------------------------------------------------
def tool_calling_demo() -> None:
    print_header("2. TOOL CALLING (one step, by hand)")

    # bind_tools attaches the tool schemas to every request this model makes.
    # It does NOT run anything; it only tells the model which tools exist.
    llm_with_tools = llm.bind_tools(tools)

    tool_by_name = {t.name: t for t in local_tools}
    prompts = [
        "What is 123456 * 654321?",  # -> local tool
        "Save a haiku about tools into haiku.txt",  # -> default tool (write_file)
        "Search the web: what is the latest stable Python version?",  # -> Claude web tool
    ]
    for prompt in prompts:
        print(f"\nQ: {prompt}")
        # Step 1: the model sees the tool list and, instead of guessing,
        # replies with a tool request (or, for web_search, just the answer).
        ai_msg = llm_with_tools.invoke(prompt)

        # Step 2: WE run the local/default tools. The model only produced a
        # request (name + args); we use the name to look up the right tool.
        # web_search never appears in .tool_calls: Anthropic already ran it.
        for call in ai_msg.tool_calls:
            result = tool_by_name[call["name"]].invoke(call["args"])
            print(f"  running {call['name']}({call['args']}) -> {result}")
        if not ai_msg.tool_calls:
            print(f"  (answered directly, web search done server-side) {ai_msg.text[:300]}")

    # Step 3 (not done here): send the tool's result back to the model as a
    # ToolMessage so it can write the final answer for the user. An agent
    # repeats steps 1-3 automatically until the model stops asking for tools.
    # That loop is the subject of 06_02_agents.py.


def main() -> None:
    tools_demo()
    tool_calling_demo()


# This guard means main() only runs when you execute this file directly, not
# when another file imports it.
if __name__ == "__main__":
    main()
