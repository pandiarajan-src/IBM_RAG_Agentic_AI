"""
02_reasoning_prompts.py - Chain-of-thought (CoT) and self-consistency
=====================================================================

LEARNING NOTES
--------------
**Chain-of-thought (CoT) prompting** asks the model to show its intermediate
reasoning steps before the final answer ("Break down each step..."). Writing
out the steps gives the model room to work, which cuts errors on multi-step
tasks: arithmetic, logic puzzles, planning and decisions.

**Self-consistency** goes one step further:
  1. Generate several *independent* reasoning paths for the same problem.
  2. Extract the final answer from each one.
  3. Take the majority vote.
A single reasoning path can go wrong. When several independent paths agree,
that answer is more trustworthy.

The notebook did self-consistency inside ONE prompt ("give three independent
calculations"). That's a useful approximation, but the three "independent"
attempts can all see each other. Section 2b below runs the real version:
N separate API calls, then a vote in Python.

**Prompted CoT vs built-in thinking**
Modern models such as Claude can *think* (reason privately) before answering.
That is CoT built into the model. With `thinking=True` you don't need to write
"think step by step", and the final answer can stay short. Section 3
compares the two. Use prompted CoT when you WANT the reasoning visible in the
answer (teaching, auditing). Use built-in thinking when you just want a better
final answer.

Run (from the repo root):
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/02_reasoning_prompts.py
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/02_reasoning_prompts.py --skip-exercises
"""

import argparse
import re
from collections import Counter

from llm_setup import llm_model, print_header, show


# ---------------------------------------------------------------------------
# 1. Chain-of-thought prompt
# ---------------------------------------------------------------------------
def chain_of_thought() -> None:
    print_header("1. CHAIN-OF-THOUGHT (CoT) PROMPT")
    prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples.
How many apples are there now?'

Break down each step of your calculation."""
    # CoT answers are longer, so give the model room (notebook: 512 tokens).
    show(prompt, llm_model(prompt, max_tokens=512))

    # Compare: the same question WITHOUT the CoT instruction.
    plain = "A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. How many apples are there now? Answer with just the number."
    print("--- Same problem, no CoT instruction ---")
    show(plain, llm_model(plain, max_tokens=20))


def cot_exercise() -> None:
    """Exercise 4: CoT for a decision and for explaining a process."""
    print_header("EXERCISE 4 - Chain-of-thought prompts")

    decision_making_prompt = """Consider this situation: A student is trying to decide whether to study tonight or go to a movie with friends. They have a test in two days.

Think through this decision step-by-step, considering the pros and cons of each option, and what factors might be most important in making this choice."""

    sandwich_making_prompt = """Explain how to make a peanut butter and jelly sandwich.

Break down each step of the process in detail, from gathering ingredients to finishing the sandwich."""

    responses = {
        "decision_making": llm_model(decision_making_prompt, max_tokens=800),
        "sandwich_making": llm_model(sandwich_making_prompt, max_tokens=800),
    }
    for prompt_type, response in responses.items():
        print(f"=== {prompt_type.upper()} RESPONSE ===")
        print(response)
        print()


# ---------------------------------------------------------------------------
# 2. Self-consistency
# ---------------------------------------------------------------------------
AGE_PROBLEM = "When I was 6, my sister was half of my age. Now I am 70, what age is my sister?"


def self_consistency_single_prompt() -> None:
    """The notebook's version: ask for several solutions inside ONE response."""
    print_header("2a. SELF-CONSISTENCY (single prompt, as in the notebook)")
    prompt = f"""{AGE_PROBLEM}

Provide three independent calculations and explanations, then determine the most consistent result."""
    show(prompt, llm_model(prompt, max_tokens=800))


def self_consistency_majority_vote(samples: int = 5) -> None:
    """The real technique: N independent calls, extract each answer, vote.

    Where does the variety come from? Each call is sampled independently, so
    the wording and reasoning differ from run to run. We ask for a fixed
    "Final answer: <number>" line so that a regex can pull out the answer:
    one more example of prompting for a FORMAT.
    """
    print_header(f"2b. SELF-CONSISTENCY (majority vote over {samples} independent calls)")
    prompt = f"""{AGE_PROBLEM}

Reason step by step, then end with a line of the form:
Final answer: <number>"""

    answers = []
    for i in range(1, samples + 1):
        response = llm_model(prompt, max_tokens=400)
        match = re.search(r"Final answer:\s*(-?\d+)", response)
        answer = match.group(1) if match else "unparsed"
        answers.append(answer)
        print(f"  sample {i}: {answer}")

    votes = Counter(answers)
    winner, count = votes.most_common(1)[0]
    print(f"\nvotes: {dict(votes)}")
    print(f"most consistent answer: {winner} ({count}/{samples} samples agree)\n")


# ---------------------------------------------------------------------------
# 3. Prompted CoT vs the model's built-in thinking
# ---------------------------------------------------------------------------
def builtin_thinking_comparison() -> None:
    print_header("3. PROMPTED CoT vs BUILT-IN THINKING")
    question = f"{AGE_PROBLEM} Reply with only the number."

    print("--- thinking OFF, no CoT: the model must answer immediately ---")
    show(question, llm_model(question, max_tokens=20))

    # With thinking on, the model reasons privately first. The reasoning tokens
    # count toward max_tokens, so leave headroom. Higher effort = more reasoning.
    print("--- thinking ON (effort=medium): reasoning happens privately, answer stays short ---")
    show(question, llm_model(question, max_tokens=4000, thinking=True, effort="medium"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--skip-exercises", action="store_true", help="run only the core demos")
    parser.add_argument("--samples", type=int, default=5, help="number of self-consistency samples")
    args = parser.parse_args()

    chain_of_thought()
    if not args.skip_exercises:
        cot_exercise()

    self_consistency_single_prompt()
    self_consistency_majority_vote(args.samples)

    builtin_thinking_comparison()


if __name__ == "__main__":
    main()
