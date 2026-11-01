# In-Context Learning & Prompt Templates (Claude + LangChain 1.x)

A script version of the IBM lab notebook `In-Context Learning and Prompt Templates for Advanced AI.ipynb`.
It uses Claude (Anthropic SDK + `langchain-anthropic`) in place of IBM watsonx, and LangChain 1.x in place of LangChain 0.2.

## Learning path

| # | Script | Concepts | Notebook exercises |
|---|--------|----------|--------------------|
| – | `llm_setup.py` | Claude client, `llm_model()` helper, `ChatAnthropic` factory, watsonx → Claude parameter mapping | – |
| 1 | `01_prompting_basics.py` | Basic, zero-shot, one-shot and few-shot prompting; few-shot as chat messages | 1, 2, 3 |
| 2 | `02_reasoning_prompts.py` | Chain-of-thought, self-consistency (single prompt + majority vote), built-in thinking | 4 |
| 3 | `03_langchain_templates.py` | `PromptTemplate`, LCEL `prompt \| llm \| parser`, summarization, Q&A, classification, `FewShotPromptTemplate`, SQL generation | – |
| 4 | `04_roleplay_and_exercise5.py` | `ChatPromptTemplate` role-play bot with memory, `.batch()`, structured output with Pydantic | 5 |

Every script opens with a **LEARNING NOTES** docstring. Read it before running the script.

## Setup (from the repo root)

```bash
uv sync                                  # installs anthropic, langchain, langchain-anthropic, python-dotenv
cp .env.example .env                     # then put your real ANTHROPIC_API_KEY in .env
uv run python InContextLearningAndPromptTemplates4AdvancedAI/01_prompting_basics.py
```

Useful flags:
- `--skip-exercises`: scripts 01 and 02 run only the core demos.
- `--samples N`: script 02 sets the number of self-consistency votes.
- `--demo` / `--only-exercise`: script 04 runs the chatbot with canned questions, or skips it.

## Key differences from the notebook

- **No `temperature` / `top_p` / `top_k`.** Current Claude models reject them. Steer output with the prompt, `max_tokens`, `stop_sequences`, a system prompt and `effort`.
- **Thinking.** Claude Haiku 5.5 can reason before answering. `llm_model()` turns this off by default, so small `max_tokens` budgets behave like the notebook's. Script 02 turns it on for comparison.
- **No `LLMChain` and no `RunnableLambda(format_prompt)`.** A `PromptTemplate` is already a Runnable, so you pipe it straight in.
- **Model.** Set in one place, `MODEL` in `llm_setup.py` (default `claude-haiku-5-5`).
