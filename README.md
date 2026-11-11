# IBM RAG & Agentic AI (Coursera)

Script versions of the IBM Coursera lab notebooks on RAG and agentic AI. The original labs use IBM watsonx.ai. Here they are rewritten to use **Claude** (Anthropic SDK and `langchain-anthropic`) and **LangChain 1.x**. Each folder keeps the original notebook next to the scripts.

Every script starts with a **LEARNING NOTES** docstring that explains the concepts. Read it before running the script.

## Projects

| Folder | Topic | Scripts |
|--------|-------|---------|
| [InContextLearningAndPromptTemplates4AdvancedAI](InContextLearningAndPromptTemplates4AdvancedAI/) | Prompt engineering and LangChain prompt templates | 4 |
| [BuildSmarterAIAppsEmpowerLLMwithLangChain](BuildSmarterAIAppsEmpowerLLMwithLangChain/) | LangChain building blocks, RAG, memory, agents | 6 |

Suggested order: do the prompting folder first, then the LangChain folder. The second builds on the first.

### 1. In-Context Learning & Prompt Templates

How to write prompts that get the output you want, and how to turn them into reusable LangChain templates.

| Script | What it covers |
|--------|----------------|
| `01_prompting_basics.py` | Basic, zero-shot, one-shot and few-shot prompting |
| `02_reasoning_prompts.py` | Chain-of-thought, self-consistency (majority vote), built-in thinking |
| `03_langchain_templates.py` | `PromptTemplate`, LCEL `prompt \| llm \| parser`, summarization, Q&A, classification, few-shot templates, SQL generation |
| `04_roleplay_and_exercise5.py` | Role-play chatbot with memory, `.batch()`, structured output with Pydantic (Exercise 5) |

See that folder's [README](InContextLearningAndPromptTemplates4AdvancedAI/README.md) for flags and the differences from the notebook.

### 2. Build Smarter AI Apps: Empower LLMs with LangChain

A walk through the full LangChain stack, ending with a complete RAG pipeline and tool-using agents.

| Script | What it covers |
|--------|----------------|
| `01_models_and_messages.py` | Raw model vs chat model, messages, the Runnable interface (Exercise 1) |
| `02_prompts_and_output_parsers.py` | Prompt templates, output parsers, LCEL (Exercise 2) |
| `03_documents_loaders_splitters.py` | `Document`, loaders (PDF, web), text splitters (Exercise 3). No API key needed |
| `04_embeddings_vectorstores_rag.py` | Embeddings, Chroma vector store, similarity and MMR retrievers, parent document retrieval, LCEL RAG chain (Exercise 4) |
| `05_memory_and_chains.py` | Conversation memory and multi-step chains (Exercises 5 and 6) |
| `06_tools_and_agents.py` | Tools, toolkits and agents with native tool calling (Exercise 7) |

`llm_setup.py` holds the shared helpers: the Claude chat model and the embeddings client.

Retrieval pipeline built in scripts 03 and 04:

```
source -> load -> Documents -> split -> chunks -> embed -> Chroma -> retriever -> prompt -> Claude -> answer
```

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # install dependencies
cp .env.example .env       # then add your keys (see below)
uv run python BuildSmarterAIAppsEmpowerLLMwithLangChain/03_documents_loaders_splitters.py
```

| Key in `.env` | Needed for |
|---------------|------------|
| `ANTHROPIC_API_KEY` | All scripts that call Claude |
| `OPENROUTER_API_KEY` | Embeddings in `BuildSmarterAIAppsEmpowerLLMwithLangChain` script 04 and later. Claude has no embedding endpoint, so embeddings go through OpenRouter |
| `MODEL` (optional) | Overrides the default Claude model (`claude-haiku-5-5`) |

## Key differences from the IBM notebooks

- Claude replaces watsonx.ai. `temperature`, `top_p` and `top_k` are not available on current Claude models, so output is steered with the prompt, `max_tokens` and system prompts.
- `OpenAIEmbeddings` pointed at OpenRouter replaces `WatsonxEmbeddings`.
- LangChain 1.x LCEL chains (`prompt | llm | parser`) replace the deprecated `LLMChain` and `RetrievalQA`.
