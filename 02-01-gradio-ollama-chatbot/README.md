# Lab 01 - Build a Customer Support Chatbot UI with Gradio and Local Ollama

**Time:** 30 minutes | **Level:** beginner | **Verified:** 2026-11-14 with Gradio 6.32.0, ollama-python 0.6.3, Python 3.12/3.13

---

## 1. Problem statement

Acme Broadband wants a support chatbot. The model already runs locally in Ollama, but customers cannot use a terminal. We need a web page where they type a question and read the answer - without writing HTML, CSS or JavaScript.

## 2. What you will learn

By the end you can:

1. Build an interactive front end for an LLM with Gradio.
2. Create a working multi-turn chatbot that streams answers from a local Ollama model.
3. Use the everyday Gradio components: `Textbox`, `Button`, `Slider`, `Dropdown`, `Markdown`, `Chatbot`, `Tab`, `Dataframe`.
4. Customise, test, containerise and reason about deploying the app.

## 3. What we will build

```
Browser  <--HTTP/SSE-->  Gradio app (Python)  <--HTTP-->  Ollama (llama3.2)
                              |
                              +--> SQLite (conversation log)
```

Three small apps, each one step up:

| Step | File | Gradio API | You get |
|---|---|---|---|
| 1 | `step1_interface.py` | `gr.Interface` | One question, one answer |
| 2 | `step2_blocks.py` | `gr.Blocks` | Custom layout, events, streaming |
| 3 | `step3_chat.py` | `gr.ChatInterface` in `gr.Blocks` | Multi-turn chat, tuning controls, SQLite log tab |

Folder map:

```
01-gradio-ollama-chatbot/
  README.md              this guide
  PRODUCTION.md          taking it to production
  step1_interface.py     step 1 app
  step2_blocks.py        step 2 app
  step3_chat.py          step 3 app (final)
  supportbot/            shared code: config.py, llm.py, db.py
  tests/                 pytest suite + fake Ollama server
  Dockerfile, docker-compose.yml, .env.example
  requirements.txt, requirements-dev.txt
```

## 4. Tech stack and why

| Choice | Why | Alternatives considered |
|---|---|---|
| **Gradio 6** (Apache-2.0) | Chat UI in ~10 lines of Python; streaming and queueing built in | Streamlit (reruns the whole script on each click; chat streaming is clunkier), React (best for production, far too much for 30 min) |
| **Ollama** (MIT) + `ollama` Python client | One command to run a model locally; no API key, no cost, data stays on the laptop | llama.cpp (more knobs, more setup), cloud APIs (cost, data leaves the machine) |
| **llama3.2 3B** | ~2 GB, runs on a CPU-only laptop | gemma3, qwen2.5, phi4-mini - swap with `OLLAMA_MODEL` |
| **SQLite** | Zero-install, one file, ships with Python | Postgres (needed at scale - see PRODUCTION.md) |
| **pytest** + fake Ollama | Tests run in seconds with no model or GPU | Mocking the client (hides HTTP/streaming bugs) |
| **Docker Compose** | Same environment on every laptop | Bare venv (fine for the lab, drifts over time) |

> **No FastAPI or React here?** Our team default is React + FastAPI. Gradio replaces both for internal tools and prototypes (it runs on FastAPI under the hood). PRODUCTION.md shows when to switch back.

> **Licence flag:** Gradio and Ollama are commercial-friendly. The *model* llama3.2 uses Meta's Llama Community License, which is not MIT/Apache. Fine for learning; check with legal before shipping, or use an Apache/MIT model (e.g. `qwen2.5`, `mistral`, `phi4-mini`).

## 5. Concepts, with analogies

| Concept | Analogy | In one line |
|---|---|---|
| **Gradio** | A restaurant's front-of-house kit | Gives you the dining room so you only cook (write Python functions) |
| **Ollama** | The chef in the kitchen | Runs the model and answers over HTTP on port 11434 |
| **`gr.Interface`** | Vending machine | Fixed shape: inputs -> function -> outputs. Fast, not flexible |
| **`gr.Blocks`** | Lego set | You place every brick (component) and wire every connection (event) |
| **`gr.ChatInterface`** | Prefab chat room | Chat window, history, retry/undo, all pre-built; you supply one function |
| **Component** | A form field | `Textbox`, `Slider`, `Button`... each is both a widget and a typed value |
| **Event** (`.click`, `.submit`) | A doorbell wire | "When this happens, call that function with these inputs, put the result there" |
| **Streaming** (`yield`) | Live subtitles vs. waiting for the transcript | Handler yields the answer-so-far; the UI updates each time |
| **History** | The waiter's notepad | The model has no memory; we re-send the whole conversation every turn |
| **System prompt** | The employee handbook | Rules the model reads before every conversation |
| **Temperature** | A creativity dial | 0 = same answer every time, 1 = more varied (and riskier) |
| **Queue / concurrency limit** | A deli ticket machine | Requests wait their turn so one laptop GPU is not swamped |
| **`render=False`** | Buying furniture but letting the decorator place it | Create a component now, let `ChatInterface` decide where it goes |

## 6. Setup (5 min)

Prerequisites: Python 3.10+, [Ollama](https://ollama.com/download) installed and running.

```bash
cd 01-gradio-ollama-chatbot

ollama pull llama3.2                 # one-time, ~2 GB
ollama run llama3.2 "say hi"         # sanity check

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**No model yet / slow download?** Use the simulator so you are not blocked:

```bash
python tests/fake_ollama.py &                   # fake server on :11435
export OLLAMA_HOST=http://127.0.0.1:11435       # it always answers "Please restart your router."
```

## 7. Exercises

### Exercise 1 - The vending machine (5 min)

```bash
python step1_interface.py     # open http://localhost:7860
```

1. Ask a question. Click an example.
2. Open `step1_interface.py`. Note there is **no HTML** - `inputs`, `outputs` and `fn` are the whole app.
3. Quit Ollama and ask again. The red toast is `gr.Error` turning a crash into a message the user can act on. Restart Ollama.

**Try it:** change the output to `gr.Markdown()` and ask for "a bulleted list of router fixes".

### Exercise 2 - The Lego set (8 min)

```bash
gradio step2_blocks.py        # `gradio` instead of `python` = auto-reload on save
```

1. Ask a question and watch it **stream**. Find the `yield` that makes this happen.
2. Trace the wiring: `gr.on(triggers=[ask_btn.click, question.submit], ...)`. Two doorbells, one handler.
3. Move the slider to 1.0, ask the same question twice. Then 0.0, twice. What changed?

**Try it:** add a `gr.Dropdown(["Friendly", "Formal", "Pirate"], label="Tone")`, pass it as a third input, and append `f"Tone: {tone}"` to the system prompt.

### Exercise 3 - The chat room (8 min)

```bash
python step3_chat.py
```

1. Say "My name is Sam." then ask "What is my name?". It remembers - because `chat()` receives `history` and `build_messages()` re-sends it.
2. Open **Tuning controls**. Change the system prompt to "Reply only in haiku." and ask again.
3. Open the **Conversation log** tab and click Refresh. Your turns are in SQLite (`data/chat_log.db`).
4. Read `chat()` in `step3_chat.py`. Match each parameter to where it comes from on screen.

**Try it:** in `supportbot/llm.py`, make `build_messages` keep only the last 6 history turns. Why would you want this? (Hint: context window and cost.)

### Exercise 4 - Tests (2 min)

```bash
pytest -v
```

15 tests, about 10 seconds, no model needed. Open `tests/fake_ollama.py` - a 50-line flight simulator for the Ollama API. `test_end_to_end_over_http` launches the real app and calls it like a browser would.

### Exercise 5 - Docker (2 min)

```bash
docker compose up --build     # open http://localhost:7860
```

The app runs in a container; Ollama stays on your host. `host.docker.internal` is the container's name for "the machine I am running on".

> Linux only: Ollama listens on 127.0.0.1 by default, which containers cannot reach. Start it with `OLLAMA_HOST=0.0.0.0 ollama serve`.

### Stretch goals

- Add thumbs up/down: `gr.Chatbot(feedback_options=["Like", "Dislike"])` and log it with `chatbot.like(...)`.
- Brand it: `demo.launch(theme=gr.themes.Soft(primary_hue="blue"))`.
- `demo.launch(share=True)` gives a public URL valid for 1 week. Discuss: why is this a bad idea with customer data?

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Cannot reach Ollama at ..." | Ollama not running | Start the Ollama app or `ollama serve` |
| "Model 'llama3.2' is not installed" | Model not pulled | `ollama pull llama3.2` |
| First answer takes 20+ s | Model loading into memory | Normal; later answers are fast |
| Port 7860 in use | Another step still running | Stop it, or `GRADIO_SERVER_PORT=7861 python ...` |
| Works locally, fails in Docker on Linux | Ollama bound to 127.0.0.1 | `OLLAMA_HOST=0.0.0.0 ollama serve` |
| Old tutorial code fails | Gradio 6 renamed things | `allow_flagging` -> `flagging_mode`; `theme=`/`css=` moved from `gr.Blocks()` to `launch()`; chat history is always a list of `{"role", "content"}` dicts |

## 9. Conclusion

You built the same support bot three ways and saw the trade-off: `Interface` is fastest, `Blocks` is most flexible, `ChatInterface` is the sweet spot for chat. The key ideas carry to any LLM UI:

- The UI is a thin layer; keep model calls in their own module (`supportbot/llm.py`) so either side can be swapped.
- The model is stateless - you own the history.
- Stream, or users think it is broken.
- Turn failures into messages a human can act on.

Next: read `PRODUCTION.md` to see what breaks when 1 user becomes 1,000.
