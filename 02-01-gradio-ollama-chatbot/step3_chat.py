"""Step 3 - the finished support chatbot.

gr.ChatInterface (the prefab chat room) inside gr.Blocks (the Lego baseplate):
  * multi-turn chat with streaming
  * tuning controls: system prompt, model, temperature
  * every turn logged to SQLite, viewable in a second tab
Run:  python step3_chat.py
"""
import gradio as gr

from supportbot import config, db, llm

# One inner list per example. With additional_inputs, Gradio requires a list of lists.
EXAMPLES = [
    ["How do I reset my router?"],
    ["My internet has been slow since yesterday."],
    ["Can I upgrade my plan mid-month?"],
]


def chat(message: str, history: list[dict], system_prompt: str, model: str,
         temperature: float, request: gr.Request):
    """Gradio calls this on every user message.

    message/history come from the chat window, the next three from the
    'additional_inputs' controls, and Gradio injects `request` automatically.
    """
    messages = llm.build_messages(system_prompt, history, message)
    answer = ""
    try:
        for answer in llm.stream_chat(messages, model=model, temperature=temperature):
            yield answer
    except llm.LLMUnavailable as exc:
        raise gr.Error(str(exc))
    if answer:  # log once, after the stream has finished
        session_id = (request.session_hash if request else None) or "unknown"
        db.log_turn(session_id, model, message, answer)


def build_demo() -> gr.Blocks:
    models = llm.list_models()
    wanted = config.default_model()
    selected = next((m for m in models if m == wanted or m.split(":")[0] == wanted), models[0])
    with gr.Blocks(title="Acme Support", fill_height=True) as demo:
        gr.Markdown("# Acme Broadband Support")
        with gr.Tab("Chat"):
            # render=False: "build it now, let ChatInterface decide where to place it".
            # Without it, each control would appear right here, above the chat window.
            controls = [
                gr.Textbox(value=config.DEFAULT_SYSTEM_PROMPT, label="System prompt", lines=3, render=False),
                gr.Dropdown(choices=models, value=selected, label="Model",
                            allow_custom_value=True, render=False),
                gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Temperature", render=False),
            ]
            gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(placeholder="Hi! How can I help you today?", height=420),
                textbox=gr.Textbox(placeholder="Type your question and press Enter", submit_btn=True),
                examples=EXAMPLES,
                additional_inputs=controls,
                additional_inputs_accordion=gr.Accordion("Tuning controls", open=False, render=False),
            )
        with gr.Tab("Conversation log"):
            gr.Markdown("Last 20 turns saved in SQLite.")
            log_table = gr.Dataframe(
                headers=["time (UTC)", "session", "model", "question", "answer"],
                value=db.recent_turns, wrap=True, interactive=False,
            )
            gr.Button("Refresh").click(fn=db.recent_turns, outputs=log_table)
    return demo


demo = build_demo()

if __name__ == "__main__":
    # queue() lets several users stream at once; the limit protects your laptop's one GPU/CPU.
    demo.queue(default_concurrency_limit=2).launch()
