"""Step 2 - gr.Blocks: the Lego set.

You place each component and wire each event yourself. The answer now
streams in word by word because the handler is a generator (it uses yield).
Run:  python step2_blocks.py
"""
import gradio as gr

from supportbot import config, llm


def answer(question: str, temperature: float):
    if not question.strip():
        raise gr.Error("Please type a question first.")
    messages = llm.build_messages(config.DEFAULT_SYSTEM_PROMPT, [], question)
    try:
        yield from llm.stream_chat(messages, temperature=temperature)
    except llm.LLMUnavailable as exc:
        raise gr.Error(str(exc))


with gr.Blocks(title="Acme Support - Step 2") as demo:
    gr.Markdown("# Acme Support - Step 2\nSame bot, but we control the layout and the events.")
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Your question", placeholder="My internet is slow... (Enter to send)")
            temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Temperature",
                                    info="0 = predictable, 1 = creative")
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.ClearButton(value="Clear")
        with gr.Column(scale=3):
            reply = gr.Markdown(label="Support agent", container=True, min_height=200)

    # Two triggers (button click, Enter key), one handler.
    gr.on(
        triggers=[ask_btn.click, question.submit],
        fn=answer,
        inputs=[question, temperature],
        outputs=reply,
    )
    clear_btn.add([question, reply])

if __name__ == "__main__":
    demo.launch()
