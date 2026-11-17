"""Step 1 - gr.Interface: the vending machine.

One function, one input, one output. Gradio builds the whole page for you.
Run:  python step1_interface.py   then open http://localhost:7860
"""
import gradio as gr

from supportbot import llm


def answer(question: str) -> str:
    try:
        return llm.ask(question)
    except llm.LLMUnavailable as exc:
        raise gr.Error(str(exc))  # shows a red toast in the browser instead of a stack trace


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Your question", placeholder="My internet is slow...", lines=2),
    outputs=gr.Textbox(label="Support agent", lines=6),
    title="Acme Support - Step 1",
    description="Ask one question, get one answer.",
    examples=["How do I reset my router?", "Why is my bill higher this month?"],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
