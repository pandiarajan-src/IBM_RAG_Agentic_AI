"""
01_prompting_basics.py - Basic, zero-shot, one-shot and few-shot prompting
==========================================================================

LEARNING NOTES
--------------
**Prompt engineering** means designing the text you send to an LLM so it
produces the output you want: the right task, format, style and level of detail.

**In-context learning (ICL)** is the model "learning" a task from what is in
the prompt itself. No weights are updated. The examples live only in this one
request; send the next request without them and they're gone.

The techniques below add more and more context:

| Technique   | Examples in the prompt | Use it when...                                    |
|-------------|------------------------|---------------------------------------------------|
| Basic       | none, no instruction   | Exploring how the model continues raw text.       |
| Zero-shot   | none, clear instruction| The task is common and easy to describe.          |
| One-shot    | 1                      | You need a specific format or style.              |
| Few-shot    | 2-5                    | The pattern is subtle, or labels must be exact.   |

Rules of thumb:
- Be explicit about the task AND the output format ("Answer with one word").
- Examples teach FORMAT more strongly than instructions do. Make them
  consistent (same labels, same layout) and varied (different cases).
- Labels like "Classification:" or "Spanish:" at the end of a prompt tell
  the model exactly where its answer should go.

Run (from the repo root):
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/01_prompting_basics.py
    uv run python InContextLearningAndPromptTemplates4AdvancedAI/01_prompting_basics.py --skip-exercises
"""

import argparse

from llm_setup import MODEL, get_client, llm_model, print_header, show, thinking_config


# ---------------------------------------------------------------------------
# 1. Basic prompt
# ---------------------------------------------------------------------------
# A bare phrase with no instruction. A base (non-chat) model would just
# continue the text. Claude is a chat model trained to be helpful, so it often
# reads "The wind is" as a request and writes about wind. That difference in
# behaviour is a lesson in itself: chat models respond better to instructions.
def basic_prompt() -> None:
    print_header("1. BASIC PROMPT")
    prompt = "The wind is "
    # max_tokens=128 mirrors the notebook's max_new_tokens=128. If the reply
    # looks cut off, that's the token limit at work - try raising it.
    show(prompt, llm_model(prompt, max_tokens=128))


def basic_prompt_excercise() -> None:
    """Exercise 1: compare responses to different open-ended basic prompts."""
    print_header("EXERCISE 1 - Basic prompts")
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The benefits of sustainable energy include",
    ]
    for prompt in prompts:
        show(prompt, llm_model(prompt, max_tokens=128))

    # The watsonx version varied temperature/top_p/top_k here. Claude doesn't
    # expose those. The levers you DO have are max_tokens, a system prompt and
    # the wording of the prompt. Watch the same phrase change with a system prompt:
    prompt = "Once upon a time in a distant galaxy"
    print("--- Same prompt, with a system prompt asking for a one-line continuation ---")
    show(prompt, llm_model(
        prompt,
        max_tokens=128,
        system="Continue the user's text as a story. Write exactly one sentence. No preamble.",
    ))


# ---------------------------------------------------------------------------
# 2. Zero-shot prompt
# ---------------------------------------------------------------------------
# No examples, only a clear instruction. This relies on what the model learned
# in pre-training and tests how well it generalizes to a task it was never
# shown in this prompt.
def zero_shot_prompt() -> None:
    print_header("2. ZERO-SHOT PROMPT")
    prompt = """Classify the following statement as true or false:
'The Eiffel Tower is located in Berlin.'

Answer:"""
    show(prompt, llm_model(prompt))


def zero_shot_excercise() -> None:
    """Exercise 2: zero-shot classification, summarization and translation."""
    print_header("EXERCISE 2 - Zero-shot prompts")

    movie_review_prompt = """Classify the following movie review as either 'positive' or 'negative'.

Review: "I was extremely disappointed by this film. The plot was predictable, the acting was wooden, and the special effects looked cheap. I can't recommend this to anyone."

Classification:"""

    climate_change_prompt = """Summarize the following paragraph about climate change in no more than two sentences.

Paragraph: "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases. The consequences of climate change include more frequent and severe droughts, storms, and heat waves, rising sea levels, melting glaciers, and warming oceans which can directly impact biodiversity, agriculture, and human health."

Summary:"""

    translation_prompt = """Translate the following English phrase into Spanish.

English: "I would like to order a coffee with milk and two sugars, please."

Spanish:"""

    responses = {
        "movie_review": llm_model(movie_review_prompt),
        "climate_change": llm_model(climate_change_prompt),
        "translation": llm_model(translation_prompt),
    }
    for prompt_type, response in responses.items():
        print(f"=== {prompt_type.upper()} RESPONSE ===")
        print(response)
        print()


# ---------------------------------------------------------------------------
# 3. One-shot prompt
# ---------------------------------------------------------------------------
# One worked example shows the model the pattern to copy: the format, the
# quoting style and how long the answer should be.
def one_shot_prompt() -> None:
    print_header("3. ONE-SHOT PROMPT")
    prompt = """Here is an example of translating a sentence from English to French:

English: "How is the weather today?"
French: "Comment est le temps aujourd'hui?"

Now, translate the following sentence from English to French:

English: "Where is the nearest supermarket?"
French:"""
    # The notebook used max_new_tokens=20 to force a short answer. The example
    # already sets the format; the system prompt keeps Claude from adding notes.
    # (A "\n" stop sequence won't work: the API rejects whitespace-only stops.)
    show(prompt, llm_model(
        prompt,
        max_tokens=60,
        system="Reply with only the translated sentence, in quotes, like the example.",
    ))


def one_shot_excercise() -> None:
    """Exercise 3: one-shot formal email, concept simplification, keyword extraction."""
    print_header("EXERCISE 3 - One-shot prompts")

    formal_email_prompt = """Here is an example of a formal email requesting information:

Subject: Inquiry Regarding Product Specifications for Model XYZ-100

Dear Customer Support Team,

I hope this email finds you well. I am writing to request detailed specifications for your product Model XYZ-100. Specifically, I am interested in learning about its dimensions, power requirements, and compatibility with third-party accessories.

Could you please provide this information at your earliest convenience? Additionally, I would appreciate any available documentation or user manuals that you could share.

Thank you for your assistance in this matter.

Sincerely,
John Smith

---

Now, please write a formal email to a university admissions office requesting information about their application deadline and required documents for the Master's program in Computer Science:"""

    technical_concept_prompt = """Here is an example of explaining a technical concept in simple terms:

Technical Concept: Blockchain
Simple Explanation: A blockchain is like a digital notebook that many people have copies of. When someone writes a new entry in this notebook, everyone's copy gets updated. Once something is written, it can't be erased or changed, and everyone can see who wrote what. This makes it useful for recording important information that needs to be secure and trusted by everyone.

---

Now, please explain the following technical concept in simple terms:

Technical Concept: Machine Learning
Simple Explanation:"""

    keyword_extraction_prompt = """Here is an example of extracting keywords from a sentence:

Sentence: "Cloud computing offers businesses flexibility, scalability, and cost-efficiency for their IT infrastructure needs."
Keywords: cloud computing, flexibility, scalability, cost-efficiency, IT infrastructure

---

Now, please extract the main keywords from the following sentence:

Sentence: "Sustainable agriculture practices focus on biodiversity, soil health, water conservation, and reducing chemical inputs."
Keywords:"""

    responses = {
        "formal_email": llm_model(formal_email_prompt),
        "technical_concept": llm_model(technical_concept_prompt),
        "keyword_extraction": llm_model(keyword_extraction_prompt),
    }
    for prompt_type, response in responses.items():
        print(f"=== {prompt_type.upper()} RESPONSE ===")
        print(response)
        print()


# ---------------------------------------------------------------------------
# 4. Few-shot prompt
# ---------------------------------------------------------------------------
# Several labelled examples make the pattern unambiguous: the label set
# (Joy / Frustration / Sadness ...), capitalization and a one-word answer.
def few_shot_prompt() -> None:
    print_header("4. FEW-SHOT PROMPT")
    prompt = """Here are few examples of classifying emotions in statements:

Statement: 'I just won my first marathon!'
Emotion: Joy

Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration

Statement: 'My best friend is moving to another country.'
Emotion: Sadness

Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.'
Emotion:"""
    # A tiny budget (the notebook used 10 tokens) keeps the answer brief.
    # Thinking is off in llm_model(), so all 10 tokens go to the answer.
    show(prompt, llm_model(prompt, max_tokens=10))


# Bonus: the chat-native way to do few-shot.
# Chat models are trained on alternating user/assistant turns. Putting the
# examples in as earlier turns of a fake conversation is often more reliable
# than packing them into one big string: the model sees its own "past answers"
# and copies their exact format.
def few_shot_as_messages() -> None:
    print_header("4b. FEW-SHOT AS CHAT MESSAGES (Claude-native pattern)")
    examples = [
        ("I just won my first marathon!", "Joy"),
        ("I can't believe I lost my keys again.", "Frustration"),
        ("My best friend is moving to another country.", "Sadness"),
    ]
    messages = []
    for statement, emotion in examples:
        messages.append({"role": "user", "content": statement})
        messages.append({"role": "assistant", "content": emotion})
    messages.append({"role": "user", "content": "That movie was so scary I had to cover my eyes."})

    response = get_client().messages.create(
        model=MODEL,
        max_tokens=10,
        system="Classify the emotion in each statement. Reply with a single word.",
        messages=messages,
        thinking=thinking_config(False),
        output_config={"effort": "low"},
    )
    answer = "".join(b.text for b in response.content if b.type == "text")
    print(f"messages sent: {len(messages)} (3 example pairs + 1 new statement)")
    print(f"response : {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--skip-exercises", action="store_true", help="run only the core demos")
    args = parser.parse_args()
    run_exercises = not args.skip_exercises

    basic_prompt()
    if run_exercises:
        basic_prompt_excercise()

    zero_shot_prompt()
    if run_exercises:
        zero_shot_excercise()

    one_shot_prompt()
    if run_exercises:
        one_shot_excercise()

    few_shot_prompt()
    few_shot_as_messages()


if __name__ == "__main__":
    main()
