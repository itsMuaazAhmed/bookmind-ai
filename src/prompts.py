"""
Prompt templates used by BookMind AI.

All prompts are kept in a single module so they can be imported from
any part of the code base (`app.py`, the LLM service, etc.) without
causing circular imports.

Only plain‑text strings are exported – the LLM wrapper will inject the
user’s question and any retrieved context at call time.
"""

# ----------------------------------------------------------------------
# System‑level prompt – sets the overall behavior of the assistant.
# ----------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are **BookMind AI**, an AI assistant that answers
questions based strictly on the contents of an uploaded PDF document.
You must:

1. Use only the supplied context (excerpts from the PDF) to form your answer.
2. Cite the source page numbers for every factual statement.
3. If the answer cannot be found in the provided context, respond with:
   "I couldn't find that information in the uploaded document."
4. Keep the tone friendly, concise, and professional.

The context will be supplied as a JSON list of chunks, each containing
the fields `text`, `page_number`, and `document_name`.  Do **not** fabricate
information or hallucinate page numbers.  Only return plain text –
no markdown tables or code blocks unless the user explicitly asks for them."""

# ----------------------------------------------------------------------
# User‑level prompt – combines the system prompt with the retrieved chunks.
# ----------------------------------------------------------------------
def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Assemble the final prompt sent to Gemini.

    Parameters
    ----------
    question: str
        The user’s natural‑language query.
    retrieved_chunks: list[dict]
        Each dict must contain `text` and `page_number`.  The function
        formats them as a numbered list.

    Returns
    -------
    str
        The full prompt (system prompt + context + user question).
    """
    # Turn the chunks into a readable context block.
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = chunk.get("page_number", "unknown")
        txt = chunk.get("text", "").strip().replace("\n", " ")
        # Truncate very long chunks to keep the prompt size reasonable.
        if len(txt) > 500:
            txt = txt[:500] + "…"
        context_lines.append(f"{i}. (Page {page}) {txt}")

    context_block = "\n".join(context_lines) if context_lines else "No relevant context found."

    return f"""{SYSTEM_PROMPT}

Relevant excerpts from the document:
{context_block}

Question: {question}
"""