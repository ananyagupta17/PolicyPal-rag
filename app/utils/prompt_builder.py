from typing import List, Dict


def build_chat_prompt(
    context_chunks: List[Dict],
    chat_history: List[Dict],
    question: str,
) -> str:
    """
    Builds a prompt for Gemini that includes:
    - Retrieved policy excerpts (the RAG context)
    - Conversation history (so follow-up questions work)
    - The current question
    - Strict instructions to prevent hallucination
    """

    # --- 1. Format the retrieved chunks ---
    # These are the most semantically similar chunks from Pinecone
    # We join them as numbered excerpts so Gemini can reference them
    if context_chunks:
        context = "\n\n".join([
            f"[Excerpt {i+1}]: {chunk.get('text', '')}"
            for i, chunk in enumerate(context_chunks)
        ])
    else:
        context = "No relevant excerpts found."

    # --- 2. Format the conversation history ---
    # We replay previous turns so Gemini understands follow-up questions
    # e.g. "what about for dependents?" only makes sense with prior context
    history_text = ""
    if chat_history:
        turns = []
        for turn in chat_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            turns.append(f"{role}: {turn['content']}")
        history_text = "\n".join(turns)

    # --- 3. Build the full prompt ---
    prompt = f"""You are PolicyPal, an AI assistant that answers questions strictly based on policy documents.

POLICY EXCERPTS (retrieved based on the question):
{context}

{"CONVERSATION SO FAR:" + chr(10) + history_text if history_text else ""}

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Answer in a natural, conversational tone — don't just copy text from the excerpts.
- Rephrase and summarize the relevant information in your own words.
- If the answer is not in the excerpts, say exactly: "I could not find this information in the document."
- Be concise but friendly. You can use 2-3 sentences.
- Quote specific figures, dates, or names exactly as written when relevant.
- If the question refers to something from the conversation history, use that context naturally.
""".strip()

    return prompt