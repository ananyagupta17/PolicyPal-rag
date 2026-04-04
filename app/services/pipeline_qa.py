import os
import hashlib
from typing import List, Dict
from google import genai
from dotenv import load_dotenv

from app.services.retrieval import semantic_search
from app.utils.prompt_builder import build_chat_prompt

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def answer_question(
    document_url: str,
    question: str,
    chat_history: List[Dict],
    top_k: int = 8,
    session_id: str = None,
) -> str:
    """
    Given a session_id, a user question, and conversation history,
    retrieve relevant chunks from Pinecone and ask Gemini to answer.
    Chat history gives Gemini memory for follow-up questions.
    """
    source_id = session_id if session_id else hashlib.md5(document_url.encode()).hexdigest()

    # Retrieve most relevant chunks for this question
    context_chunks = semantic_search(
        question,
        top_k=top_k,
        namespace=source_id,
        fltr={"source": {"$eq": source_id}},
    )

    # Build prompt with context + history + question
    prompt = build_chat_prompt(
        context_chunks=context_chunks,
        chat_history=chat_history,
        question=question,
    )

    # Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate an answer. Error: {str(e)}"