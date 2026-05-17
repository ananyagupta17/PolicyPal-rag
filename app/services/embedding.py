import os
import hashlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"  # 3072-dim, free tier
DIMENSION = 3072

embedding_cache = {}


def get_embedding(text: str) -> list[float]:
    """
    Return a 3072-dim embedding using Gemini gemini-embedding-001.
    Uses md5-based in-memory cache for repeated inputs.
    """
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    embedding = result.embeddings[0].values
    embedding_cache[text_hash] = embedding
    return embedding