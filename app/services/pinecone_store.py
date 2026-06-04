import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from app.services.embedding import get_embedding
from app.services.text_chunker import chunk_text
from app.services.document_parser import extract_text

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

INDEX_NAME = "policy-embeddings"
DIMENSION = 3072
CLOUD = "aws"
REGION = "us-east-1"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist (serverless, free tier)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)


def generate_source_id(document_url: str) -> str:
    """
    Creates a stable unique ID from the document URL using md5 hash.
    Same URL always produces the same source_id — prevents duplicates.
    """
    return hashlib.md5(document_url.encode()).hexdigest()


def clear_namespace(source_id: str):
    """
    Wipes all vectors in a namespace before re-ingesting.
    Important when switching embedding models — old vectors are incompatible.
    """
    try:
        index.delete(delete_all=True, namespace=source_id)
        print(f"Cleared namespace: {source_id}")
    except Exception as e:
        print(f"Could not clear namespace (may be empty): {e}")


def store_embeddings_for_text(text: str, source_id: str) -> int:
    """
    Chunks the text, embeds each chunk with Gemini,
    and upserts into Pinecone under the document's namespace.
    Returns the number of vectors stored.
    """
    chunks = chunk_text(text)
    vectors = []

    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        vector_id = f"{i:06d}"          # e.g. "000001", "000002"
        embedding = get_embedding(chunk)
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i,
                "source": source_id,
            }
        })

    # Namespace per document = clean isolation, no cross-doc contamination
    index.upsert(vectors=vectors, namespace=source_id)
    print(f"Stored {len(vectors)} embeddings under namespace={source_id}")
    return len(vectors)


def _check_and_ingest(source_id: str, file_path_or_url: str) -> str:
    """Shared logic: skip if already ingested, otherwise extract + embed + store."""
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        if source_id in namespaces and namespaces[source_id].get("vector_count", 0) > 0:
            print(f"Document already ingested (namespace={source_id}), skipping.")
            return source_id
    except Exception as e:
        print(f"Could not check index stats: {e}")

    clear_namespace(source_id)
    text = extract_text(file_path_or_url)
    if not text or not text.strip():
        raise ValueError("No extractable text found in the document.")

    store_embeddings_for_text(text, source_id=source_id)
    return source_id


def ingest_document(document_url: str) -> str:
    """
    Full ingestion pipeline for a document URL.
    Skips re-ingestion if the document was already uploaded before.
    Returns the source_id (used as namespace for querying).
    """
    source_id = generate_source_id(document_url)
    return _check_and_ingest(source_id, document_url)


def ingest_file(local_path: str) -> str:
    """
    Full ingestion pipeline for a locally saved file (e.g. from a multipart upload).
    source_id is derived from md5 of file content so the same file never gets re-ingested.
    Returns the source_id (used as namespace for querying).
    """
    with open(local_path, "rb") as f:
        source_id = hashlib.md5(f.read()).hexdigest()
    return _check_and_ingest(source_id, local_path)