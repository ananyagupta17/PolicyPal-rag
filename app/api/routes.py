import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict

from app.services.pinecone_store import ingest_document, ingest_file
from app.services.pipeline_qa import answer_question

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".eml"}

router = APIRouter()


# --- Request / Response Models ---
# These define the shape of data coming in and going out
# Pydantic validates them automatically — wrong types = instant 422 error

class UploadRequest(BaseModel):
    document_url: str          # e.g. "https://example.com/policy.pdf"

class UploadResponse(BaseModel):
    session_id: str            # md5 hash of the URL, used as Pinecone namespace
    message: str               # confirmation message


class ChatRequest(BaseModel):
    session_id: str            # which document to query
    message: str               # the user's question
    chat_history: List[Dict]   # previous turns: [{"role": "user", "content": "..."}]

class ChatResponse(BaseModel):
    answer: str                # Gemini's response
    chat_history: List[Dict]   # updated history including this turn


# --- Endpoints ---

@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest):
    """
    Ingests a document from a URL into Pinecone.
    Steps: download → extract text → chunk → embed → store.
    Returns a session_id the frontend uses for all subsequent chat requests.
    """
    try:
        session_id = ingest_document(request.document_url)
        return UploadResponse(
            session_id=session_id,
            message="Document ingested successfully. You can now ask questions."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/upload-file", response_model=UploadResponse)
async def upload_document_file(file: UploadFile = File(...)):
    """
    Ingests an uploaded file (PDF, DOCX, EML) directly — no URL needed.
    Saves to a temp path, runs the same ingest pipeline, returns session_id.
    """
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        session_id = ingest_file(tmp_path)
        return UploadResponse(
            session_id=session_id,
            message="Document ingested successfully. You can now ask questions."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answers a question about the uploaded document.
    Uses the session_id to find the right Pinecone namespace,
    retrieves relevant chunks, and asks Gemini to answer.
    Chat history is passed in and returned updated — the frontend
    is responsible for storing and sending history each time.
    """
    try:
        answer = answer_question(
            document_url="",           # not needed — session_id carries the namespace
            question=request.message,
            chat_history=request.chat_history,
            session_id=request.session_id,  # pass directly
        )

        # Append this turn to history and return it
        updated_history = request.chat_history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": answer},
        ]

        return ChatResponse(answer=answer, chat_history=updated_history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")