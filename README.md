# PolicyPal

**Conversational document QA over any policy file — powered by RAG, Gemini, and Pinecone.**

Upload a PDF, DOCX, or EML. Ask questions in plain English. Get answers grounded strictly in the document, with full conversation memory.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple)
![Gemini](https://img.shields.io/badge/Gemini-2.5-orange)

---

## What it does

Most LLMs can't reliably answer questions about long documents — they hit context limits, hallucinate, or lose precision on dense legal/policy text.

PolicyPal solves this with a **Retrieval Augmented Generation (RAG)** pipeline: instead of feeding the entire document to the model, it embeds the document into a vector store, retrieves only the most semantically relevant chunks per question, and grounds the LLM's answer strictly in those excerpts. The result is accurate, fast, and cost-efficient at any document length.

---

## Architecture

```
INGEST
Document (URL or file upload)
  → extract text (PyMuPDF / python-docx / email)
  → recursive chunking (1200 chars, 250 overlap)
  → embed each chunk (Gemini gemini-embedding-001, 3072-dim)
  → upsert to Pinecone under a per-document namespace

QUERY
User question
  → embed question (same model)
  → cosine similarity search in Pinecone (top-k=8)
  → inject retrieved chunks + conversation history into prompt
  → Gemini gemini-2.5-flash generates a grounded answer
```

---

## Key Design Decisions

**Namespace-per-document isolation** — each document is hashed (MD5) to a unique Pinecone namespace. Multiple documents coexist without polluting each other's retrieval results, and the same hash serves as the session ID passed to the frontend.

**Idempotent ingestion** — before embedding, the pipeline checks if the namespace already has vectors. If it does, the upload is a no-op. This eliminates redundant Gemini API calls on duplicate uploads.

**Stateless chat API** — conversation history is owned by the client and sent with each request. The server is fully stateless, which keeps it horizontally scalable with no session storage.

**Hallucination guardrails** — the prompt explicitly instructs the model to respond with "I could not find this information in the document" rather than infer or guess. The model is given only retrieved excerpts, never the full document.

**Dual ingestion modes** — documents can be ingested via a public URL (downloaded server-side) or via direct multipart file upload. Both paths converge on the same chunking and embedding pipeline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Embeddings | Gemini `gemini-embedding-001` — 3072-dim vectors |
| LLM | Gemini `gemini-2.5-flash` |
| Vector DB | Pinecone Serverless (AWS us-east-1, cosine similarity) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Document Parsing | PyMuPDF (PDF), python-docx (DOCX), stdlib email (EML) |
| Frontend | Vanilla HTML/CSS/JS |

---

## API

### `POST /api/upload`
Ingest a document from a URL.
```json
Request:  { "document_url": "https://example.com/policy.pdf" }
Response: { "session_id": "a3f9...", "message": "Document ingested successfully." }
```

### `POST /api/upload-file`
Ingest a document via direct file upload (`multipart/form-data`).
```
form-data: file=<PDF|DOCX|EML>
Response:  { "session_id": "b7c2...", "message": "Document ingested successfully." }
```

### `POST /api/chat`
Answer a question against the ingested document.
```json
Request: {
  "session_id": "a3f9...",
  "message": "What is the waiting period for pre-existing conditions?",
  "chat_history": []
}
Response: {
  "answer": "The waiting period is 36 months...",
  "chat_history": [...]
}
```

---

## Setup

```bash
git clone https://github.com/ananyagupta17/PolicyPal-rag.git
cd PolicyPal-rag
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY and PINECONE_API_KEY
uvicorn main:app --reload
```

Open `http://localhost:8000`.

**Get free API keys:**
- Gemini — [aistudio.google.com](https://aistudio.google.com)
- Pinecone — [app.pinecone.io](https://app.pinecone.io)

---

## Project Structure

```
app/
├── api/routes.py              # REST endpoints + request/response models
├── services/
│   ├── pipeline_qa.py         # RAG orchestration (retrieve → prompt → generate)
│   ├── pinecone_store.py      # Vector upsert, namespace management, ingestion
│   ├── retrieval.py           # Semantic search over Pinecone
│   ├── embedding.py           # Gemini embedding calls + in-memory cache
│   ├── document_parser.py     # PDF / DOCX / EML text extraction
│   └── text_chunker.py        # LangChain recursive splitting
└── utils/prompt_builder.py    # Prompt assembly (context + history + question)
main.py                        # FastAPI app, CORS, static file serving
sample_docs/                   # Sample policy files for testing
```

---

## License

MIT
