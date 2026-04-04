# PolicyPal 🤖📄

A conversational RAG (Retrieval Augmented Generation) system that lets you upload any policy document and chat with it in natural language.

Built with FastAPI, Gemini AI, and Pinecone.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple)
![Gemini](https://img.shields.io/badge/Gemini-AI-orange)

## Demo

Upload a document URL → ask questions → get answers with full conversation memory.

Supports PDF, DOCX, and EML files.

## How It Works

PolicyPal uses a RAG pipeline:
```
INGEST (on upload):
Document URL → extract text → chunk into segments →
embed with Gemini → store vectors in Pinecone

QUERY (on each message):
User question → embed question → find similar chunks in Pinecone →
send chunks + chat history → Gemini LLM → answer
```

The key insight: instead of feeding the entire document to an LLM
(expensive, slow, hits context limits), we only send the most
relevant excerpts for each question. This makes answers faster,
cheaper, and more accurate.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| Embeddings | Gemini `gemini-embedding-001` (3072-dim) |
| LLM | Gemini `gemini-2.5-flash` |
| Vector DB | Pinecone Serverless |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Document Parsing | PyMuPDF, python-docx |
| Frontend | Vanilla HTML/CSS/JS |

## Project Structure
```
PolicyPal-rag/
├── main.py                      # FastAPI app entry point
├── frontend/
│   └── index.html               # Chat UI
├── app/
│   ├── api/
│   │   └── routes.py            # /upload and /chat endpoints
│   ├── services/
│   │   ├── document_parser.py   # PDF/DOCX/EML text extraction
│   │   ├── text_chunker.py      # Recursive text splitting
│   │   ├── embedding.py         # Gemini embedding API
│   │   ├── pinecone_store.py    # Vector storage and ingestion
│   │   ├── retrieval.py         # Semantic search
│   │   └── pipeline_qa.py       # RAG orchestration
│   └── utils/
│       └── prompt_builder.py    # Prompt engineering
├── .env.example                 # Environment variables template
└── requirements.txt             # Dependencies
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/ananyagupta17/PolicyPal-rag.git
cd PolicyPal-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

Get your keys here:
- Gemini: [aistudio.google.com](https://aistudio.google.com) → Get API Key (free)
- Pinecone: [app.pinecone.io](https://app.pinecone.io) → API Keys (free tier)

### 4. Run
```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

## API Endpoints

### POST `/api/upload`
Ingests a document from a URL into Pinecone.
```json
Request:  { "document_url": "https://example.com/policy.pdf" }
Response: { "session_id": "abc123...", "message": "Document ingested successfully." }
```

### POST `/api/chat`
Answers a question about the uploaded document.
```json
Request: {
  "session_id": "abc123...",
  "message": "What is the waiting period?",
  "chat_history": []
}
Response: {
  "answer": "The waiting period is 30 days...",
  "chat_history": [...]
}
```

## Key Design Decisions

**Namespace per document** — each document gets its own Pinecone namespace using an md5 hash of the URL. This means multiple documents can coexist without polluting each other's search results.

**Skip re-ingestion** — if the same URL is uploaded again, the app detects existing vectors and skips the embedding step entirely, saving API quota.

**Chat history** — conversation history is passed with every request so Gemini can handle follow-up questions like "tell me more about that" naturally.

**Prompt engineering** — strict instructions prevent hallucination. The model is explicitly told to say "I could not find this" rather than guess.

## License

MIT