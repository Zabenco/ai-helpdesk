# AI Helpdesk — Intelligent IT Support Assistant

A RAG-powered helpdesk that answers IT questions using your knowledge base. Built for SLU IT Support and deployable to production with minimal configuration.

## Features

- **RAG Q&A** — Answers questions by searching your indexed documents
- **Multi-backend LLM** — Supports MiniMax (cloud), OpenAI, Anthropic, and Ollama (local)
- **OpenAI Embeddings** — Fast, cheap, no rate limits for query embedding
- **Per-user Chat Memory** — Conversation context persists across sessions
- **Override System** — Priority responses for emergency keywords
- **Admin Panel** — Upload docs, re-index, clear index, view user history (Firebase frontend)
- **Multi-format Support** — PDF, TXT, MD, CSV, DOCX, PPTX, XLSX
- **Deploy Ready** — Render + Firebase, persistent disk storage

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Firebase Frontend (ai-frontend-fbc5f.web.app)          │
│  React + Vite + Firebase Auth                           │
└────────────────┬────────────────────────────────────────┘
                 │  POST /ask, /upload, /clear-index
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Render Backend (ai-helpdesk-bqkv.onrender.com)         │
│  FastAPI + LlamaIndex + RAG pipeline                    │
│                                                          │
│  Chat LLM:  MiniMax-M2.7 (or OpenAI/gpt-4o-mini)      │
│  Embedding: OpenAI text-embedding-3-small               │
│  Index:    Persistent disk at /mnt/data/index            │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  MiniMax API (chat)  +  OpenAI API (embeddings)         │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Zabenco/ai-helpdesk.git
cd ai-helpdesk
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```bash
# LLM (chat) — choose one
LLM_PROVIDER=minimax      # cloud, uses MiniMax-M2.7
LLM_MODEL=MiniMax-M2.7

# OR for OpenAI chat:
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini

# Embeddings (required for query-time retrieval)
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# MiniMax (only needed if LLM_PROVIDER=minimax)
MINIMAX_API_KEY=sk-cp-...
MINIMAX_API_BASE=https://api.minimax.io/v1

# Optional: local Ollama for dev (embeddings + chat offline)
# EMBED_PROVIDER=ollama
# LLM_PROVIDER=ollama
# EMBED_MODEL=nomic-embed-text
# LLM_MODEL=llama3
```

### 3. Install & ingest

```bash
pip install -r requirements.txt
python -m app.ingest
```

### 4. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

Visit `http://localhost:8000/docs` for the interactive API docs.

---

## Document Ingest

### Local ingest (dev)

On your local machine, add docs to `docs/` then run:

```bash
python -m app.ingest
```

This builds the vector index using your configured embedder (OpenAI by default).

### Production ingest (admin panel)

Upload docs via the Admin → Upload Files tab in the Firebase frontend. The backend receives files, saves to `docs/`, and triggers a re-index automatically.

### Pre-built index upload

If you have a pre-built index (from local ingest), upload it via Admin → Upload Index Zip. Useful for migrating an index from local dev to production.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question, returns answer + sources |
| `POST` | `/ask/stream` | Streaming response |
| `GET` | `/history/{user_id}` | Get user's chat history |
| `DELETE` | `/history/{user_id}` | Clear user's history |
| `GET` | `/models` | List available LLM providers |
| `GET` | `/debug` | Directory status, index state |
| `POST` | `/upload` | Upload docs + auto re-index |
| `POST` | `/upload-index-zip` | Upload pre-built index as zip |
| `POST` | `/clear-index` | Wipe index + docs (admin) |

### Example

```bash
curl -X POST https://ai-helpdesk-bqkv.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset a password?", "user_id": "ezabenco@gmail.com"}'
```

```json
{
  "question": "How do I reset a password?",
  "answer": "To reset a password, follow these steps from the IT policy...",
  "override_used": false,
  "sources": [
    {
      "file_name": "policies.txt",
      "file_path": "D:\\Coding Projects\\ai-helpdesk\\docs\\policies.txt"
    }
  ]
}
```

---

## Configuration

### Override System

Add priority responses in `overrides.json`:

```json
{
  "vaporized": "EMERGENCY: Follow the Unexpected Vaporization Procedure. Contact the Incident Commander immediately.",
  "server down": "Check the server status dashboard and escalate to NOC."
}
```

When a question contains an override keyword, that response is used with document context.

### Render Deployment

On Render:
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port 10000`
- **Build command:** `pip install -r requirements.txt`
- **Disk:** Mount at `/mnt/` for persistent `docs/` and `index/` storage
- **Env vars:** Set in Render dashboard — `OPENAI_API_KEY`, `EMBED_PROVIDER=openai`, `EMBED_MODEL=text-embedding-3-small`, `LLM_PROVIDER`, `LLM_MODEL`

### Firebase Frontend

Frontend repo: `https://github.com/Zabenco/ai-helpdesk-frontend`

Deployed at: `https://ai-frontend-fbc5f.web.app`

Admin login required. Admin email: `ezabenco@gmail.com`

---

## Supported File Types

| Type | Extension | Notes |
|------|-----------|-------|
| Plain text | `.txt` | Direct read |
| PDF | `.pdf` | Text extraction via pypdf |
| Markdown | `.md` | Direct read |
| CSV | `.csv` | Raw text (structured parsing TBD) |
| Word | `.docx` | Via python-docx |
| Excel | `.xlsx` | Via openpyxl |
| PowerPoint | `.pptx` | Via python-pptx |

---

## Troubleshooting

**"No index found"** — Run `python -m app.ingest` locally, or upload index via admin panel.

**Embedding errors / 401** — Verify `OPENAI_API_KEY` is set in Render env vars.

**Rate limits** — Use OpenAI embeddings (`EMBED_PROVIDER=openai`) instead of MiniMax embeddings.

**CORS errors** — Backend allows `https://ai-frontend-fbc5f.web.app` and `http://localhost:5173`. Update `allow_origins` in `main.py` if adding new frontend origins.

**Empty index after deploy** — Render disk is ephemeral on free tier. Use the Upload Index Zip feature or set `PERSISTENT_ROOT=/mnt/` with a persistent disk mount.

---

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — Web framework
- **[LlamaIndex](https://www.llamaindex.ai/)** — RAG framework
- **[MiniMax](https://platform.minimax.io/)** — Chat LLM (cloud)
- **[OpenAI](https://openai.com/)** — Embeddings + optional chat
- **[Ollama](https://ollama.ai/)** — Local dev LLM
- **[Firebase](https://firebase.google.com/)** — Frontend hosting + auth
- **[Render](https://render.com/)** — Backend hosting

---

**Built with love, by Ethan Zabenco**