import zipfile
import io
import json
import re
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage
import os
import shutil

from app.ingest import load_index, build_index, setup_embedding_model
from app.config import get_llm, get_available_providers, DEFAULT_MODEL_PROVIDER, DEFAULT_MODEL_NAME

# Per-user chat memory buffers
user_memories: dict[str, ChatMemoryBuffer] = {}
MAX_TOKENS = int(os.environ.get("MEMORY_TOKEN_LIMIT", "32000"))

# System prompt for IT professional context
SYSTEM_PROMPT = """You are an AI assistant designed to help IT Support Specialists and IT professionals ONLY.

You assist IT staff with:
- Troubleshooting technical issues
- Following escalation procedures
- Finding relevant KB articles and documentation
- Identifying correct forms, contacts, and departments
- Guiding through documented processes
- Downloading required software or files from appropriate sources

You are NOT customer-facing. You do NOT interact with end users directly.

When answering:
- Reference specific KB articles, policies, or procedures when available
- Include relevant file downloads, form names, or contact information
- Escalation paths should include department/team names and contact info
- If information isn't in the knowledge base, say so clearly"""

index = load_index()
query_engine = None
if index:
    llm = get_llm()
    query_engine = index.as_query_engine(llm=llm)

app = FastAPI(
    title="Universal AI Assistant",
    docs_url=None,
    redoc_url=None,
    swagger_ui_oauth2_redirect_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-frontend-fbc5f.web.app",
        "https://ai.zaben.co",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    user_id: str = "default"

class ModelRequest(BaseModel):
    provider: str = DEFAULT_MODEL_PROVIDER
    model: str = DEFAULT_MODEL_NAME

def _get_memory_path(user_id: str) -> str:
    """Return path to a user's memory JSON file."""
    return os.path.join(MEMORY_DIR, f"{user_id.replace('/', '_').replace(':', '_')}.json")

def _save_memory(user_id: str, memory: ChatMemoryBuffer) -> None:
    """Persist a user's memory buffer to disk."""
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        msgs = []
        try:
            raw = memory.get_all()
            if isinstance(raw, list):
                msgs = [{"role": m.role.value if hasattr(m.role, 'value') else str(m.role), "content": m.content} for m in raw]
        except Exception:
            pass
        path = _get_memory_path(user_id)
        with open(path, "w") as f:
            json.dump({"user_id": user_id, "messages": msgs}, f)
    except Exception as e:
        print(f"[MEMORY] Save error for {user_id}: {e}")

def _load_memory(user_id: str) -> list[dict]:
    """Load persisted memory messages for a user."""
    try:
        path = _get_memory_path(user_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("messages", [])
    except Exception as e:
        print(f"[MEMORY] Load error for {user_id}: {e}")
    return []

# Max history chars fed into the prompt (keeps room for system prompt + retrieved context + response)
MAX_HISTORY_CHARS = 25000

def _load_history_text(user_id: str) -> str:
    """Load history from disk, truncated to MAX_HISTORY_CHARS to avoid context overflow."""
    path = _get_memory_path(user_id)
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            data = json.load(f)
        messages = data.get("messages", [])
        if not messages:
            return ""
        lines = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        full_text = "\n".join(lines)
        # Truncate to avoid context overflow in the LLM prompt
        if len(full_text) > MAX_HISTORY_CHARS:
            full_text = full_text[-MAX_HISTORY_CHARS:]
            full_text = "... [earlier conversation truncated] ...\n\n" + full_text
        return full_text
    except Exception as e:
        print(f"[_HISTORY] Load error: {e}")
        return ""

def get_memory(user_id: str) -> ChatMemoryBuffer:
    """Get or create a chat memory buffer for a user, backed by disk persistence."""
    if user_id not in user_memories:
        llm = get_llm()
        memory = ChatMemoryBuffer.from_defaults(
            llm=llm,
            token_limit=MAX_TOKENS
        )
        saved = _load_memory(user_id)
        for msg in saved:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                memory.put(ChatMessage(role=role, content=content))
        user_memories[user_id] = memory
    return user_memories[user_id]

def build_prompt(question: str, user_id: str, memory: ChatMemoryBuffer, override: str | None) -> str:
    """Build prompt with full conversation history from disk (bypasses token limit)."""
    history_text = _load_history_text(user_id)
    # Fall back to memory buffer.get() if disk read is empty
    if not history_text:
        history_text = memory.get()

    if override:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Full conversation history:\n{history_text}\n\n"
            f"Authoritative information (MANDATORY): {override}\n\n"
            f"User question: {question}"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Full conversation history:\n{history_text}\n\n"
        f"User question: {question}"
    )

@app.post("/ask")
async def ask(request: AskRequest):
    """Standard non-streaming response with full metadata."""
    if not query_engine:
        return {"error": "No index loaded. Run the ingest script first."}

    print(f"[ASK] Question received: {request.question}")

    memory = get_memory(request.user_id)
    override = get_override_for_question(request.question)

    if override:
        print(f"[ASK] Override found and applied: {override}")

    prompt = build_prompt(question=request.question, user_id=request.user_id, memory=memory, override=override)

    response = query_engine.query(prompt)
    answer_text = str(response)

    memory.put(ChatMessage(role="user", content=request.question))
    memory.put(ChatMessage(role="assistant", content=answer_text))
    _save_memory(request.user_id, memory)

    return {
        "question": request.question,
        "answer": answer_text,
        "answer_raw": answer_text,
        "override_used": bool(override),
        "sources": [node.metadata for node in response.source_nodes] if hasattr(response, 'source_nodes') else [],
    }

@app.post("/ask/detailed")
async def ask_detailed(request: AskRequest):
    """Non-streaming response with full metadata including think tags and sources."""
    if not query_engine:
        return {"error": "No index loaded. Run the ingest script first."}

    print(f"[ASK/DETAILED] Question received: {request.question}")

    memory = get_memory(request.user_id)
    override = get_override_for_question(request.question)

    if override:
        print(f"[ASK/DETAILED] Override found and applied: {override}")

    prompt = build_prompt(question=request.question, user_id=request.user_id, memory=memory, override=override)

    response = query_engine.query(prompt)
    answer_text = str(response)

    memory.put(ChatMessage(role="user", content=request.question))
    memory.put(ChatMessage(role="assistant", content=answer_text))
    _save_memory(request.user_id, memory)

    sources = []
    if hasattr(response, 'source_nodes') and response.source_nodes:
        for node in response.source_nodes:
            sources.append({
                "file_name": node.metadata.get('file_name', 'unknown'),
                "page": node.metadata.get('page_label', node.metadata.get('page', 'N/A')),
                "score": getattr(node, 'score', None),
                "text_snippet": node.text[:200] if hasattr(node, 'text') else '',
            })

    return {
        "question": request.question,
        "answer_clean": re.sub(r"<think>[\s\S]*?<\/think>", "", answer_text).strip(),
        "answer_raw": answer_text,
        "override_used": bool(override),
        "sources": sources,
    }

@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Streaming response for real-time answer delivery."""
    if not query_engine:
        return {"error": "No index loaded. Run the ingest script first."}

    print(f"[ASK/STREAM] Question received: {request.question}")

    memory = get_memory(request.user_id)
    override = get_override_for_question(request.question)

    if override:
        print(f"[ASK/STREAM] Override found and applied: {override}")

    prompt = build_prompt(question=request.question, user_id=request.user_id, memory=memory, override=override)

    async def generate():
        full_response = ""
        streaming_response = query_engine.query(prompt)

        response_gen = getattr(streaming_response, 'response_gen', None)
        if response_gen is None:
            response_gen = getattr(streaming_response, 'response_generator', None)
        if response_gen is None:
            response_gen = getattr(streaming_response, 'raw', None)
        if response_gen is None:
            yield str(streaming_response)
            return

        import inspect
        if inspect.iscoroutine(response_gen):
            response_gen = await response_gen

        try:
            async for chunk in response_gen:
                full_response += str(chunk)
                yield chunk
        except TypeError:
            for chunk in response_gen:
                full_response += str(chunk)
                yield chunk

        memory.put(ChatMessage(role="user", content=request.question))
        memory.put(ChatMessage(role="assistant", content=full_response))
        _save_memory(request.user_id, memory)

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear a user's chat history from memory and disk."""
    if user_id in user_memories:
        user_memories[user_id].reset()
        del user_memories[user_id]
    try:
        path = _get_memory_path(user_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    return {"status": "ok", "message": f"History cleared for {user_id}"}

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get a user's chat history as a string (legacy format)."""
    if user_id in user_memories:
        return {"history": user_memories[user_id].get()}
    saved = _load_memory(user_id)
    if saved:
        lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in saved]
        return {"history": "\n".join(lines)}
    return {"history": ""}

@app.get("/history/{user_id}/full")
async def get_full_history(user_id: str):
    """Get a user's full conversation with metadata."""
    history = []
    if user_id in user_memories:
        try:
            raw = user_memories[user_id].get_all()
            if isinstance(raw, list):
                for m in raw:
                    history.append({
                        "role": m.role.value if hasattr(m.role, 'value') else str(m.role),
                        "content": m.content,
                        "timestamp": getattr(m, 'created_at', None),
                    })
        except Exception:
            pass
    if len(history) == 0:
        saved = _load_memory(user_id)
        for m in saved:
            history.append({
                "role": m.get('role', 'user'),
                "content": m.get('content', ''),
                "timestamp": m.get('timestamp', None),
            })
    return {"history": history, "count": len(history)}

@app.get("/models")
async def list_models():
    """List available LLM providers and their status."""
    return {
        "current": {
            "provider": DEFAULT_MODEL_PROVIDER,
            "model": DEFAULT_MODEL_NAME,
        },
        "available": get_available_providers(),
    }

@app.get("/debug")
async def debug_info():
    """Debug info about the backend."""
    providers = get_available_providers()
    llm = get_llm()
    return {
        "index_loaded": index is not None,
        "query_engine_ready": query_engine is not None,
        "llm_type": type(llm).__name__,
        "llm_model": getattr(llm, 'model', DEFAULT_MODEL_NAME),
        "available_providers": providers,
        "max_tokens_from_env": MAX_TOKENS,
        "data_dir": _MNT_DATA,
        "index_dir": INDEX_DIR,
        "memory_dir": MEMORY_DIR,
        "docs_dir": DOCS_DIR,
    }

@app.on_event("startup")
async def startup_event():
    """Load index and warm up the query engine on startup."""
    global index, query_engine
    print("[STARTUP] Loading index...")
    index = load_index()
    if index:
        llm = get_llm()
        query_engine = index.as_query_engine(llm=llm)
        print("[STARTUP] Index loaded and query engine ready.")
    else:
        print("[STARTUP] No index found — using empty knowledge base.")

@app.post("/clear")
async def clear_all():
    """Clear the in-memory index and docs directory (for re-ingest)."""
    global index, query_engine
    removed_index = []
    removed_docs = []
    errors = []

    if os.path.exists(INDEX_DIR):
        for fname in os.listdir(INDEX_DIR):
            fpath = os.path.join(INDEX_DIR, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    removed_index.append(fname)
                    print(f"[CLEAR] Removed index: {fname}")
            except Exception as e:
                errors.append(f"index:{fname}: {str(e)}")
                print(f"[CLEAR] Error removing index {fname}: {e}")

    if os.path.exists(DOCS_DIR):
        for fname in os.listdir(DOCS_DIR):
            fpath = os.path.join(DOCS_DIR, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    removed_docs.append(fname)
                    print(f"[CLEAR] Removed doc: {fname}")
            except Exception as e:
                errors.append(f"doc:{fname}: {str(e)}")
                print(f"[CLEAR] Error removing doc {fname}: {e}")
    else:
        print(f"[CLEAR] DOCS_DIR does not exist: {DOCS_DIR}")

    index = None
    query_engine = None
    print(f"[CLEAR] Done. index={index}, query_engine={query_engine}")

    return {
        "message": "Index and docs cleared." + (" Errors: " + ", ".join(errors) if errors else ""),
        "index_removed": removed_index,
        "docs_removed": removed_docs,
        "errors": errors,
        "paths": {"index_dir": INDEX_DIR, "docs_dir": DOCS_DIR},
    }

@app.post("/upload-index-zip")
async def upload_index_zip(file: UploadFile = File(...)):
    """Receive a zip file containing pre-built index files and extract to INDEX_DIR."""
    global index, query_engine

    if not file.filename.endswith(".zip"):
        return {"error": "Must upload a .zip file"}

    print(f"[INDEX-ZIP] Receiving index zip: {file.filename}")
    content = await file.read()

    os.makedirs(INDEX_DIR, exist_ok=True)
    extracted = []
    errors = []

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                if member.endswith("/") or member.startswith("."):
                    continue
                target = os.path.join(INDEX_DIR, member)
                if not target.startswith(INDEX_DIR):
                    errors.append(f"Unsafe path skipped: {member}")
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted.append(member)
                print(f"[INDEX-ZIP] Extracted: {member}")
    except Exception as e:
        print(f"[INDEX-ZIP] Error: {e}")
        return {"error": f"Failed to extract zip: {str(e)}"}

    try:
        index = load_index()
        if index:
            llm = get_llm()
            query_engine = index.as_query_engine(llm=llm)
            print("[INDEX-ZIP] Index reloaded successfully.")
        else:
            print("[INDEX-ZIP] load_index returned None.")
    except Exception as e:
        print(f"[INDEX-ZIP] Error reloading index: {e}")

    return {
        "message": f"Extracted {len(extracted)} files to index.",
        "extracted": extracted,
        "errors": errors,
    }

# Import at bottom to avoid circular import
from app.overrides import get_override_for_question

# Path configuration — use persistent disk at /mnt/ when available and writable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

_MNT_BASE = os.environ.get("PERSISTENT_ROOT", "")

def _get_data_dir():
    """Return the writable data directory, creating it if needed."""
    if _MNT_BASE:
        try:
            data_dir = _MNT_BASE
            os.makedirs(data_dir, exist_ok=True)
            test_file = os.path.join(data_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return data_dir
        except PermissionError:
            pass
    return PROJECT_ROOT

_MNT_DATA = _get_data_dir()

DOCS_DIR = os.path.join(_MNT_DATA, "docs")
INDEX_DIR = os.path.join(_MNT_DATA, "index")
MEMORY_DIR = os.path.join(_MNT_DATA, "memory")
