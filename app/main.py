import zipfile
import io
import json
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
MAX_TOKENS = int(__import__("os").environ.get("MEMORY_TOKEN_LIMIT", "4096"))

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
    query_engine = index.as_query_engine(llm=llm, streaming=True)

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
        # ChatMemoryBuffer stores messages via .messages attribute
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

def get_memory(user_id: str) -> ChatMemoryBuffer:
    """Get or create a chat memory buffer for a user, backed by disk persistence."""
    if user_id not in user_memories:
        llm = get_llm()
        memory = ChatMemoryBuffer.from_defaults(
            llm=llm,
            token_limit=MAX_TOKENS
        )
        # Restore persisted messages
        saved = _load_memory(user_id)
        for msg in saved:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                memory.put(ChatMessage(role=role, content=content))
        user_memories[user_id] = memory
    return user_memories[user_id]

def build_prompt(question: str, memory: ChatMemoryBuffer, override: str | None) -> str:
    """Build the prompt with memory context and override."""
    memory_str = memory.get()
    if override:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Chat history for context: {memory_str}\n"
            f"Authoritative information (MANDATORY): {override}\n"
            f"User question: {question}"
        )
    else:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Chat history for context: {memory_str}\n"
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

    prompt = build_prompt(question=request.question, memory=memory, override=override)

    response = query_engine.query(prompt)
    answer_text = str(response)

    memory.put(ChatMessage(role="user", content=request.question))
    memory.put(ChatMessage(role="assistant", content=answer_text))
    _save_memory(request.user_id, memory)

    return {
        "question": request.question,
        "answer": answer_text,
        "override_used": bool(override),
        "sources": [node.metadata for node in response.source_nodes] if hasattr(response, 'source_nodes') else [],
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

    prompt = build_prompt(question=request.question, memory=memory, override=override)

    async def generate():
        full_response = ""
        streaming_response = query_engine.query(prompt)
        
        # Get the response generator — different llama-index versions use different attribute names
        response_gen = getattr(streaming_response, 'response_gen', None)
        if response_gen is None:
            response_gen = getattr(streaming_response, 'response_generator', None)
        if response_gen is None:
            response_gen = getattr(streaming_response, 'raw', None)
        if response_gen is None:
            # Last resort: treat the whole response as text
            yield str(streaming_response)
            return
        
        # response_gen may be sync or async — handle both
        try:
            async for chunk in response_gen:
                full_response += str(chunk)
                yield chunk
        except TypeError:
            # Sync generator passed to async for — iterate synchronously
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
    # Also delete the persisted file
    try:
        path = _get_memory_path(user_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    return {"status": "ok", "message": f"History cleared for {user_id}"}

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get a user's chat history as a string."""
    if user_id in user_memories:
        return {"history": user_memories[user_id].get()}
    # Try loading from disk if not in memory
    saved = _load_memory(user_id)
    if saved:
        lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in saved]
        return {"history": "\n".join(lines)}
    return {"history": ""}

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
async def debug_status():
    """Return current state of docs and index directories."""
    def list_dir(path):
        if not os.path.exists(path):
            return {"exists": False}
        try:
            items = os.listdir(path)
            return {"exists": True, "files": items, "count": len(items)}
        except Exception as e:
            return {"exists": True, "error": str(e)}

    return {
        "index_dir": INDEX_DIR,
        "index_status": list_dir(INDEX_DIR),
        "docs_dir": DOCS_DIR,
        "docs_status": list_dir(DOCS_DIR),
        "index_loaded": index is not None,
        "query_engine_ready": query_engine is not None,
        "persistent_root": _MNT_BASE or "(not set)",
    }

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Receive new documents, save to docs/, and rebuild the index."""
    global index, query_engine

    print(f"[UPLOAD] Starting. DOCS_DIR={DOCS_DIR}")
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []

    for file in files:
        dest = os.path.join(DOCS_DIR, file.filename)
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        print(f"[UPLOAD] Saved: {file.filename}")
        saved.append(file.filename)

    if not saved:
        return {"message": "No files saved.", "files": []}

    try:
        print("[UPLOAD] Calling build_index()...")
        build_index()
        print("[UPLOAD] build_index complete. Loading index...")
        index = load_index()
        if index:
            llm = get_llm()
            query_engine = index.as_query_engine(llm=llm)
            print("[UPLOAD] query_engine ready.")
        return {
            "message": f"Indexed {len(saved)} file(s). Index rebuilt.",
            "files": saved,
        }
    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        return {"message": f"Files saved but re-index failed: {str(e)}", "files": saved}

@app.post("/clear-index")
async def clear_index():
    """Delete all files in the index folder and clear the docs folder."""
    global index, query_engine

    removed_index = []
    removed_docs = []
    errors = []

    print(f"[CLEAR] Starting. INDEX_DIR={INDEX_DIR}, DOCS_DIR={DOCS_DIR}")

    if os.path.exists(INDEX_DIR):
        for fname in os.listdir(INDEX_DIR):
            fpath = os.path.join(INDEX_DIR, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    removed_index.append(fname)
                    print(f"[CLEAR] Removed file: {fname}")
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                    removed_index.append(fname + "/")
                    print(f"[CLEAR] Removed dir: {fname}/")
            except Exception as e:
                errors.append(f"{fname}: {str(e)}")
                print(f"[CLEAR] Error removing {fname}: {e}")
    else:
        print(f"[CLEAR] INDEX_DIR does not exist: {INDEX_DIR}")

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
                # Skip directories and hidden files
                if member.endswith("/") or member.startswith("."):
                    continue
                target = os.path.join(INDEX_DIR, member)
                # Prevent zip slip
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
    
    # Reload index
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
            # Test writeability
            test_file = os.path.join(data_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return data_dir
        except PermissionError:
            pass
    # Fallback to project directory (works on Render + local dev)
    return PROJECT_ROOT

_MNT_DATA = _get_data_dir()

DOCS_DIR = os.path.join(_MNT_DATA, "docs")
INDEX_DIR = os.path.join(_MNT_DATA, "index")
MEMORY_DIR = os.path.join(_MNT_DATA, "memory")