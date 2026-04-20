from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import ChatMessage

from app.ingest import load_index
from app.config import get_llm, get_available_providers, DEFAULT_MODEL_PROVIDER, DEFAULT_MODEL_NAME

# Per-user chat memory buffers
user_memories: dict[str, ChatMemoryBuffer] = {}
MAX_TOKENS = int(__import__("os").environ.get("MEMORY_TOKEN_LIMIT", "4096"))

# Load the index
index = load_index()
query_engine = None
if index:
    llm = get_llm()
    query_engine = index.as_query_engine(llm=llm)

app = FastAPI(title="Universal AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    user_id: str = "default"

class ModelRequest(BaseModel):
    provider: str = DEFAULT_MODEL_PROVIDER
    model: str = DEFAULT_MODEL_NAME

def get_memory(user_id: str) -> ChatMemoryBuffer:
    """Get or create a chat memory buffer for a user."""
    if user_id not in user_memories:
        llm = get_llm()
        user_memories[user_id] = ChatMemoryBuffer.from_defaults(
            llm=llm,
            token_limit=MAX_TOKENS
        )
    return user_memories[user_id]

def build_prompt(question: str, memory: ChatMemoryBuffer, override: str | None) -> str:
    """Build the prompt with memory context and override."""
    # Get relevant chat history from memory buffer
    memory_str = memory.get()
    
    if override:
        return (
            f"Chat history for context: {memory_str}\n"
            f"Authoritative information: {override}\n"
            f"User question: {question}"
        )
    else:
        return (
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

    # Full response collected
    response = query_engine.query(prompt)
    answer_text = str(response)

    # Store the conversation in memory
    memory.put(ChatMessage(role="user", content=request.question))
    memory.put(ChatMessage(role="assistant", content=answer_text))

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
        # Stream the response
        streaming_response = query_engine.query(prompt, stream=True)
        async for chunk in streaming_response.response_gen:
            full_response += str(chunk)
            yield chunk

        # Store the conversation in memory after streaming completes
        memory.put(ChatMessage(role="user", content=request.question))
        memory.put(ChatMessage(role="assistant", content=full_response))

    return StreamingResponse(generate(), media_type="text/plain")

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear a user's chat history."""
    if user_id in user_memories:
        user_memories[user_id].reset()
        del user_memories[user_id]
        return {"status": "ok", "message": f"History cleared for {user_id}"}
    return {"status": "ok", "message": f"No history found for {user_id}"}

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get a user's chat history as a string."""
    if user_id in user_memories:
        return {"history": user_memories[user_id].get()}
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

# Import at bottom to avoid circular import
from app.overrides import get_override_for_question
