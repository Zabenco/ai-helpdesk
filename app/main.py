from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.ingest import load_index
from llama_index.llms.ollama import Ollama
from app.overrides import get_override_for_question

chat_histories = {}
MAX_HISTORY_LENGTH = 6

# Load the index
index = load_index()
query_engine = None
if index:
    llm = Ollama(model="llama3")
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

def build_prompt(question: str, history: list, override: str | None) -> str:
    """Build the prompt with history and override context."""
    history_str = ""
    for q, a in history:
        history_str += f"\nPrevious question: {q}\nPrevious answer: {a}\n"

    if override:
        return (
            f"Chat history for context: {history_str}"
            f"Authoritative information: {override} "
            f"User question: {question}"
        )
    else:
        return (
            f"Chat history for context: {history_str}"
            f"User question: {question}"
        )

@app.post("/ask")
async def ask(request: AskRequest):
    """Standard non-streaming response with full metadata."""
    if not query_engine:
        return {"error": "No index loaded. Run the ingest script first."}

    print(f"[ASK] Question received: {request.question}")

    history = chat_histories.get(request.user_id, [])
    override = get_override_for_question(request.question)

    if override:
        print(f"[ASK] Override found and applied: {override}")

    prompt = build_prompt(request.question, history, override)

    # Full response collected
    response = query_engine.query(prompt)
    answer_text = str(response)

    # Update chat history
    history.append((request.question, answer_text))
    chat_histories[request.user_id] = history[-MAX_HISTORY_LENGTH:]

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

    history = chat_histories.get(request.user_id, [])
    override = get_override_for_question(request.question)

    if override:
        print(f"[ASK/STREAM] Override found and applied: {override}")

    prompt = build_prompt(request.question, history, override)

    async def generate():
        full_response = ""
        # Stream the response
        streaming_response = query_engine.query(prompt, stream=True)
        async for chunk in streaming_response.response_gen:
            full_response += str(chunk)
            yield chunk

        # Update chat history after streaming completes
        history.append((request.question, full_response))
        chat_histories[request.user_id] = history[-MAX_HISTORY_LENGTH:]

    return StreamingResponse(generate(), media_type="text/plain")

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear a user's chat history."""
    if user_id in chat_histories:
        del chat_histories[user_id]
        return {"status": "ok", "message": f"History cleared for {user_id}"}
    return {"status": "ok", "message": f"No history found for {user_id}"}
