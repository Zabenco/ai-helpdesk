from fastapi import FastAPI, Request
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

@app.post("/ask")
async def ask(request: AskRequest):
    if not query_engine:
        return {"error": "No index loaded. Run the ingest script first."}

    print(f"[ASK] Question received: {request.question}")

    # This builds a chat history string
    history = chat_histories.get(request.user_id, [])

    # Add chat history context (if you want to include it in future)
    history_str = ""
    for q, a in history:
        history_str += f"\nPrevious question: {q}\nPrevious answer: {a}\n"

    # This part checks for any overrides via a json file
    override = get_override_for_question(request.question)

    if override:
        print(f"[ASK] Override found and applied: {override}")
        modified_prompt = (
            f"Chat history for context: {history_str}"
            f"Authoritative information: {override} "
            f"User question: {request.question}"
        )
    else:
        modified_prompt = (
            f"Chat history for context: {history_str}"
            f"User question: {request.question}"
        )

    # Asks the question
    response = query_engine.query(modified_prompt)
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
