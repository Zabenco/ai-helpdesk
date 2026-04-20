import os
import requests
from typing import Any
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.embeddings.base import BaseEmbedding

# Resolve paths relative to this script's location, not CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load .env file from project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")

# Chunking configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "256"))

# Embedding model configuration
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "nomic-embed-text")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.chat/v1")


class CustomEmbedding(BaseEmbedding):
    """Custom embedding class that calls OpenAI-compatible API directly.
    
    Bypasses llama-index's hardcoded model enum to support any
    OpenAI-compatible embedding model (MiniMax, local proxies, etc.)
    """
    model_config = {
        "extra": "allow",  # Allow extra fields not defined in parent
    }
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ):
        # Initialize with dummy model_name to satisfy Pydantic
        super().__init__(model_name=model, embed_batch_size=embed_batch_size, **kwargs)
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model
    
    def _get_text_embedding(self, text: str) -> list[float]:
        response = requests.post(
            f"{self._api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": text,
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self._api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
    
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)
    
    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)


def setup_embedding_model():
    """Configure the embedding model based on provider."""
    if EMBED_PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
    elif EMBED_PROVIDER == "openai":
        # Use MiniMax API if key is set, otherwise falls back to OpenAI
        api_key = MINIMAX_API_KEY or OPENAI_API_KEY
        api_base = MINIMAX_API_BASE if MINIMAX_API_KEY else "https://api.openai.com/v1"
        Settings.embed_model = CustomEmbedding(
            api_key=api_key,
            api_base=api_base,
            model=EMBED_MODEL_NAME,
        )
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {EMBED_PROVIDER}")


# Supported file extensions
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".md", ".csv", ".docx", ".pptx", ".xlsx"]


def load_documents():
    """Load all supported documents from the docs directory."""
    if not os.path.exists(DOCS_DIR):
        print(f"[Ingest] Docs directory not found: {DOCS_DIR}")
        return []
    
    file_count = len([f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))])
    print(f"[Ingest] Found {file_count} files in docs directory")
    
    reader = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        recursive=True,
        exclude_hidden=False,  # Don't exclude hidden files
        exclude=None,         # Don't exclude any files by pattern
        required_exts=SUPPORTED_EXTENSIONS,
    )
    docs = reader.load_data()
    return docs


def build_index():
    """Build the vector index from documents."""
    print("[Ingest] Loading all documents")
    documents = load_documents()

    print(f"[Ingest] Loaded {len(documents)} documents")

    if not documents:
        print("[Ingest] No documents to index. Add files to the docs folder and try again.")
        return
    
    # Configure chunking
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    print(f"[Ingest] Building vector index with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    print(f"[Ingest] Using embedding provider: {EMBED_PROVIDER}, model: {EMBED_MODEL_NAME}")
    
    # Setup embedding model FIRST before any Settings.embed_model access
    setup_embedding_model()
    
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
    )

    print("[Ingest] Saving built index to your disk...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print("[Ingest] All done! Your helpdesk is ready.")


def load_index():
    """Load the existing vector index."""
    if not os.path.exists(INDEX_DIR):
        print("[Ingest] No index found. Run 'build_index()' first.")
        return None
    
    print("[Ingest] Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    return load_index_from_storage(storage_context)


if __name__ == "__main__":
    build_index()
