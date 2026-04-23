import os
import requests
import time
from typing import Any
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

# Use persistent disk mount at /mnt/ when available and writable
_MNT_BASE = os.environ.get("PERSISTENT_ROOT", "")

def _get_data_dir():
    """Return the writable data directory."""
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

# Chunking configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "256"))

# Embedding model configuration
# Priority: OpenAI (text-embedding-3-small, 1536 dims) > Ollama (nomic-embed-text, 768 dims)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")  # default to openai since that's what's on Render
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


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
        for attempt in range(3):
            payload = {
                "model": self._model,
                "texts": [text],
                "type": "query",
            }
            response = requests.post(
                f"{self._api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=15,
            )
            print(f"[Embed] Status: {response.status_code}, Response: {response.text[:500]}")
            
            if response.status_code == 200:
                json_data = response.json()
                vectors = json_data.get("vectors")
                if vectors:
                    return vectors[0]
                status_msg = json_data.get("base_resp", {}).get("status_msg", "")
            else:
                status_msg = str(response.text)

            if attempt < 2 and ("rate limit" in status_msg.lower() or "RPM" in status_msg or response.status_code == 429):
                wait = 2 ** attempt
                print(f"[Embed] Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise ValueError(f"Embedding API error: {response.status_code} {status_msg[:200]}")
        raise ValueError("Embedding failed after 3 retries")

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._model,
            "texts": texts,
            "type": "db",
        }
        response = requests.post(
            f"{self._api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if response.status_code != 200:
            print(f"[Embed Error] Status: {response.status_code}")
            print(f"[Embed Error] Response: {response.text}")
            response.raise_for_status()

        json_data = response.json()
        vectors = json_data.get("vectors")
        if not vectors:
            print(f"[Embed Error] Response missing vectors: {json_data}")
            raise ValueError(f"Embedding API returned no vectors: {json_data}")
        return [v for v in vectors]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)


def setup_embedding_model():
    """Configure the embedding model based on provider."""
    if EMBED_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI embeddings")
        Settings.embed_model = OpenAIEmbedding(
            model=EMBED_MODEL_NAME or "text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        )
    elif EMBED_PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL_NAME or "nomic-embed-text",
            base_url=OLLAMA_BASE_URL,
        )
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {EMBED_PROVIDER}. Use 'openai' or 'ollama'.")


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
        print("[Ingest] No index directory found. Run 'build_index()' first.")
        return None

    docstore_path = os.path.join(INDEX_DIR, "docstore.json")
    if not os.path.exists(docstore_path):
        print(f"[Ingest] No docstore.json found in index dir — index not built yet or corrupted. Run 'build_index()' first.")
        return None

    print("[Ingest] Loading index from disk...")
    setup_embedding_model()
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    return load_index_from_storage(storage_context)


if __name__ == "__main__":
    build_index()
