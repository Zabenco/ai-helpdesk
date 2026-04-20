import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter

# Resolve paths relative to this script's location, not CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")

# Chunking configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "256"))

# Embedding model configuration
# Supports: "ollama" (local) or "openai" (works with MiniMax, OpenAI, etc.)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "nomic-embed-text")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.chat/v1")

def setup_embedding_model():
    """Configure the embedding model based on provider."""
    if EMBED_PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
    elif EMBED_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        # Use MiniMax API if key is set, otherwise falls back to OpenAI
        api_key = MINIMAX_API_KEY or OPENAI_API_KEY
        api_base = MINIMAX_API_BASE if MINIMAX_API_KEY else "https://api.openai.com/v1"
        Settings.embed_model = OpenAIEmbedding(
            model=EMBED_MODEL_NAME,
            api_key=api_key,
            api_base=api_base,
        )
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {EMBED_PROVIDER}")

# Setup embedding model on import
setup_embedding_model()

# Load valid documents from the docs folder, including subfolders
def load_documents():
    reader = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        recursive=True,
        required_exts=[".txt", ".pdf", ".md", ".csv"],
    )
    docs = reader.load_data()
    return docs

# Where the magic happens, building the index from the documents
def build_index():
    print("[Ingest] Loading all documents")
    documents = load_documents()

    print(f"[Ingest] Loaded {len(documents)} documents")

    if not documents:
        print("[Ingest] Keeping it real, there isn't anything to index :(")
        return
    
    # Configure chunking
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    print(f"[Ingest] Building vector index with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    print(f"[Ingest] Using embedding provider: {EMBED_PROVIDER}, model: {EMBED_MODEL_NAME}")
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
    )

    print("[Ingest] Saving built index to your disk...")
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print("[Ingest] All done")


def load_index():
    if not os.path.exists(INDEX_DIR):
        print("[Ingest] No index found anywhere. You gotta run 'build_index()' first.")
        return None
    
    print("[Ingest] Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    return load_index_from_storage(storage_context)

if __name__ == "__main__":
    build_index()
