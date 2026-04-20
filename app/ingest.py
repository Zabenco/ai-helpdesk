import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Resolve paths relative to this script's location, not CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
EMBED_MODEL_NAME = "nomic-embed-text"

# Chunking configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "256"))

# Setup the embedding model here
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

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
