"""
Multi-backend LLM configuration.
Supports Ollama (local), OpenAI, Anthropic Claude, and MiniMax.
"""
import os

# Default to Ollama if nothing set
DEFAULT_MODEL_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
DEFAULT_MODEL_NAME = os.environ.get("LLM_MODEL", "llama3")

# API Keys (set as environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io/v1")

# Embedding model (shared between ingest and backend startup)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(provider: str = DEFAULT_MODEL_PROVIDER, model: str = DEFAULT_MODEL_NAME):
    """
    Factory function to get an LLM instance based on provider.
    
    Providers:
    - "ollama" : Local Ollama instance (default)
    - "openai" : OpenAI GPT models
    - "anthropic" : Anthropic Claude models
    - "minimax" : MiniMax models
    """
    provider = provider.lower()
    
    if provider == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(model=model)
    
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=model or "gpt-4o",
            api_key=OPENAI_API_KEY
        )
    
    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        from llama_index.llms.anthropic import Anthropic
        return Anthropic(
            model=model or "claude-sonnet-4-20250514"
        )
    
    elif provider == "minimax":
        if not MINIMAX_API_KEY:
            raise ValueError("MINIMAX_API_KEY environment variable not set")
        # Use a direct MiniMax adapter to avoid client-side model-name
        # validation and allow passing MiniMax native model names.
        from app.minimax_llm import MiniMaxLLM
        return MiniMaxLLM(api_key=MINIMAX_API_KEY, api_base=MINIMAX_API_BASE, model=model)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# Minimal MiniMax adapter (fallback) — uses requests directly
# to avoid client-side model name validation. This is a simple
# helper and may need to be adapted to the llama_index LLM
# interface if you want to fully plug it into `as_query_engine()`.
class MiniMaxAdapter:
    def __init__(self, api_key: str, api_base: str, model: str):
        import requests
        self._requests = requests
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model

    def chat(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = self._requests.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        # Typical OpenAI-compatible response structure
        return data["choices"][0]["message"]["content"]
    


def get_available_providers():
    """Return dict of available providers and their status."""
    return {
        "ollama": True,  # Always available if running
        "openai": bool(OPENAI_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "minimax": bool(MINIMAX_API_KEY),
    }
