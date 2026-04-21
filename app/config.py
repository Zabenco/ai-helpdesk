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
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.chat/v1")

# Embedding model (Ollama only for now)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "nomic-embed-text")


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
        from llama_index.llms.openai import OpenAI
        # MiniMax uses OpenAI-compatible API with MiniMax-M2.7 model
        return OpenAI(
            model="MiniMax-M2.7",
            api_key=MINIMAX_API_KEY,
            api_base=f"{MINIMAX_API_BASE}"
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_available_providers():
    """Return dict of available providers and their status."""
    return {
        "ollama": True,  # Always available if running
        "openai": bool(OPENAI_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "minimax": bool(MINIMAX_API_KEY),
    }
