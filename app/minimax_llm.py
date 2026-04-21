from typing import Any, Dict, Generator, Sequence
import requests
from pydantic import PrivateAttr

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    LLMMetadata,
    ChatMessage,
    CompletionResponseGen,
)


class MiniMaxLLM(CustomLLM):
    """Simple MiniMax LLM adapter using OpenAI-compatible HTTP endpoints.

    This adapter bypasses client-side validation by calling the MiniMax
    API directly via `requests`.
    """

    # private attrs (pydantic)
    _api_key: str = PrivateAttr()
    _api_base: str = PrivateAttr()
    _model: str = PrivateAttr()
    _timeout: int = PrivateAttr()

    def __init__(self, api_key: str, api_base: str, model: str, timeout: int = 30, **kwargs: Any):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self._model, is_chat_model=True)

    def _call_api(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        resp = self._call_api(prompt)
        # Try to extract text from common OpenAI-compatible shapes
        try:
            choice = resp["choices"][0]
            # Some providers embed message in 'message' -> 'content'
            if isinstance(choice, dict) and "message" in choice:
                text = choice["message"].get("content") or choice["message"].get("content_raw") or ""
            else:
                text = choice.get("text", "")
        except Exception:
            text = ""

        return CompletionResponse(text=text, raw=resp, additional_kwargs={})

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # MiniMax streaming support isn't implemented here. Yield a single response
        resp = self._call_api(prompt)
        try:
            choice = resp["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                text = choice["message"]["content"]
            else:
                text = choice.get("text", "")
        except Exception:
            text = ""

        def gen():
            yield CompletionResponse(text=text, raw=resp, additional_kwargs={})

        return gen()

    # Sync wrappers required by BaseLLM abstract interface
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self._complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        return self._stream_complete(prompt, **kwargs)
