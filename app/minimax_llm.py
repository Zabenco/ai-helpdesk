import httpx
import json
from typing import Any, AsyncIterator
from pydantic import PrivateAttr

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    LLMMetadata,
)


class MiniMaxLLM(CustomLLM):
    """MiniMax LLM adapter using OpenAI-compatible HTTP endpoints.

    Uses httpx for both sync (complete) and async (stream_complete) requests.
    """

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

    def _build_payload(self, messages: list, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }

    def _build_headers(self, stream: bool) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Sync completion — regular non-streaming request."""
        messages = [{"role": "user", "content": prompt}]
        payload = self._build_payload(messages, stream=False)
        headers = self._build_headers(stream=False)
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(
                f"{self._api_base}/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            resp = r.json()
        try:
            choice = resp["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                text = choice["message"].get("content") or choice["message"].get("content_raw") or ""
            else:
                text = choice.get("text", "")
        except Exception:
            text = ""
        return CompletionResponse(text=text, raw=resp, additional_kwargs={})

    async def _astream_iter(self, prompt: str) -> AsyncIterator[str]:
        """Async generator yielding text chunks from MiniMax SSE stream."""
        messages = [{"role": "user", "content": prompt}]
        payload = self._build_payload(messages, stream=True)
        headers = self._build_headers(stream=True)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._api_base}/chat/completions",
                headers=headers,
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                    elif line.startswith("data:"):
                        data = line[5:]
                    else:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception:
                        continue

    async def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        """Async streaming — returns a sync generator wrapped to satisfy llama-index."""
        async def gen():
            async for text in self._astream_iter(prompt):
                yield CompletionResponse(text=text, raw={}, additional_kwargs={})
        # Return the async generator directly — llama-index's streaming path consumes this as async
        return gen()
