from typing import Any, Generator, Iterator
import requests
import json
from pydantic import PrivateAttr

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    LLMMetadata,
    CompletionResponseGen,
)


class MiniMaxLLM(CustomLLM):
    """MiniMax LLM adapter using OpenAI-compatible HTTP endpoints.

    Supports both regular and streaming chat completions via MiniMax's SSE endpoint.
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

    def _post(self, messages: list, stream: bool = False) -> requests.Response:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"

        r = requests.post(
            f"{self._api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
            stream=stream,
        )
        r.raise_for_status()
        return r

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        r = self._post(messages, stream=False)
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

    def _stream_iter(self, prompt: str) -> Iterator[str]:
        """Yield text chunks from MiniMax's SSE streaming endpoint."""
        messages = [{"role": "user", "content": prompt}]
        r = self._post(messages, stream=True)

        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
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

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> Generator[CompletionResponse, None, None]:
            for text in self._stream_iter(prompt):
                yield CompletionResponse(text=text, raw={}, additional_kwargs={})
        return gen()

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self._complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        return self._stream_complete(prompt, **kwargs)
