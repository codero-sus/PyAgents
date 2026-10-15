"""Ollama adapter for locally hosted models.

Ollama needs no API key and speaks newline-delimited JSON rather than SSE, so
this adapter overrides the stream plumbing in :class:`BaseProvider`.

    >>> provider = OllamaProvider("llama3.2")           # doctest: +SKIP
    >>> provider.complete("Hello!").text                # doctest: +SKIP
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from ..types import LLMResponse, Message, Role, StreamChunk, ToolCall, Usage
from .base import BaseProvider

__all__ = ["OllamaProvider"]


class OllamaProvider(BaseProvider):
    """Chat provider for a local (or remote) Ollama server."""

    name = "ollama"
    api_key_env = "OLLAMA_API_KEY"
    requires_api_key = False
    default_model = "llama3.2"

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", self._resolve_host())
        kwargs.setdefault("timeout", 120.0)
        super().__init__(model or self.default_model, **kwargs)

    @staticmethod
    def _resolve_host() -> Optional[str]:
        """Read ``OLLAMA_HOST``, adding a scheme when the user omitted one."""
        host = os.environ.get("OLLAMA_HOST")
        if not host:
            return None
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host

    @classmethod
    def default_base_url(cls) -> str:
        """Local Ollama server root."""
        return cls._resolve_host() or "http://localhost:11434"

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def _auth_headers(self) -> Dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    # ------------------------------------------------------------------ #
    # Wire format
    # ------------------------------------------------------------------ #
    def _convert_messages(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        """Render messages in Ollama's chat format."""
        wire: List[Dict[str, Any]] = []
        for message in messages:
            if message.role is Role.TOOL:
                wire.append({"role": "tool", "content": message.text})
                continue
            if message.role is Role.ASSISTANT and message.tool_calls:
                wire.append(
                    {
                        "role": "assistant",
                        "content": message.text,
                        "tool_calls": [
                            {"function": {"name": call.name, "arguments": call.arguments}}
                            for call in message.tool_calls
                        ],
                    }
                )
                continue
            wire.append({"role": str(message.role), "content": message.content})
        return wire

    def _payload(
        self,
        messages: List[Message],
        *,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any],
        stream: bool,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": bool(stream),
        }
        if tools:
            payload["tools"] = self._as_openai_tools(tools)
        ollama_options: Dict[str, Any] = dict(options.get("options") or {})
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "stop",
            "num_ctx",
            "num_predict",
            "repeat_penalty",
        ):
            if key in options and options[key] is not None:
                ollama_options.setdefault(key, options[key])
        if "max_tokens" in options and options["max_tokens"] is not None:
            ollama_options.setdefault("num_predict", options["max_tokens"])
        if ollama_options:
            payload["options"] = ollama_options
        for key, value in options.items():
            if key in {
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "stop",
                "num_ctx",
                "num_predict",
                "repeat_penalty",
                "max_tokens",
                "options",
                "model",
                "messages",
                "tools",
                "stream",
            }:
                continue
            payload[key] = value
        return payload

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    def _parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        raw_message = data.get("message") or {}
        calls = [
            ToolCall(
                name=str((call.get("function") or {}).get("name", "")),
                arguments=dict((call.get("function") or {}).get("arguments") or {}),
            )
            for call in raw_message.get("tool_calls") or []
        ]
        message = Message.assistant(str(raw_message.get("content") or ""), tool_calls=calls)
        return LLMResponse(
            message=message,
            model=str(data.get("model") or self.model),
            usage=self._usage_from(data),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    @staticmethod
    def _usage_from(data: Mapping[str, Any]) -> Usage:
        """Ollama reports prompt/eval token counts rather than a usage object."""
        prompt = int(data.get("prompt_eval_count", 0) or 0)
        completion = int(data.get("eval_count", 0) or 0)
        return Usage(prompt, completion, prompt + completion)

    # ------------------------------------------------------------------ #
    # Streaming (newline-delimited JSON, not SSE)
    # ------------------------------------------------------------------ #
    def _iter_events(self, payload: Mapping[str, Any]) -> Iterator[Any]:
        lines = self.transport.post_sse(
            self._endpoint(),
            headers=self._headers(),
            payload=payload,
            timeout=self.timeout,
            provider=self.name,
        )
        for line in lines:
            raw = line.strip()
            if raw:
                yield None, raw

    def _new_stream_state(self) -> Dict[str, Any]:
        return {"text": "", "tool_calls": {}, "usage": Usage(), "finish_reason": None, "prompt": 0}

    def _parse_stream_data(
        self, data: Mapping[str, Any], state: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        raw_message = data.get("message") or {}
        text = str(raw_message.get("content") or "")
        if text:
            state["text"] += text
            yield StreamChunk(kind="text", text=text)
        for index, call in enumerate(raw_message.get("tool_calls") or []):
            function = call.get("function") or {}
            state["tool_calls"][index] = {
                "id": call.get("id") or f"call_{index}",
                "name": str(function.get("name", "")),
                "args": function.get("arguments") or {},
            }
            yield StreamChunk(
                kind="tool_call",
                tool_call=ToolCall(
                    name=str(function.get("name", "")),
                    arguments=dict(function.get("arguments") or {}),
                    id=call.get("id") or f"call_{index}",
                    index=index,
                ),
            )
        if data.get("done"):
            state["prompt"] = int(data.get("prompt_eval_count", 0) or 0)
            completion = int(data.get("eval_count", 0) or 0)
            state["usage"] = Usage(state["prompt"], completion, state["prompt"] + completion)
            state["finish_reason"] = data.get("done_reason") or "stop"
            yield StreamChunk(kind="usage", usage=state["usage"])

    def _finalize_stream(self, state: Dict[str, Any]) -> LLMResponse:
        calls: List[ToolCall] = []
        for index, entry in sorted(state["tool_calls"].items()):
            args = entry["args"]
            calls.append(
                ToolCall.from_raw(
                    entry["name"],
                    args if isinstance(args, str) else "",
                    id=entry["id"],
                    index=index,
                )
                if isinstance(args, str)
                else ToolCall(
                    name=entry["name"], arguments=dict(args or {}), id=entry["id"], index=index
                )
            )
        message = Message.assistant(state.get("text", ""), tool_calls=calls)
        return LLMResponse(
            message=message,
            model=self.model,
            usage=state.get("usage") or Usage(),
            finish_reason=state.get("finish_reason"),
        )

    # ------------------------------------------------------------------ #
    # Model management helpers
    # ------------------------------------------------------------------ #
    def list_models(self) -> List[Dict[str, Any]]:
        """Return the models installed on the server (``GET /api/tags``)."""
        import json
        import urllib.request

        request = urllib.request.Request(f"{self.base_url}/api/tags", headers=self._headers())
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return list(payload.get("models") or [])

    def has_model(self, model: Optional[str] = None) -> bool:
        """Return ``True`` when ``model`` (default: this provider's) is available."""
        target = (model or self.model).split(":")[0]
        try:
            return any(
                str(item.get("name", "")).split(":")[0] == target for item in self.list_models()
            )
        except Exception:  # pragma: no cover - server unreachable
            return False
