"""OpenAI Chat Completions adapter.

Also drives any OpenAI-compatible endpoint (Azure OpenAI, Groq, Together,
Fireworks, DeepSeek, OpenRouter, vLLM, LM Studio, ...) by overriding
``base_url`` and ``model``.

    >>> provider = OpenAIProvider("gpt-4o-mini", api_key="sk-...")
    >>> response = provider.complete("Say hi in three words.")  # doctest: +SKIP
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from ..types import LLMResponse, Message, Role, StreamChunk, ToolCall, Usage, new_id
from .base import BaseProvider

__all__ = ["OpenAICompatibleProvider", "OpenAIProvider"]


class OpenAIProvider(BaseProvider):
    """Chat-completions provider for OpenAI and compatible APIs."""

    name = "openai"
    api_key_env = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", os.environ.get("OPENAI_BASE_URL") or None)
        super().__init__(model or self.default_model, **kwargs)

    # ------------------------------------------------------------------ #
    # Wire format
    # ------------------------------------------------------------------ #
    @classmethod
    def default_base_url(cls) -> str:
        """OpenAI API root."""
        return os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _auth_headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Azure OpenAI and some proxies expect an explicit api-key header too.
        if self.options.get("azure_api_key_header"):
            headers["api-key"] = self.api_key
        org = self.options.get("organization") or os.environ.get("OPENAI_ORG_ID")
        if org:
            headers["OpenAI-Organization"] = str(org)
        project = self.options.get("project") or os.environ.get("OPENAI_PROJECT_ID")
        if project:
            headers["OpenAI-Project"] = str(project)
        return headers

    def _convert_messages(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        """Render messages in the OpenAI wire format."""
        wire: List[Dict[str, Any]] = []
        for message in messages:
            if message.role is Role.TOOL:
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": message.content
                        if isinstance(message.content, str)
                        else str(message.content),
                    }
                )
                continue
            if message.role is Role.ASSISTANT and message.tool_calls:
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments_json},
                        }
                        for call in message.tool_calls
                    ],
                }
                wire.append(entry)
                continue
            entry = {"role": str(message.role), "content": message.content}
            if message.name and message.role is Role.SYSTEM:
                entry["name"] = message.name
            wire.append(entry)
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
        }
        resolved_choice = tool_choice if tool_choice is not None else options.get("tool_choice")
        if tools:
            payload["tools"] = self._as_openai_tools(tools)
            if resolved_choice is not None:
                payload["tool_choice"] = self._wire_tool_choice(resolved_choice)
            if options.get("parallel_tool_calls") is not None:
                payload["parallel_tool_calls"] = bool(options["parallel_tool_calls"])
        for key, value in options.items():
            if key in {"model", "messages", "tools", "tool_choice", "stream", "stream_options"}:
                continue
            if key == "parallel_tool_calls" and not tools:
                continue
            payload[key] = value
        if stream:
            payload["stream"] = True
            if options.get("include_usage", True):
                payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _wire_tool_choice(choice: Any) -> Any:
        """Normalise ``tool_choice`` into OpenAI's expected shape."""
        if isinstance(choice, str):
            return choice
        if isinstance(choice, Mapping):
            return dict(choice)
        name = getattr(choice, "name", None)
        if name:
            return {"type": "function", "function": {"name": name}}
        return str(choice)

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    def _parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        raw_message = choice.get("message") or {}
        calls = [
            ToolCall.from_raw(
                (call.get("function") or {}).get("name", ""),
                (call.get("function") or {}).get("arguments", "") or "",
                id=call.get("id"),
            )
            for call in raw_message.get("tool_calls") or []
        ]
        content = raw_message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, Mapping)
            )
        message = Message.assistant(content or "", tool_calls=calls)
        reasoning = raw_message.get("reasoning_content") or raw_message.get("reasoning")
        if reasoning:
            message.metadata["reasoning"] = reasoning
        return LLMResponse(
            message=message,
            model=str(data.get("model") or self.model),
            usage=self._usage_from(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    @staticmethod
    def _usage_from(usage: Optional[Mapping[str, Any]]) -> Usage:
        """Convert an OpenAI usage payload, including cached-token details."""
        result = Usage.from_dict(dict(usage) if usage else None)
        if usage:
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, Mapping) and details.get("cached_tokens"):
                result.cached_tokens = int(details["cached_tokens"])
        return result

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def _new_stream_state(self) -> Dict[str, Any]:
        return {"text": "", "tool_calls": {}, "usage": Usage(), "finish_reason": None}

    def _parse_stream_data(
        self, data: Mapping[str, Any], state: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        if usage := data.get("usage"):
            state["usage"] = self._usage_from(usage)
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                state["finish_reason"] = choice["finish_reason"]
            text = delta.get("content")
            if text:
                state["text"] += text
                yield StreamChunk(kind="text", text=text)
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                yield StreamChunk(kind="reasoning", text=reasoning)
            for partial in delta.get("tool_calls") or []:
                index = int(partial.get("index", 0) or 0)
                entry = state["tool_calls"].setdefault(
                    index, {"id": partial.get("id") or new_id("call"), "name": "", "args": ""}
                )
                if partial.get("id"):
                    entry["id"] = partial["id"]
                function = partial.get("function") or {}
                if function.get("name"):
                    entry["name"] = function["name"]
                if function.get("arguments"):
                    entry["args"] += function["arguments"]
                yield StreamChunk(
                    kind="tool_call",
                    tool_call=ToolCall.from_raw(
                        entry["name"], entry["args"], id=entry["id"], index=index
                    ),
                )


class OpenAICompatibleProvider(OpenAIProvider):
    """Explicit alias for third-party OpenAI-compatible servers."""

    name = "openai-compatible"
    requires_api_key = False

    def __init__(self, model: str, *, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault(
            "api_key", kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY", "not-needed")
        )
        super().__init__(model, base_url=base_url, **kwargs)
