"""Anthropic Messages API adapter.

Handles the parts of Anthropic's API that differ from OpenAI's:

* the system prompt is a top-level ``system`` field, not a message
* ``max_tokens`` is mandatory
* tools use ``input_schema`` and calls come back as ``tool_use`` content blocks
* tool results are ``tool_result`` blocks inside a *user* message, so
  consecutive tool observations are merged into a single user turn
* streaming emits typed events (``content_block_delta`` & friends)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from ..types import LLMResponse, Message, Role, StreamChunk, ToolCall, Usage, new_id
from .base import BaseProvider

__all__ = ["AnthropicProvider"]

DEFAULT_MAX_TOKENS = 4096
DEFAULT_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Provider for ``api.anthropic.com`` (Claude models)."""

    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    default_model = "claude-3-5-haiku-latest"

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", os.environ.get("ANTHROPIC_BASE_URL") or None)
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        super().__init__(model or self.default_model, **kwargs)

    # ------------------------------------------------------------------ #
    # Wire format
    # ------------------------------------------------------------------ #
    @classmethod
    def default_base_url(cls) -> str:
        """Anthropic API root."""
        return os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"

    def _endpoint(self) -> str:
        return f"{self.base_url}/messages"

    def _auth_headers(self) -> Dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": str(self.options.get("anthropic_version") or DEFAULT_VERSION),
        }
        beta = self.options.get("betas")
        if beta:
            headers["anthropic-beta"] = (
                ",".join(beta)
                if isinstance(beta, Iterable) and not isinstance(beta, str)
                else str(beta)
            )
        return headers

    def _system_prompt(self, messages: Sequence[Message]) -> Optional[List[Dict[str, Any]]]:
        """Collect system messages into Anthropic's ``system`` blocks."""
        blocks: List[Dict[str, Any]] = []
        for message in messages:
            if message.role is Role.SYSTEM and message.content:
                block: Dict[str, Any] = {"type": "text", "text": message.text}
                cache_control = message.metadata.get("cache_control")
                if cache_control:
                    block["cache_control"] = cache_control
                blocks.append(block)
        return blocks or None

    def _convert_messages(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        """Render non-system messages, merging tool results into user turns."""
        wire: List[Dict[str, Any]] = []
        for message in messages:
            if message.role is Role.SYSTEM:
                continue
            if message.role is Role.TOOL:
                block: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.text,
                }
                if message.metadata.get("is_error"):
                    block["is_error"] = True
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
                continue
            if message.role is Role.ASSISTANT:
                blocks: List[Dict[str, Any]] = []
                if message.text:
                    blocks.append({"type": "text", "text": message.text})
                for call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                if message.metadata.get("reasoning"):
                    # Extended thinking requires the signed block if provided.
                    thinking = message.metadata.get("thinking_block")
                    if thinking:
                        blocks.insert(0, thinking)
                wire.append({"role": "assistant", "content": blocks})
                continue
            # user
            if isinstance(message.content, list):
                blocks = []
                for block in message.content:
                    if isinstance(block, Mapping) and block.get("type") == "image_url":
                        image = block.get("image_url")
                        url = (
                            image.get("url", "") if isinstance(image, Mapping) else str(image or "")
                        )
                        parsed = self._image_block(url)
                        if parsed:
                            blocks.append(parsed)
                    elif isinstance(block, Mapping) and block.get("type") == "text":
                        blocks.append({"type": "text", "text": block.get("text", "")})
                    elif isinstance(block, Mapping):
                        blocks.append(dict(block))
                wire.append({"role": "user", "content": blocks})
            else:
                wire.append({"role": "user", "content": message.content})
        return wire

    @staticmethod
    def _image_block(url: str) -> Optional[Dict[str, Any]]:
        """Convert an image URL/data-URI into an Anthropic image block."""
        if not url:
            return None
        if url.startswith("data:"):
            header, _, data = url.partition(",")
            media_type = header[5:].split(";")[0] or "image/png"
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        return {"type": "image", "source": {"type": "url", "url": url}}

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
            "max_tokens": int(options.get("max_tokens") or DEFAULT_MAX_TOKENS),
            "messages": self._convert_messages(messages),
        }
        system = self._system_prompt(messages)
        if system:
            payload["system"] = system
        resolved_choice = tool_choice if tool_choice is not None else options.get("tool_choice")
        if tools:
            payload["tools"] = self._as_anthropic_tools(tools)
            if resolved_choice is not None:
                payload["tool_choice"] = self._wire_tool_choice(resolved_choice)
        for key, value in options.items():
            if key in {
                "max_tokens",
                "model",
                "messages",
                "system",
                "tools",
                "tool_choice",
                "stream",
            }:
                continue
            if key == "temperature" and value is not None and value > 1.0:
                # Anthropic caps temperature at 1.0; clamp instead of failing.
                payload[key] = 1.0
                continue
            payload[key] = value
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _as_anthropic_tools(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Render canonical tool specs in Anthropic's wire format."""
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
            for spec in specs
        ]

    @staticmethod
    def _wire_tool_choice(choice: Any) -> Optional[Dict[str, Any]]:
        """Normalise ``tool_choice`` into Anthropic's shape."""
        if isinstance(choice, Mapping):
            data = dict(choice)
            if "function" in data and isinstance(data["function"], Mapping):
                return {"type": "tool", "name": data["function"].get("name", "")}
            return data
        if isinstance(choice, str):
            if choice in {"auto", "any", "none"}:
                return {"type": choice}
            return {"type": "tool", "name": choice}
        name = getattr(choice, "name", None)
        if name:
            return {"type": "tool", "name": name}
        return None

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    def _parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        blocks = data.get("content") or []
        text_parts: List[str] = []
        calls: List[ToolCall] = []
        thinking: List[Dict[str, Any]] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                calls.append(
                    ToolCall(
                        name=str(block.get("name", "")),
                        arguments=dict(block.get("input") or {}),
                        id=str(block.get("id") or new_id("call")),
                        raw_arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                    )
                )
            elif block_type in {"thinking", "redacted_thinking"}:
                thinking.append(dict(block))
        message = Message.assistant("\n".join(text_parts), tool_calls=calls)
        if thinking:
            message.metadata["thinking_block"] = thinking[0]
        return LLMResponse(
            message=message,
            model=str(data.get("model") or self.model),
            usage=self._usage_from(data.get("usage")),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    @staticmethod
    def _usage_from(usage: Optional[Mapping[str, Any]]) -> Usage:
        """Convert an Anthropic usage payload."""
        if not usage:
            return Usage()
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        cached = int(usage.get("cache_read_input_tokens", 0) or 0)
        return Usage(prompt, completion, prompt + completion, cached)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def _new_stream_state(self) -> Dict[str, Any]:
        return {
            "text": "",
            "tool_calls": {},
            "usage": Usage(),
            "finish_reason": None,
            "current_index": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking": [],
        }

    def _parse_stream_data(
        self, data: Mapping[str, Any], state: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        event = str(data.get("type") or "")
        if event == "error":  # pragma: no cover - provider-side error frame
            error = data.get("error") or {}
            raise self._stream_error(str(error.get("message") or "stream error"))
        if event == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            state["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
            state["usage"] = Usage(state["input_tokens"], 0, state["input_tokens"])
            return
        if event == "content_block_start":
            index = int(data.get("index", 0))
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                state["current_index"] = index
                state["tool_calls"][index] = {
                    "id": block.get("id") or new_id("call"),
                    "name": block.get("name", ""),
                    "args": "",
                }
            return
        if event == "content_block_delta":
            index = int(data.get("index", state.get("current_index") or 0))
            delta = data.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = str(delta.get("text", ""))
                state["text"] += text
                yield StreamChunk(kind="text", text=text)
            elif delta_type == "input_json_delta":
                entry = state["tool_calls"].setdefault(
                    index, {"id": new_id("call"), "name": "", "args": ""}
                )
                entry["args"] += str(delta.get("partial_json", ""))
            elif delta_type == "thinking_delta":
                state["thinking"].append(str(delta.get("thinking", "")))
            return
        if event == "content_block_stop":
            index = int(data.get("index", 0))
            entry = state["tool_calls"].get(index)
            if entry:
                yield StreamChunk(
                    kind="tool_call",
                    tool_call=ToolCall.from_raw(
                        entry["name"], entry["args"], id=entry["id"], index=index
                    ),
                )
            return
        if event == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason"):
                state["finish_reason"] = delta["stop_reason"]
            usage = data.get("usage") or {}
            if usage.get("output_tokens") is not None:
                state["output_tokens"] = int(usage["output_tokens"])
            prompt = state.get("input_tokens", 0)
            state["usage"] = Usage(prompt, state["output_tokens"], prompt + state["output_tokens"])
            yield StreamChunk(kind="usage", usage=state["usage"])
            return

    def _finalize_stream(self, state: Dict[str, Any]) -> LLMResponse:
        calls = [
            ToolCall.from_raw(entry["name"], entry["args"], id=entry["id"], index=index)
            for index, entry in sorted(state["tool_calls"].items())
        ]
        message = Message.assistant(state.get("text", ""), tool_calls=calls)
        if state.get("thinking"):
            message.metadata["reasoning"] = "".join(state["thinking"])
        return LLMResponse(
            message=message,
            model=self.model,
            usage=state.get("usage") or Usage(),
            finish_reason=state.get("finish_reason"),
        )

    def _stream_error(self, message: str) -> Exception:
        """Build the right error type for a mid-stream error frame."""
        from ..errors import ProviderError

        return ProviderError(message, provider=self.name)
