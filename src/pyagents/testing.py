"""Test doubles for agent applications.

These helpers make it possible to unit-test an agent, team or workflow without
touching the network or spending tokens::

    from pyagents import Agent
    from pyagents.testing import ScriptedProvider, tool_response, text_response

    provider = ScriptedProvider([
        tool_response("calculator", {"expression": "2 + 2"}),
        text_response("The answer is 4."),
    ])
    agent = Agent("math", provider=provider, tools=[calculator])
    result = agent.run("What is 2 + 2?")
    assert result.output == "The answer is 4."

The fakes record everything they see, so assertions about prompts and tool
calls are easy to write.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from .errors import ProviderError
from .llm.base import BaseProvider
from .llm.echo import EchoProvider
from .llm.transport import HttpTransport
from .types import LLMResponse, Message, StreamChunk, ToolCall, Usage, new_id

__all__ = [
    "EchoProvider",
    "FakeTransport",
    "RecordingProvider",
    "ScriptedProvider",
    "response",
    "text_response",
    "tool_response",
]


class FakeTransport(HttpTransport):
    """An :class:`~pyagents.llm.transport.HttpTransport` that never hits the network.

    Args:
        responses: Queued JSON bodies for :meth:`post_json`. A callable is
            invoked with the request payload; an :class:`Exception` instance is
            raised; a list means "use these instead of the queue".
        sse: Queued line lists for :meth:`post_sse`.
        record: When true (default) every request is appended to
            :attr:`requests`.
    """

    name = "fake"

    def __init__(
        self,
        responses: Optional[Iterable[Any]] = None,
        *,
        sse: Optional[Iterable[Iterable[str]]] = None,
        record: bool = True,
    ) -> None:
        self.responses: Deque[Any] = deque(responses or [])
        self.sse_responses: Deque[Any] = deque(sse or [])
        self.record = record
        self.requests: List[Dict[str, Any]] = []

    # ---- helpers --------------------------------------------------------- #
    @property
    def call_count(self) -> int:
        """Number of HTTP calls made through this transport."""
        return len(self.requests)

    @property
    def last_request(self) -> Dict[str, Any]:
        """The most recent request, or an empty mapping."""
        return self.requests[-1] if self.requests else {}

    @property
    def last_payload(self) -> Dict[str, Any]:
        """JSON body of the most recent request."""
        return dict(self.last_request.get("payload") or {})

    def queue(self, *responses: Any) -> FakeTransport:
        """Append responses to the queue and return ``self``."""
        self.responses.extend(responses)
        return self

    def queue_sse(self, lines: Iterable[str]) -> FakeTransport:
        """Append a streamed SSE payload and return ``self``."""
        self.sse_responses.append(list(lines))
        return self

    # ---- transport API --------------------------------------------------- #
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Dict[str, Any]:
        """Return the next queued response."""
        if self.record:
            self.requests.append(
                {
                    "url": url,
                    "headers": dict(headers),
                    "payload": dict(payload),
                    "provider": provider,
                }
            )
        if not self.responses:
            raise ProviderError("FakeTransport ran out of queued responses", provider=provider)
        item = self.responses.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(payload)
        if isinstance(item, (str, bytes)):
            return json.loads(item)
        return dict(item)

    def post_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Iterator[str]:
        """Yield the next queued SSE line list."""
        if self.record:
            self.requests.append(
                {
                    "url": url,
                    "headers": dict(headers),
                    "payload": dict(payload),
                    "provider": provider,
                }
            )
        if not self.sse_responses:
            raise ProviderError("FakeTransport ran out of queued SSE responses", provider=provider)
        item = self.sse_responses.popleft()
        if isinstance(item, BaseException):  # pragma: no cover - error path
            raise item
        for line in item:
            if isinstance(line, BaseException):
                # Models a connection dropping in the middle of a stream.
                raise line
            yield line

    def reset(self) -> None:
        """Clear queues and recorded requests."""
        self.responses.clear()
        self.sse_responses.clear()
        self.requests.clear()


def response(
    text: str = "",
    *,
    tool_calls: Optional[Sequence[Any]] = None,
    usage: Optional[Usage] = None,
    model: str = "scripted",
    finish_reason: str = "stop",
) -> LLMResponse:
    """Build an :class:`LLMResponse` from text and/or tool calls.

    Tool calls may be :class:`ToolCall` objects or ``(name, arguments)`` pairs.
    """
    calls: List[ToolCall] = []
    for call in tool_calls or []:
        if isinstance(call, ToolCall):
            calls.append(call)
        elif isinstance(call, Mapping):
            calls.append(
                ToolCall(
                    name=str(call.get("name", "")),
                    arguments=dict(call.get("arguments") or {}),
                    id=str(call.get("id") or new_id("call")),
                )
            )
        else:
            name, arguments = call
            calls.append(ToolCall(name=name, arguments=dict(arguments or {})))
    return LLMResponse(
        message=Message.assistant(text, tool_calls=calls),
        model=model,
        usage=usage or Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason="tool_calls" if calls else finish_reason,
    )


def text_response(text: str, **kwargs: Any) -> LLMResponse:
    """Convenience wrapper for a plain text response."""
    return response(text, **kwargs)


def tool_response(
    name: str, arguments: Optional[Mapping[str, Any]] = None, **kwargs: Any
) -> LLMResponse:
    """Convenience wrapper for a tool-call response."""
    return response("", tool_calls=[(name, dict(arguments or {}))], **kwargs)


class ScriptedProvider(BaseProvider):
    """A provider that replays a fixed script of responses.

    Args:
        responses: Anything accepted by :func:`response`, a callable
            ``(messages) -> LLMResponse | str``, or a mix. The list is consumed
            in order; when it is exhausted, ``default_text`` is returned.
        default_text: Fallback answer once the script runs out.
        stream_delay: Artificial per-chunk delay, to exercise streaming paths.
    """

    name = "scripted"
    requires_api_key = False
    default_model = "scripted-model"

    def __init__(
        self,
        responses: Optional[Iterable[Any]] = None,
        *,
        model: str = "scripted-model",
        default_text: str = "Done.",
        stream_delay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("api_key", None)
        kwargs.setdefault("base_url", "http://scripted.local/v1")
        super().__init__(model or self.default_model, api_key="scripted", **kwargs)
        self.script: Deque[Any] = deque(responses or [])
        self.default_text = default_text
        self.stream_delay = stream_delay
        #: Every message list passed to a call, in order.
        self.calls: List[List[Message]] = []
        #: Tool specs seen on each call.
        self.tools_seen: List[List[Dict[str, Any]]] = []

    @classmethod
    def default_base_url(cls) -> str:
        return "http://scripted.local/v1"

    # ---- plumbing -------------------------------------------------------- #
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer scripted"}

    def _payload(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        return {"model": self.model, "messages": [message.to_dict() for message in messages]}

    def _parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        return response(str(data.get("content", "")))

    def pop(self, messages: Sequence[Message]) -> LLMResponse:
        """Return the next scripted response for ``messages``."""
        if not self.script:
            return response(self.default_text)
        item = self.script.popleft()
        if callable(item):
            item = item(list(messages))
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, Message):
            return LLMResponse(message=item, model=self.model)
        if isinstance(item, str):
            return response(item)
        if isinstance(item, Mapping):
            return response(
                str(item.get("content", item.get("text", ""))),
                tool_calls=item.get("tool_calls"),
            )
        raise TypeError(f"Cannot interpret scripted response of type {type(item)!r}")

    # ---- overrides (no HTTP involved) ------------------------------------ #
    def complete(
        self, messages: Iterable[Any], *, tools: Any = None, tool_choice: Any = None, **options: Any
    ) -> LLMResponse:
        """Return the next scripted response."""
        from .llm.base import normalize_tool_specs, resolve_messages

        merged = resolve_messages(messages)
        self.calls.append(merged)
        self.tools_seen.append(normalize_tool_specs(tools))
        return self.pop(merged)

    def stream(
        self, messages: Iterable[Any], *, tools: Any = None, tool_choice: Any = None, **options: Any
    ) -> Iterator[StreamChunk]:
        """Stream the next scripted response chunk by chunk."""
        result = self.complete(messages, tools=tools, tool_choice=tool_choice, **options)
        yield from self._as_chunks(result)

    async def acomplete(
        self, messages: Iterable[Any], *, tools: Any = None, tool_choice: Any = None, **options: Any
    ) -> LLMResponse:
        """Async variant of :meth:`complete`."""
        await asyncio.sleep(0)
        return self.complete(messages, tools=tools, tool_choice=tool_choice, **options)

    async def astream(
        self, messages: Iterable[Any], *, tools: Any = None, tool_choice: Any = None, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        """Async variant of :meth:`stream`."""
        result = await self.acomplete(messages, tools=tools, tool_choice=tool_choice, **options)
        for chunk in self._as_chunks(result):
            if self.stream_delay:
                await asyncio.sleep(self.stream_delay)
            yield chunk

    def _as_chunks(self, result: LLMResponse) -> Iterator[StreamChunk]:
        """Split a complete response into word-sized stream chunks."""
        text = result.message.text
        for token in _tokens(text):
            yield StreamChunk(kind="text", text=token)
        for call in result.message.tool_calls:
            yield StreamChunk(kind="tool_call", tool_call=call)
        if result.usage:
            yield StreamChunk(kind="usage", usage=result.usage)
        yield StreamChunk(kind="response", response=result)

    # ---- assertions helpers ---------------------------------------------- #
    def last_messages(self) -> List[Message]:
        """Messages passed on the most recent call."""
        return self.calls[-1] if self.calls else []

    def last_user_message(self) -> str:
        """Text of the last user message seen."""
        for message in reversed(self.last_messages()):
            if message.role.value == "user":
                return message.text
        return ""

    def tool_call_count(self) -> int:
        """Number of tool calls requested across all scripted responses."""
        return sum(len(item.tool_calls) for item in self.calls_seen_responses())

    def calls_seen_responses(self) -> List[LLMResponse]:
        """Responses already returned (requires the script to be drained)."""
        return list(getattr(self, "_returned", []))

    def reset(self) -> None:
        """Clear recorded calls."""
        self.calls.clear()
        self.tools_seen.clear()


class RecordingProvider(ScriptedProvider):
    """Alias of :class:`ScriptedProvider` emphasising its recording behaviour."""


def _tokens(text: str) -> List[str]:
    """Split text into small chunks that look like streamed tokens."""
    if not text:
        return []
    parts = text.split(" ")
    return [part + " " for part in parts[:-1]] + [parts[-1]] if len(parts) > 1 else [text]


class StreamingClock:
    """Tiny helper for asserting on timing-sensitive streaming tests."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def tick(self) -> float:
        """Seconds elapsed since creation."""
        return time.perf_counter() - self.start


def cycle(provider: Callable[[int], Any]) -> Callable[[List[Message]], Any]:
    """Build a scripted responder that varies by call index.

    >>> responder = cycle(lambda i: f"call {i}")
    >>> responder([])
    'call 0'
    """
    counter = itertools.count()
    return lambda messages: provider(next(counter))  # noqa: ARG005
