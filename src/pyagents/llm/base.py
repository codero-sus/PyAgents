"""Provider base class: request building, retries, streaming and async glue.

Concrete adapters only implement four small hooks:

* :meth:`BaseProvider._endpoint` - where to POST
* :meth:`BaseProvider._auth_headers` - authentication headers
* :meth:`BaseProvider._payload` - request body for a chat completion
* :meth:`BaseProvider._parse_response` / :meth:`BaseProvider._parse_stream_data`

Everything else - retry/backoff, streaming aggregation, threading the transport,
async variants - lives here so all providers behave identically.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

from ..errors import ConfigurationError, ProviderError, ProviderRateLimitError, ProviderTimeoutError
from ..types import LLMResponse, Message, Role, StreamChunk, ToolCall, Usage
from ..utils import estimate_message_tokens, json_dumps
from .transport import HttpTransport, get_default_transport

__all__ = [
    "BaseProvider",
    "LLMProvider",
    "normalize_tool_spec",
    "normalize_tool_specs",
    "resolve_messages",
]

T = TypeVar("T")

#: Kinds of failures that are worth another attempt.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def normalize_tool_spec(tool: Any) -> Dict[str, Any]:
    """Normalise any accepted tool description into a canonical mapping.

    Accepts:

    * a :class:`pyagents.tools.Tool` style object exposing ``name``,
      ``description`` and ``parameters``,
    * an OpenAI-style ``{"type": "function", "function": {...}}`` mapping,
    * a plain ``{"name": ..., "parameters": ...}`` mapping.

    Returns:
        ``{"name": str, "description": str, "parameters": {...}}``

    Raises:
        ConfigurationError: if the tool has no usable name.
    """
    if hasattr(tool, "to_spec") and callable(tool.to_spec):
        spec = tool.to_spec()
        return normalize_tool_spec(spec)
    if isinstance(tool, Mapping):
        data: Mapping[str, Any] = tool
        if isinstance(data.get("function"), Mapping):
            data = data["function"]
        name = data.get("name")
        if not name:
            raise ConfigurationError("Tool definition is missing a 'name'", tool=repr(tool)[:120])
        parameters = (
            data.get("parameters")
            or data.get("input_schema")
            or {"type": "object", "properties": {}}
        )
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(
                "Tool 'parameters' must be a JSON schema object", tool=str(name)
            )
        return {
            "name": str(name),
            "description": str(data.get("description") or ""),
            "parameters": dict(parameters),
        }
    name = getattr(tool, "name", None)
    if not name:
        raise ConfigurationError(f"Cannot interpret {type(tool)!r} as a tool definition")
    return {
        "name": str(name),
        "description": str(getattr(tool, "description", "") or ""),
        "parameters": dict(
            getattr(tool, "parameters", None) or {"type": "object", "properties": {}}
        ),
    }


def normalize_tool_specs(tools: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """Normalise a collection of tools into canonical specs."""
    if not tools:
        return []
    return [normalize_tool_spec(tool) for tool in tools]


def resolve_messages(messages: Iterable[Any]) -> List[Message]:
    """Coerce a mixed iterable of strings/mappings/messages into messages."""
    return [Message.coerce(item) for item in messages]


class BaseProvider(ABC):
    """Abstract chat-completion provider.

    Args:
        model: Model identifier passed to the vendor API.
        api_key: Credential; falls back to ``api_key_env``.
        base_url: Override the API root (proxies, Azure, self-hosted servers).
        timeout: Per-request timeout in seconds.
        max_retries: Extra attempts for retryable failures (429/5xx/timeouts).
        transport: Custom :class:`~pyagents.llm.transport.HttpTransport`.
        temperature: Default sampling temperature for every call.
        max_tokens: Default completion token budget for every call.
        headers: Extra HTTP headers merged into every request.
        options: Provider-specific defaults merged into every payload.
    """

    #: Registry name, e.g. ``"openai"``.
    name: str = "base"
    #: Environment variable consulted when ``api_key`` is not passed.
    api_key_env: Optional[str] = None
    #: Whether the provider requires an API key at all.
    requires_api_key: bool = True
    #: Options consumed by PyAgents itself; never sent to the vendor API.
    CONTROL_OPTIONS = frozenset(
        {
            "retry_base_delay",
            "azure_api_key_header",
            "organization",
            "project",
            "anthropic_version",
            "betas",
            "include_usage",
        }
    )

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: Optional[HttpTransport] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        headers: Optional[Mapping[str, str]] = None,
        default_options: Optional[Mapping[str, Any]] = None,
        **options: Any,
    ) -> None:
        if not model:
            raise ConfigurationError(f"{type(self).__name__} requires a model name")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers: Dict[str, str] = dict(headers or {})
        self.options: Dict[str, Any] = dict(default_options or {})
        self.options.update(options)
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self.api_key = (
            api_key if api_key is not None else (os.environ.get(self.api_key_env or "") or "")
        )
        if self.requires_api_key and not self.api_key:
            raise ConfigurationError(
                f"{self.name} provider needs an API key. Pass api_key=... or set ${self.api_key_env}.",
                provider=self.name,
            )
        self._transport = transport
        self._owns_transport = transport is None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    @classmethod
    def default_base_url(cls) -> str:
        """Default API root for the provider."""
        raise NotImplementedError

    @abstractmethod
    def _endpoint(self) -> str:
        """Full URL of the chat-completions endpoint."""

    @abstractmethod
    def _auth_headers(self) -> Dict[str, str]:
        """Authentication headers for the provider."""

    @abstractmethod
    def _payload(
        self,
        messages: List[Message],
        *,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any],
        stream: bool,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build the request body."""

    @abstractmethod
    def _parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        """Convert a non-streaming response body into an :class:`LLMResponse`."""

    def _new_stream_state(self) -> Dict[str, Any]:
        """Create the mutable accumulator used while streaming."""
        return {"text": "", "tool_calls": {}, "usage": Usage(), "finish_reason": None}

    def _parse_stream_data(
        self, data: Mapping[str, Any], state: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        """Consume one decoded SSE payload, yielding incremental chunks."""
        raise NotImplementedError(f"{type(self).__name__} does not implement streaming")

    def _finalize_stream(self, state: Dict[str, Any]) -> LLMResponse:
        """Assemble the accumulated stream state into a full response.

        Stream state holds tool calls as either :class:`ToolCall` instances or
        ``{"id", "name", "args"|"arguments"}`` dictionaries; both are accepted.
        """
        calls: List[ToolCall] = []
        for index, value in sorted(state.get("tool_calls", {}).items()):
            if isinstance(value, ToolCall):
                value.index = index
                calls.append(value)
            elif isinstance(value, Mapping) and "args" in value:
                calls.append(
                    ToolCall.from_raw(
                        value.get("name", ""),
                        value.get("args") or "",
                        id=value.get("id"),
                        index=index,
                    )
                )
            else:
                calls.append(ToolCall(**value, index=index))
        message = Message.assistant(state.get("text", ""), tool_calls=calls)
        return LLMResponse(
            message=message,
            model=self.model,
            usage=state.get("usage") or Usage(),
            finish_reason=state.get("finish_reason"),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def transport(self) -> HttpTransport:
        """Transport used for HTTP calls (lazily resolved)."""
        if self._transport is None:
            self._transport = get_default_transport()
        return self._transport

    def complete(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[Any] = None,
        **options: Any,
    ) -> LLMResponse:
        """Run a chat completion and return the full response."""
        return self._call_with_retries(
            lambda: self._complete_once(
                messages, tools=tools, tool_choice=tool_choice, options=options
            )
        )

    def stream(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[Any] = None,
        **options: Any,
    ) -> Iterator[StreamChunk]:
        """Run a streaming chat completion, yielding :class:`StreamChunk` items.

        The last chunk always has ``kind == "response"`` and carries the
        assembled :class:`LLMResponse`.
        """
        merged = list(resolve_messages(messages))
        specs = normalize_tool_specs(tools)
        payload_options = self._merge_options(options)
        payload = self._payload(
            merged, tools=specs, tool_choice=tool_choice, stream=True, options=payload_options
        )
        state = self._new_stream_state()
        attempt = 0
        while True:
            emitted = False
            try:
                for _event, raw in self._iter_events(payload):
                    if raw == "[DONE]":
                        break
                    parsed = self._decode_event(raw)
                    if parsed is None:
                        continue
                    for chunk in self._parse_stream_data(parsed, state):
                        emitted = True
                        yield chunk
                yield StreamChunk(kind="response", response=self._finalize_stream(state))
                return
            except (ProviderRateLimitError, ProviderTimeoutError, ProviderError) as exc:
                if emitted or not self._should_retry(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                time.sleep(self._retry_delay(attempt, exc))

    async def acomplete(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[Any] = None,
        **options: Any,
    ) -> LLMResponse:
        """Async variant of :meth:`complete`."""
        merged = list(resolve_messages(messages))
        specs = normalize_tool_specs(tools)
        payload_options = self._merge_options(options)
        attempt = 0
        while True:
            try:
                payload = self._payload(
                    merged,
                    tools=specs,
                    tool_choice=tool_choice,
                    stream=False,
                    options=payload_options,
                )
                data = await self.transport.apost_json(
                    self._endpoint(),
                    headers=self._headers(),
                    payload=payload,
                    timeout=self.timeout,
                    provider=self.name,
                )
                return self._parse_response(data)
            except ProviderError as exc:
                if not self._should_retry(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                await asyncio.sleep(self._retry_delay(attempt, exc))

    async def astream(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[Any] = None,
        **options: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async variant of :meth:`stream`."""
        merged = list(resolve_messages(messages))
        specs = normalize_tool_specs(tools)
        payload_options = self._merge_options(options)
        attempt = 0
        while True:
            emitted = False
            try:
                payload = self._payload(
                    merged,
                    tools=specs,
                    tool_choice=tool_choice,
                    stream=True,
                    options=payload_options,
                )
                state = self._new_stream_state()
                lines = self.transport.apost_sse(
                    self._endpoint(),
                    headers=self._headers(),
                    payload=payload,
                    timeout=self.timeout,
                    provider=self.name,
                )
                async for line in lines:
                    if not line.strip():
                        continue
                    if line.startswith("event:"):
                        continue
                    raw = line[5:].strip() if line.startswith("data:") else line.strip()
                    if raw == "[DONE]":
                        break
                    parsed = self._decode_event(raw)
                    if parsed is None:
                        continue
                    for chunk in self._parse_stream_data(parsed, state):
                        emitted = True
                        yield chunk
                yield StreamChunk(kind="response", response=self._finalize_stream(state))
                return
            except (ProviderRateLimitError, ProviderTimeoutError, ProviderError) as exc:
                if emitted or not self._should_retry(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                await asyncio.sleep(self._retry_delay(attempt, exc))

    def count_tokens(self, messages: Iterable[Any], tools: Optional[Sequence[Any]] = None) -> int:
        """Estimate the prompt token count for ``messages``."""
        total = estimate_message_tokens(resolve_messages(messages))
        if tools:
            total += estimate_message_tokens([]) + sum(
                len(json_dumps(spec)) // 4 + 8 for spec in normalize_tool_specs(tools)
            )
        return total

    def close(self) -> None:
        """Close owned resources (transport) if this provider created them."""
        if self._owns_transport and self._transport is not None:
            self._transport.close()

    def __enter__(self) -> BaseProvider:  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info: Any) -> None:  # pragma: no cover - convenience
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _complete_once(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]],
        tool_choice: Optional[Any],
        options: Mapping[str, Any],
    ) -> LLMResponse:
        merged = list(resolve_messages(messages))
        specs = normalize_tool_specs(tools)
        payload = self._payload(
            merged,
            tools=specs,
            tool_choice=tool_choice,
            stream=False,
            options=self._merge_options(options),
        )
        data = self.transport.post_json(
            self._endpoint(),
            headers=self._headers(),
            payload=payload,
            timeout=self.timeout,
            provider=self.name,
        )
        return self._parse_response(data)

    def _iter_events(self, payload: Mapping[str, Any]) -> Iterator[Any]:
        """Yield ``(event, data)`` SSE pairs for a blocking stream request."""
        from .transport import iter_sse

        lines = self.transport.post_sse(
            self._endpoint(),
            headers=self._headers(),
            payload=payload,
            timeout=self.timeout,
            provider=self.name,
        )
        return iter_sse(lines)

    def _decode_event(self, raw: str) -> Optional[Mapping[str, Any]]:
        """Decode an SSE data payload, tolerating keep-alives."""
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, Mapping):
            return data
        return {"data": data}

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "pyagents/0.1"}
        headers.update(self._auth_headers())
        headers.update(self.headers)
        return headers

    def _merge_options(self, options: Mapping[str, Any]) -> Dict[str, Any]:
        merged = dict(self.options)
        merged.update({key: value for key, value in options.items() if value is not None})
        if self.temperature is not None and "temperature" not in options:
            merged.setdefault("temperature", self.temperature)
        if self.max_tokens is not None:
            merged.setdefault("max_tokens", self.max_tokens)
        for key in self.CONTROL_OPTIONS:
            merged.pop(key, None)
        return merged

    def _should_retry(self, exc: BaseException) -> bool:
        """Decide whether ``exc`` is worth retrying."""
        if isinstance(exc, ProviderRateLimitError):
            return True
        if isinstance(exc, ProviderTimeoutError):
            return True
        if isinstance(exc, ProviderError):
            status = exc.status_code
            return bool(status is not None and status in _RETRYABLE_STATUS)
        return False

    def _retry_delay(self, attempt: int, exc: Optional[BaseException] = None) -> float:
        """Exponential backoff with jitter, honouring ``Retry-After`` hints."""
        if isinstance(exc, ProviderRateLimitError) and exc.retry_after:
            return float(exc.retry_after)
        base = float(self.options.get("retry_base_delay", 0.5))
        delay = min(8.0, base * (2 ** (attempt - 1)))
        return delay + random.uniform(0, base / 2)

    def _call_with_retries(self, call: Callable[[], T]) -> T:
        """Run ``call`` with retry/backoff semantics for transient errors."""
        attempt = 0
        while True:
            try:
                return call()
            except ProviderError as exc:
                if not self._should_retry(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                time.sleep(self._retry_delay(attempt, exc))

    # ---- shared helpers for adapters ---------------------------------- #
    @staticmethod
    def _text_of(message: Message) -> str:
        """Return the plain text of a message."""
        return message.text

    @staticmethod
    def _system_messages(messages: Sequence[Message]) -> List[Message]:
        """Return only the system messages."""
        return [message for message in messages if message.role is Role.SYSTEM]

    @staticmethod
    def _as_openai_tools(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Render canonical tool specs in OpenAI's ``tools`` wire format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for spec in specs
        ]


#: Public alias - ``LLMProvider`` reads better in type hints.
LLMProvider = BaseProvider
