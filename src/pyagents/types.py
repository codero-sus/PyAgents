"""Core data structures shared across PyAgents.

Everything the framework passes around - messages, tool calls, usage counters,
run results - is defined here as a plain :mod:`dataclasses` object with
``to_dict``/``from_dict`` helpers so it can be serialised to JSON, SQLite or
the wire formats expected by LLM providers.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

__all__ = [
    "AgentEvent",
    "EventType",
    "Handoff",
    "LLMResponse",
    "Message",
    "MessageLike",
    "Role",
    "RunResult",
    "RunStatus",
    "Step",
    "StreamChunk",
    "ToolCall",
    "ToolResult",
    "Usage",
    "new_id",
    "utc_now",
]


def utc_now() -> float:
    """Return the current UNIX timestamp as a float."""
    return time.time()


def new_id(prefix: str = "") -> str:
    """Return a short, unique, sortable-enough identifier.

    >>> new_id("run").startswith("run_")
    True
    """
    token = uuid.uuid4().hex[:12]
    return f"{prefix}_{token}" if prefix else token


class Role(str, Enum):
    """Chat roles understood by every supported provider."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


class RunStatus(str, Enum):
    """Lifecycle state of an agent or team run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS = "max_steps"
    CANCELLED = "cancelled"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value

    @property
    def is_terminal(self) -> bool:
        """``True`` when no further work will happen without an explicit resume."""
        return self in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.MAX_STEPS,
            RunStatus.CANCELLED,
        }


class EventType(str, Enum):
    """Kinds of :class:`AgentEvent` emitted while a run executes.

    Streams are the observability primitive of PyAgents: the CLI prints them,
    the test-suite asserts on them, and checkpointing is driven by them.
    """

    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP_START = "step_start"
    TEXT = "text"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HANDOFF = "handoff"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    CHECKPOINT = "checkpoint"
    ERROR = "error"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


# --------------------------------------------------------------------------- #
# Tool calling
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A model's request to invoke a tool.

    ``arguments`` is always a parsed mapping. Models sometimes emit malformed
    JSON; :meth:`from_raw` keeps the raw string in ``raw_arguments`` so callers
    can decide how strict to be.
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))
    raw_arguments: str = ""
    index: Optional[int] = None

    @classmethod
    def from_raw(
        cls,
        name: str,
        raw_arguments: str,
        *,
        id: Optional[str] = None,
        index: Optional[int] = None,
    ) -> ToolCall:
        """Build a tool call from a raw JSON string, tolerating bad JSON."""
        raw = raw_arguments or ""
        parsed: Dict[str, Any]
        if not raw.strip():
            parsed = {}
        else:
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                parsed = loaded
            elif loaded is None:
                parsed = {}
                raw = raw  # keep the original text for the error observation
            else:
                parsed = {"value": loaded}
        call = cls(name=name, arguments=parsed, raw_arguments=raw, index=index)
        if id:
            call.id = id
        return call

    @property
    def arguments_json(self) -> str:
        """Arguments as a JSON string, preferring the original model output.

        Providers expect the raw JSON text the model produced; when a tool call
        was built programmatically (or the model sent malformed JSON) this
        serialises the parsed mapping instead so the field is never empty.
        """
        if self.raw_arguments.strip():
            return self.raw_arguments
        return (
            json.dumps(self.arguments, ensure_ascii=False, default=str) if self.arguments else "{}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data: Dict[str, Any] = {"id": self.id, "name": self.name, "arguments": self.arguments}
        if self.raw_arguments:
            data["raw_arguments"] = self.raw_arguments
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCall:
        """Rebuild a tool call from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            arguments=dict(data.get("arguments") or {}),
            id=data.get("id") or new_id("call"),
            raw_arguments=data.get("raw_arguments", ""),
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.name}({json.dumps(self.arguments, ensure_ascii=False, default=str)})"


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
MessageLike = Union["Message", str, Dict[str, Any]]


@dataclass
class Message:
    """A single chat message.

    ``content`` is a plain string for text messages. For multimodal input it
    may be a list of provider-native content blocks, which PyAgents passes
    through untouched. :meth:`user` offers a portable shortcut for images.
    """

    role: Role
    content: Any = ""
    name: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=utc_now)

    # ---- constructors ---------------------------------------------------- #
    @classmethod
    def system(cls, content: str, **metadata: Any) -> Message:
        """Create a system message."""
        return cls(Role.SYSTEM, content, metadata=metadata)

    @classmethod
    def user(
        cls, content: Any, *, images: Optional[Sequence[str]] = None, **metadata: Any
    ) -> Message:
        """Create a user message, optionally with images.

        Args:
            content: Text of the message.
            images: Image URLs or ``data:`` URIs attached to the message.
        """
        if images:
            blocks: List[Dict[str, Any]] = [{"type": "text", "text": content}]
            blocks.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
            return cls(Role.USER, blocks, metadata=metadata)
        return cls(Role.USER, content, metadata=metadata)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        *,
        tool_calls: Optional[Sequence[ToolCall]] = None,
        **metadata: Any,
    ) -> Message:
        """Create an assistant message, optionally requesting tool calls."""
        return cls(Role.ASSISTANT, content, tool_calls=list(tool_calls or []), metadata=metadata)

    @classmethod
    def tool_result(
        cls,
        content: Any,
        *,
        tool_call_id: str,
        name: Optional[str] = None,
        is_error: bool = False,
        **metadata: Any,
    ) -> Message:
        """Create a tool-observation message."""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        metadata["is_error"] = is_error
        return cls(Role.TOOL, content, name=name, tool_call_id=tool_call_id, metadata=metadata)

    @classmethod
    def coerce(cls, value: MessageLike, *, role: Role = Role.USER) -> Message:
        """Convert a string or mapping into a :class:`Message`."""
        if isinstance(value, Message):
            return value
        if isinstance(value, str):
            return cls(role, value)
        if isinstance(value, dict):
            return cls.from_dict(value)
        raise TypeError(f"Cannot convert {type(value)!r} into a Message")

    # ---- helpers --------------------------------------------------------- #
    @property
    def text(self) -> str:
        """Best-effort plain-text view of ``content``."""
        if isinstance(self.content, str):
            return self.content
        parts: List[str] = []
        for block in self.content or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)

    @property
    def is_tool_call_request(self) -> bool:
        """``True`` when the assistant asked for at least one tool call."""
        return bool(self.tool_calls)

    def with_metadata(self, **values: Any) -> Message:
        """Return a copy with extra metadata merged in."""
        clone = self.copy()
        clone.metadata.update(values)
        return clone

    def copy(self) -> Message:
        """Return a deep-ish copy of the message."""
        return Message(
            role=self.role,
            content=list(self.content) if isinstance(self.content, list) else self.content,
            name=self.name,
            tool_calls=[ToolCall.from_dict(call.to_dict()) for call in self.tool_calls],
            tool_call_id=self.tool_call_id,
            metadata=dict(self.metadata),
            created_at=self.created_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.metadata:
            data["metadata"] = self.metadata
        data["created_at"] = self.created_at
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Rebuild a message from :meth:`to_dict` output."""
        return cls(
            role=Role(data.get("role", "user")),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=[ToolCall.from_dict(call) for call in data.get("tool_calls") or []],
            tool_call_id=data.get("tool_call_id"),
            metadata=dict(data.get("metadata") or {}),
            created_at=data.get("created_at", utc_now()),
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"[{self.role}] {self.text}"


# --------------------------------------------------------------------------- #
# Usage / responses
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    """Token accounting for a single call or an entire run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __iadd__(self, other: Usage) -> Usage:  # pragma: no cover - trivial
        combined = self + other
        self.prompt_tokens = combined.prompt_tokens
        self.completion_tokens = combined.completion_tokens
        self.total_tokens = combined.total_tokens
        self.cached_tokens = combined.cached_tokens
        return self

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Usage:
        """Build usage from a provider payload, tolerating missing fields."""
        if not data:
            return cls()
        prompt = int(data.get("prompt_tokens", 0) or 0)
        completion = int(data.get("completion_tokens", 0) or 0)
        total = int(data.get("total_tokens", 0) or (prompt + completion))
        cached = int(data.get("cached_tokens", 0) or 0)
        return cls(prompt, completion, total, cached)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class ToolResult:
    """Outcome of executing one :class:`ToolCall`."""

    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False
    duration_ms: float = 0.0
    output: Any = None

    @property
    def text(self) -> str:
        """Return the textual form of the result payload."""
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, ensure_ascii=False, default=str)

    def to_message(self) -> Message:
        """Convert the result into the tool role message fed back to the model."""
        return Message.tool_result(
            self.content,
            tool_call_id=self.tool_call_id,
            name=self.name,
            is_error=self.is_error,
            duration_ms=round(self.duration_ms, 3),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolResult:
        """Rebuild a tool result from :meth:`to_dict` output."""
        return cls(
            tool_call_id=data["tool_call_id"],
            name=data["name"],
            content=data.get("content", ""),
            is_error=bool(data.get("is_error", False)),
            duration_ms=float(data.get("duration_ms", 0.0)),
        )


@dataclass
class LLMResponse:
    """A complete (non-streamed) provider response."""

    message: Message
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: Optional[str] = None
    raw: Any = None

    @property
    def text(self) -> str:
        """Text content of the response message."""
        return self.message.text

    @property
    def tool_calls(self) -> List[ToolCall]:
        """Tool calls requested by the model."""
        return self.message.tool_calls


@dataclass
class StreamChunk:
    """One increment of a streaming provider response.

    ``kind`` is one of ``"text"``, ``"tool_call"``, ``"usage"`` or
    ``"response"``. The final ``"response"`` chunk always carries the assembled
    :class:`LLMResponse` so callers can rely on a single code path.
    """

    kind: str
    text: str = ""
    tool_call: Optional[ToolCall] = None
    usage: Optional[Usage] = None
    response: Optional[LLMResponse] = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text if self.kind == "text" else f"<{self.kind}>"


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """One reasoning iteration: an assistant turn plus any tool executions."""

    index: int
    message: Message
    tool_results: List[ToolResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=utc_now)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "tool_results": [result.to_dict() for result in self.tool_results],
            "usage": self.usage.to_dict(),
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Step:
        """Rebuild a step from :meth:`to_dict` output."""
        return cls(
            index=int(data.get("index", 0)),
            message=Message.from_dict(data["message"]),
            tool_results=[ToolResult.from_dict(item) for item in data.get("tool_results") or []],
            usage=Usage.from_dict(data.get("usage")),
            started_at=data.get("started_at", utc_now()),
            duration_ms=float(data.get("duration_ms", 0.0)),
        )


@dataclass
class AgentEvent:
    """An observable event emitted while a run executes."""

    type: EventType
    agent: str = ""
    message: Optional[Message] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    step: int = 0
    created_at: float = field(default_factory=utc_now)
    #: Live object attached to the event (an ``LLMResponse`` on MESSAGE events,
    #: a ``RunResult`` on RUN_END). Excluded from :meth:`to_dict`.
    raw: Any = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        if self.type is EventType.TEXT:
            return self.text
        return f"<{self.type} agent={self.agent or '-'} step={self.step}>"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "type": self.type.value,
            "agent": self.agent,
            "text": self.text,
            "run_id": self.run_id,
            "step": self.step,
            "data": self.data,
            "created_at": self.created_at,
        }


@dataclass
class Handoff:
    """A transfer of control from one agent to another."""

    from_agent: str
    to_agent: str
    task: str = ""
    reason: str = ""
    created_at: float = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "task": self.task,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass
class RunResult:
    """The outcome of an agent or team run.

    Attributes:
        run_id: Unique id, also the key used by run stores.
        output: Final textual answer.
        status: Terminal :class:`RunStatus`.
        messages: Full transcript, including system and tool messages.
        steps: Reasoning iterations with their tool results.
        usage: Aggregated token usage.
        handoffs: Agent-to-agent transfers that happened during the run.
        structured: Parsed ``output_model`` instance when one was configured.
    """

    run_id: str = field(default_factory=lambda: new_id("run"))
    output: str = ""
    status: RunStatus = RunStatus.PENDING
    agent: str = ""
    messages: List[Message] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    handoffs: List[Handoff] = field(default_factory=list)
    error: Optional[str] = None
    structured: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=utc_now)
    finished_at: Optional[float] = None

    # ---- convenience ----------------------------------------------------- #
    @property
    def ok(self) -> bool:
        """``True`` when the run completed successfully."""
        return self.status is RunStatus.COMPLETED

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration of the run in milliseconds."""
        end = self.finished_at if self.finished_at is not None else utc_now()
        return (end - self.started_at) * 1000.0

    @property
    def tool_calls(self) -> List[ToolCall]:
        """Every tool call made during the run, in order."""
        return [call for step in self.steps for call in step.message.tool_calls]

    @property
    def tool_results(self) -> List[ToolResult]:
        """Every tool result produced during the run, in order."""
        return [result for step in self.steps for result in step.tool_results]

    @property
    def iterations(self) -> int:
        """Number of reasoning steps taken."""
        return len(self.steps)

    def raise_for_status(self) -> RunResult:
        """Raise :class:`~pyagents.errors.RunFailed` unless the run succeeded."""
        from .errors import RunFailed

        if not self.ok:
            raise RunFailed(
                f"Run {self.run_id} ended with status '{self.status}'"
                + (f": {self.error}" if self.error else ""),
                run_id=self.run_id,
                status=str(self.status),
            )
        return self

    def text(self) -> str:
        """Alias for :attr:`output` (mirrors common agent-framework APIs)."""
        return self.output

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.output

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data: Dict[str, Any] = {
            "run_id": self.run_id,
            "output": self.output,
            "status": self.status.value,
            "agent": self.agent,
            "messages": [message.to_dict() for message in self.messages],
            "steps": [step.to_dict() for step in self.steps],
            "usage": self.usage.to_dict(),
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "error": self.error,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.structured is not None:
            data["structured"] = _jsonable(self.structured)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunResult:
        """Rebuild a result from :meth:`to_dict` output."""
        return cls(
            run_id=data.get("run_id") or new_id("run"),
            output=data.get("output", ""),
            status=_coerce_status(data.get("status")),
            agent=data.get("agent", ""),
            messages=[Message.from_dict(item) for item in data.get("messages") or []],
            steps=[Step.from_dict(item) for item in data.get("steps") or []],
            usage=Usage.from_dict(data.get("usage")),
            handoffs=[Handoff(**item) for item in data.get("handoffs") or []],
            error=data.get("error"),
            structured=data.get("structured"),
            metadata=dict(data.get("metadata") or {}),
            started_at=data.get("started_at", utc_now()),
            finished_at=data.get("finished_at"),
        )

    def last_message(self) -> Optional[Message]:
        """Return the final message of the transcript, if any."""
        return self.messages[-1] if self.messages else None

    def transcript(self, *, include_system: bool = False) -> Iterator[Message]:
        """Iterate the transcript, optionally skipping system messages."""
        for message in self.messages:
            if include_system or message.role is not Role.SYSTEM:
                yield message


def _coerce_status(value: Any) -> RunStatus:
    """Accept ``"running"``, ``"RunStatus.RUNNING"`` or a :class:`RunStatus`."""
    if isinstance(value, RunStatus):
        return value
    token = str(value or "pending").split(".")[-1].strip().lower()
    try:
        return RunStatus(token)
    except ValueError:  # pragma: no cover - unknown status from an old store
        return RunStatus.PENDING


def _jsonable(value: Any) -> Any:
    """Convert pydantic models / dataclasses into JSON-friendly structures."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and hasattr(value, "__fields__"):
        return value.dict()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def messages_to_dicts(messages: Iterable[Message]) -> List[Dict[str, Any]]:
    """Serialise an iterable of messages into a list of mappings."""
    return [message.to_dict() for message in messages]
