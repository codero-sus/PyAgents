"""Conversation memory strategies.

Memory objects decide which past messages are replayed to the model on the next
turn. They are deliberately small and composable:

* :class:`BufferMemory` - keep the last *N* messages
* :class:`WindowMemory` - keep the last *N* messages but always keep the system
  prompt and the first user turn ("anchored" window)
* :class:`TokenWindowMemory` - keep as much as fits in a token budget
* :class:`SummaryMemory` - summarise older turns with an LLM once a threshold is
  crossed
* :class:`VectorMemory` - retrieve the most relevant past messages by embedding
  similarity (see :mod:`pyagents.embeddings`)

All of them support ``+`` to concatenate history and are JSON-serialisable via
``to_dict``/``from_dict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .errors import ConfigurationError
from .types import Message, Role
from .utils import estimate_message_tokens, maybe_await

__all__ = [
    "BufferMemory",
    "CompositeMemory",
    "Memory",
    "SummaryMemory",
    "TokenWindowMemory",
    "VectorMemory",
    "WindowMemory",
    "memory_from_dict",
]


class Memory(ABC):
    """Abstract conversation memory."""

    #: Human-readable strategy name.
    kind: str = "memory"

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append ``message`` to the memory."""

    @abstractmethod
    def messages(self) -> List[Message]:
        """Return the messages that should be replayed to the model."""

    @abstractmethod
    def clear(self) -> None:
        """Forget everything."""

    def extend(self, messages: Iterable[Message]) -> None:
        """Append several messages."""
        for message in messages:
            self.add(message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the memory (including stored messages) to JSON-friendly data."""
        return {
            "kind": self.kind,
            "messages": [message.to_dict() for message in getattr(self, "_messages", [])],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        """Rebuild a memory from :meth:`to_dict` output."""
        memory = cls()
        memory.clear()
        memory.extend(Message.from_dict(item) for item in data.get("messages", []))
        return memory

    def __len__(self) -> int:
        return len(self.messages())

    def __iter__(self):
        return iter(self.messages())

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{type(self).__name__}({len(self)} messages)"


class _ListMemory(Memory):
    """Base class for memories backed by a simple list."""

    kind = "list"

    def __init__(self) -> None:
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def messages(self) -> List[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    # ---- serialisation --------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        for key in ("max_messages", "max_tokens", "keep_system", "keep_first_user"):
            if hasattr(self, key):
                data[key] = getattr(self, key)
        return data

    def __getstate__(self) -> Dict[str, Any]:  # pragma: no cover - pickling
        return self.__dict__.copy()


class BufferMemory(_ListMemory):
    """Keep the most recent ``max_messages`` messages.

    >>> memory = BufferMemory(max_messages=2)
    >>> for index in range(4):
    ...     memory.add(Message.user(f"m{index}"))
    >>> ", ".join(message.text for message in memory.messages())
    'm2, m3'
    >>> len(memory.messages())
    2
    """

    kind = "buffer"

    def __init__(self, max_messages: Optional[int] = 20) -> None:
        super().__init__()
        if max_messages is not None and max_messages <= 0:
            raise ConfigurationError(
                "max_messages must be positive or None", max_messages=max_messages
            )
        self.max_messages = max_messages

    def messages(self) -> List[Message]:
        if self.max_messages is None:
            return list(self._messages)
        return list(self._messages[-self.max_messages :])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BufferMemory:
        memory = cls(max_messages=data.get("max_messages", 20))
        memory.extend(Message.from_dict(item) for item in data.get("messages", []))
        return memory


class WindowMemory(_ListMemory):
    """Keep a sliding window that always retains the system prompt.

    Useful when an agent has long-running context (a persona, coding rules)
    that must never be evicted.
    """

    kind = "window"

    def __init__(
        self,
        max_messages: int = 20,
        *,
        keep_system: bool = True,
        keep_first_user: bool = True,
    ) -> None:
        super().__init__()
        self.max_messages = max_messages
        self.keep_system = keep_system
        self.keep_first_user = keep_first_user

    def messages(self) -> List[Message]:
        pinned: List[Message] = []
        rest: List[Message] = []
        first_user_taken = False
        for message in self._messages:
            if self.keep_system and message.role is Role.SYSTEM:
                pinned.append(message)
                continue
            if self.keep_first_user and not first_user_taken and message.role is Role.USER:
                pinned.append(message)
                first_user_taken = True
                continue
            rest.append(message)
        budget = max(0, self.max_messages - len(pinned))
        return pinned + rest[-budget:] if budget else pinned

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WindowMemory:
        memory = cls(
            max_messages=data.get("max_messages", 20),
            keep_system=data.get("keep_system", True),
            keep_first_user=data.get("keep_first_user", True),
        )
        memory.extend(Message.from_dict(item) for item in data.get("messages", []))
        return memory


class TokenWindowMemory(_ListMemory):
    """Keep the newest messages that fit into a token budget.

    Args:
        max_tokens: Budget for the replayed history (excluding the agent's own
            system prompt, which is always sent).
        token_counter: Optional callable returning a token count for a list of
            messages. Defaults to the built-in estimate.
    """

    kind = "token_window"

    def __init__(
        self,
        max_tokens: int = 4000,
        *,
        token_counter: Optional[Callable[[Sequence[Message]], int]] = None,
        keep_system: bool = True,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.keep_system = keep_system
        self.token_counter = token_counter or estimate_message_tokens

    def messages(self) -> List[Message]:
        pinned = [
            message
            for message in self._messages
            if self.keep_system and message.role is Role.SYSTEM
        ]
        regular = [message for message in self._messages if message not in pinned]
        kept: List[Message] = []
        used = 0
        for message in reversed(regular):
            cost = self.token_counter([message])
            # Always keep the newest turn, but never exceed the budget once
            # something is already in the window (or a system prompt is pinned).
            if used + cost > self.max_tokens and (kept or pinned):
                break
            used += cost
            kept.append(message)
        keep_ids = {id(message) for message in kept} | {id(message) for message in pinned}
        return [message for message in self._messages if id(message) in keep_ids]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenWindowMemory:
        memory = cls(max_tokens=data.get("max_tokens", 4000))
        memory.extend(Message.from_dict(item) for item in data.get("messages", []))
        return memory


class SummaryMemory(_ListMemory):
    """Compress old turns into a rolling summary using an LLM.

    Once more than ``keep_recent`` messages have accumulated beyond
    ``summary_threshold``, the oldest messages are summarised in a single
    provider call and replaced by a system message that starts with
    ``SUMMARY_PREFIX``.

    Args:
        provider: Any object with ``complete``/``acomplete`` - normally an
            :class:`~pyagents.llm.base.BaseProvider`.
        summary_threshold: Number of messages that triggers summarisation.
        keep_recent: Number of newest messages always kept verbatim.
        max_summary_tokens: Soft cap for the summary text.
    """

    kind = "summary"
    SUMMARY_PREFIX = "[conversation summary]"

    def __init__(
        self,
        provider: Any = None,
        *,
        summary_threshold: int = 12,
        keep_recent: int = 6,
        max_summary_tokens: int = 400,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        self.max_summary_tokens = max_summary_tokens
        self.summary: str = ""

    def messages(self) -> List[Message]:
        if self.summary:
            return [
                Message.system(f"{self.SUMMARY_PREFIX} {self.summary}"),
                *self._messages[-self.keep_recent :],
            ]
        return list(self._messages)

    def add(self, message: Message) -> None:
        super().add(message)
        if (
            self.provider is not None
            and len(self._messages) > self.summary_threshold + self.keep_recent
        ):
            self.summarize(in_place=True)

    async def aadd(self, message: Message) -> None:
        """Async variant of :meth:`add` that awaits the summarisation call."""
        self._messages.append(message)
        if (
            self.provider is not None
            and len(self._messages) > self.summary_threshold + self.keep_recent
        ):
            await self.summarize_async(in_place=True)

    # ---- summarisation --------------------------------------------------- #
    def _needs_summary(self) -> bool:
        return len(self._messages) > self.keep_recent

    def _pending(self) -> List[Message]:
        return self._messages[: -self.keep_recent] if self.keep_recent else list(self._messages)

    def _transcript(self, messages: Sequence[Message]) -> str:
        lines = []
        for message in messages:
            text = message.text.strip()
            if not text:
                continue
            lines.append(f"{message.role.value}: {text}")
        return "\n".join(lines)

    def _prompt(self, transcript: str) -> str:
        return (
            "You compress conversation history. Write a short factual summary that preserves "
            "names, numbers, decisions, code identifiers and open questions. Do not add advice.\n\n"
            + (f"Existing summary:\n{self.summary}\n\n" if self.summary else "")
            + f"Conversation to fold in:\n{transcript}\n\nSummary:"
        )

    def summarize(self, *, in_place: bool = False) -> str:
        """Summarise pending messages synchronously and store the result."""
        if self.provider is None:
            raise ConfigurationError("SummaryMemory needs a provider to summarise")
        pending = self._pending()
        if not pending:
            return self.summary
        from .types import Message as _Message

        response = self.provider.complete(
            [_Message.user(self._prompt(self._transcript(pending)))],
            max_tokens=self.max_summary_tokens,
        )
        self.summary = response.text.strip() or self.summary
        if in_place:
            self._messages = self._messages[-self.keep_recent :]
        return self.summary

    async def summarize_async(self, *, in_place: bool = False) -> str:
        """Async variant of :meth:`summarize`."""
        if self.provider is None:
            raise ConfigurationError("SummaryMemory needs a provider to summarise")
        pending = self._pending()
        if not pending:
            return self.summary
        from .types import Message as _Message

        response = await maybe_await(
            self.provider.acomplete(
                [_Message.user(self._prompt(self._transcript(pending)))],
                max_tokens=self.max_summary_tokens,
            )
        )
        self.summary = response.text.strip() or self.summary
        if in_place:
            self._messages = self._messages[-self.keep_recent :]
        return self.summary

    def clear(self) -> None:
        super().clear()
        self.summary = ""

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["summary"] = self.summary
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SummaryMemory:
        memory = cls()
        memory.summary = data.get("summary", "")
        memory.extend(Message.from_dict(item) for item in data.get("messages", []))
        return memory


class VectorMemory(_ListMemory):
    """Keep the most relevant past messages, ranked by embedding similarity.

    Args:
        embedder: Object with ``embed(texts) -> list[list[float]]``; see
            :mod:`pyagents.embeddings`.
        top_k: How many messages to replay, most similar first.
        always_keep: Number of newest messages always replayed.
    """

    kind = "vector"

    def __init__(self, embedder: Any, *, top_k: int = 6, always_keep: int = 2) -> None:
        super().__init__()
        if embedder is None:
            raise ConfigurationError("VectorMemory requires an embedder")
        self.embedder = embedder
        self.top_k = top_k
        self.always_keep = always_keep
        self._vectors: List[List[float]] = []
        self._query: str = ""

    def add(self, message: Message) -> None:
        self._messages.append(message)
        try:
            vector = self.embedder.embed([message.text or " "])[0]
        except Exception:
            vector = []
        self._vectors.append(list(vector))

    async def aadd(self, message: Message) -> None:
        """Async variant of :meth:`add`."""
        self._messages.append(message)
        try:
            vectors = await maybe_await(self.embedder.aembed([message.text or " "]))
            vector = list(vectors[0])
        except Exception:
            vector = []
        self._vectors.append(vector)

    def set_query(self, query: str) -> None:
        """Record the current user query used for retrieval.

        :meth:`pyagents.agent.Agent.run` calls this automatically so the next
        :meth:`messages` call returns query-relevant history.
        """
        self._query = query

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[Message]:
        """Return the ``top_k`` messages most similar to ``query``."""
        from .embeddings import cosine_similarity

        if not self._messages:
            return []
        k = top_k or self.top_k
        try:
            query_vector = list(self.embedder.embed([query or " "])[0])
        except Exception:
            return list(self._messages[-k:])
        scored: List[Tuple[int, float]] = []
        for index, vector in enumerate(self._vectors):
            if not vector or not query_vector or len(vector) != len(query_vector):
                continue
            scored.append((index, cosine_similarity(vector, query_vector)))
        scored.sort(key=lambda item: item[1], reverse=True)
        picked = sorted(index for index, _ in scored[:k])
        recent_from = max(0, len(self._messages) - self.always_keep)
        for index in range(recent_from, len(self._messages)):
            if index not in picked:
                picked.append(index)
        return [self._messages[index] for index in sorted(set(picked))]

    def messages(self) -> List[Message]:
        if not self._query:
            return list(self._messages[-max(self.top_k, self.always_keep) :])
        return self.retrieve(self._query)

    def clear(self) -> None:
        super().clear()
        self._vectors.clear()
        self._query = ""

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["top_k"] = self.top_k
        data["always_keep"] = self.always_keep
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorMemory:
        raise NotImplementedError("VectorMemory cannot be restored without its embedder")


class CompositeMemory(Memory):
    """Chain several memories, concatenating their outputs.

    Messages that appear in more than one member are returned once, so a
    composite never repeats history::

        >>> long_term = BufferMemory(max_messages=10)
        >>> short_term = BufferMemory(max_messages=1)
        >>> memory = CompositeMemory([long_term, short_term])
        >>> memory.extend([Message.user("first"), Message.user("second")])
        >>> ", ".join(message.text for message in memory.messages())
        'first, second'
    """

    kind = "composite"

    def __init__(self, memories: Sequence[Memory]) -> None:
        if not memories:
            raise ConfigurationError("CompositeMemory needs at least one memory")
        self.memories = list(memories)

    def add(self, message: Message) -> None:
        for memory in self.memories:
            memory.add(message)

    def messages(self) -> List[Message]:
        seen: List[str] = []
        merged: List[Message] = []
        for memory in self.memories:
            for message in memory.messages():
                fingerprint = f"{message.role}:{message.text}:{message.tool_call_id}"
                if fingerprint in seen:
                    continue
                seen.append(fingerprint)
                merged.append(message)
        return merged

    def clear(self) -> None:
        for memory in self.memories:
            memory.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "memories": [memory.to_dict() for memory in self.memories]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompositeMemory:
        return cls([memory_from_dict(item) for item in data.get("memories", [])])


#: Registry used by :func:`memory_from_dict`.
_MEMORY_TYPES: Dict[str, Any] = {
    "buffer": BufferMemory,
    "window": WindowMemory,
    "token_window": TokenWindowMemory,
    "summary": SummaryMemory,
    "composite": CompositeMemory,
    "list": BufferMemory,
}


def memory_from_dict(data: Dict[str, Any]) -> Memory:
    """Rebuild a memory object from :meth:`Memory.to_dict` output."""
    kind = str(data.get("kind", "buffer"))
    if kind not in _MEMORY_TYPES:
        raise ConfigurationError(f"Unknown memory kind '{kind}'", known=sorted(_MEMORY_TYPES))
    return _MEMORY_TYPES[kind].from_dict(data)
