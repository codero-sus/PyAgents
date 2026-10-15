"""An offline provider that echoes the prompt.

No network, no credentials, no model weights - :class:`EchoProvider` simply
repeats the user's last message. It exists so that examples, the CLI and the
test suite can run end-to-end on a laptop with no configuration::

    $ pyagents run "hello" --provider echo
    You said: "hello". (This is the offline echo provider.)

It also demonstrates the minimum surface a custom provider must implement:
:meth:`EchoProvider.complete` / :meth:`EchoProvider.stream` (and their async
counterparts) plus the four wire-format hooks required by
:class:`~pyagents.llm.base.BaseProvider`.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from ..types import LLMResponse, Message, Role, StreamChunk, Usage
from .base import BaseProvider, resolve_messages
from .transport import get_default_transport

__all__ = ["EchoProvider"]


class EchoProvider(BaseProvider):
    """Offline provider that replies by echoing the user's message."""

    name = "echo"
    requires_api_key = False
    default_model = "echo"

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        kwargs.setdefault("transport", get_default_transport())
        kwargs["api_key"] = "offline"
        super().__init__(model or self.default_model, **kwargs)
        self._transport = kwargs.get("transport")
        self._owns_transport = False
        self._call_count = 0

    @classmethod
    def default_base_url(cls) -> str:
        """Echo never performs HTTP, but the base class wants a value."""
        return "http://offline.local/v1"

    # ---- BaseProvider hooks (unused: all network calls are overridden) ---- #
    def _endpoint(self) -> str:  # pragma: no cover - never called
        return f"{self.base_url}/chat/completions"

    def _auth_headers(self) -> Dict[str, str]:  # pragma: no cover - never called
        return {}

    def _payload(self, messages, **kwargs) -> Dict[str, Any]:  # pragma: no cover - never called
        return {"model": self.model}

    def _parse_response(
        self, data: Mapping[str, Any]
    ) -> LLMResponse:  # pragma: no cover - never called
        return LLMResponse(
            message=Message.assistant(str(data.get("content", ""))), model=self.model
        )

    # ---- real behaviour --------------------------------------------------- #
    def _reply(self, messages: Sequence[Message]) -> LLMResponse:
        """Build the echo response for ``messages``."""
        self._call_count += 1
        last_user = ""
        for message in reversed(messages):
            if message.role is Role.USER:
                last_user = message.text
                break
        text = f'You said: "{last_user}". (This is the offline echo provider.)'
        usage = Usage(prompt_tokens=len(last_user) // 4 + 1, completion_tokens=len(text) // 4 + 1)
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return LLMResponse(
            message=Message.assistant(text),
            model=self.model,
            usage=usage,
            finish_reason="stop",
        )

    def complete(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Any = None,
        **options: Any,
    ) -> LLMResponse:
        """Return an echo of the last user message."""
        return self._reply(resolve_messages(messages))

    async def acomplete(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Any = None,
        **options: Any,
    ) -> LLMResponse:
        """Async variant of :meth:`complete`."""
        await asyncio.sleep(0)
        return self.complete(messages, tools=tools, tool_choice=tool_choice, **options)

    def stream(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Any = None,
        **options: Any,
    ) -> Iterator[StreamChunk]:
        """Stream the echo response word by word."""
        response = self.complete(messages, tools=tools, tool_choice=tool_choice, **options)
        yield from self._chunks(response)

    async def astream(
        self,
        messages: Iterable[Any],
        *,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Any = None,
        **options: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async variant of :meth:`stream`."""
        response = await self.acomplete(messages, tools=tools, tool_choice=tool_choice, **options)
        for chunk in self._chunks(response):
            await asyncio.sleep(0)
            yield chunk

    @staticmethod
    def _chunks(response: LLMResponse) -> Iterator[StreamChunk]:
        words = response.message.text.split(" ")
        for index, word in enumerate(words):
            suffix = "" if index == len(words) - 1 else " "
            yield StreamChunk(kind="text", text=word + suffix)
        yield StreamChunk(kind="usage", usage=response.usage)
        yield StreamChunk(kind="response", response=response)
