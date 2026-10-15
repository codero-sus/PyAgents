"""HTTP transports used by the provider adapters.

PyAgents ships with a stdlib transport (:class:`UrllibTransport`) so the core
package has **zero required dependencies**, and an optional
:class:`HttpxTransport` that is picked up automatically when ``httpx`` is
installed. Both expose the same tiny interface, which makes them trivial to
fake in tests - see :class:`pyagents.testing.FakeTransport`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Tuple

from ..errors import ProviderAuthError, ProviderError, ProviderRateLimitError, ProviderTimeoutError

__all__ = [
    "HttpTransport",
    "HttpxTransport",
    "SSEEvent",
    "UrllibTransport",
    "get_default_transport",
    "iter_sse",
]

SSEEvent = Tuple[Optional[str], str]


def iter_sse(lines: Iterator[str]) -> Iterator[SSEEvent]:
    """Parse a stream of SSE lines into ``(event, data)`` pairs.

    Comments and unknown fields are skipped; multi-line ``data:`` payloads are
    joined with newlines as required by the SSE specification.
    """
    event: Optional[str] = None
    data_lines: List[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data_lines or event:
                yield event, "\n".join(data_lines)
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data_lines.append(value)
        elif field == "event":
            event = value
        elif field == "done":
            continue
    if data_lines or event:
        yield event, "\n".join(data_lines)


def _status_to_error(provider: str, status: int, body: Any, message: str = "") -> ProviderError:
    """Map an HTTP status code onto the right PyAgents error type."""
    detail = message or _extract_message(body)
    if status in (401, 403):
        return ProviderAuthError(
            detail or "authentication failed", provider=provider, status_code=status, body=body
        )
    if status == 429:
        return ProviderRateLimitError(
            detail or "rate limited", provider=provider, status_code=status, body=body
        )
    if status in (408, 504):
        return ProviderTimeoutError(
            detail or "request timed out", provider=provider, status_code=status, body=body
        )
    return ProviderError(
        detail or f"HTTP {status}", provider=provider, status_code=status, body=body
    )


def _extract_message(body: Any) -> str:
    """Pull a human-readable error message out of a provider error payload."""
    if isinstance(body, str):
        return body.strip()[:500]
    if isinstance(body, Mapping):
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, Mapping):
                inner = value.get("message") or value.get("detail")
                if inner:
                    return str(inner)[:500]
            elif value:
                return str(value)[:500]
    return ""


class HttpTransport(ABC):
    """Minimal transport interface: JSON POST plus server-sent events."""

    #: Short name used in error messages and ``repr``.
    name = "http"

    @abstractmethod
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Dict[str, Any]:
        """POST ``payload`` as JSON and return the decoded response body."""

    @abstractmethod
    def post_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Iterator[str]:
        """POST ``payload`` and yield raw SSE lines from the response body."""

    async def apost_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Dict[str, Any]:
        """Async variant of :meth:`post_json` (defaults to a worker thread)."""
        return await asyncio.to_thread(
            self.post_json,
            url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            provider=provider,
        )

    async def apost_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> AsyncIterator[str]:
        """Async variant of :meth:`post_sse`.

        The default implementation bridges the blocking iterator through a
        worker thread, which keeps async streaming working on the stdlib
        transport as well.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        sentinel = object()

        iterator = self.post_sse(
            url, headers=headers, payload=payload, timeout=timeout, provider=provider
        )

        def pump() -> None:
            try:
                for line in iterator:
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        pump_task = loop.run_in_executor(None, pump)
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not pump_task.done():  # pragma: no cover - cooperative shutdown
                pump_task.cancel()

    def close(self) -> None:  # noqa: B027 - optional hook with a no-op default
        """Release any resources held by the transport."""

    def __enter__(self) -> HttpTransport:  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info: Any) -> None:  # pragma: no cover - convenience
        self.close()


class UrllibTransport(HttpTransport):
    """Blocking transport built on :mod:`urllib.request`.

    Requires no third-party packages. Async calls run in the default executor.
    """

    name = "urllib"

    def _request(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str,
    ) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8", "replace")
            except Exception:  # pragma: no cover - stream already consumed
                raw = ""
            raise _status_to_error(provider, exc.code, raw) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise ProviderTimeoutError(
                    f"{provider} request timed out after {timeout}s", provider=provider
                ) from exc
            raise ProviderError(f"{provider} request failed: {reason}", provider=provider) from exc
        except (socket.timeout, TimeoutError) as exc:  # pragma: no cover - rare
            raise ProviderTimeoutError(
                f"{provider} request timed out after {timeout}s", provider=provider
            ) from exc

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Dict[str, Any]:
        """POST JSON and decode the response body."""
        with self._request(url, headers, payload, timeout, provider) as response:
            raw = response.read().decode("utf-8", "replace")
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - provider bug
            raise ProviderError(
                f"{provider} returned invalid JSON: {raw[:200]}", provider=provider, body=raw
            ) from exc

    def post_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Iterator[str]:
        """POST JSON and yield raw SSE lines."""
        response = self._request(url, headers, payload, timeout, provider)
        try:
            for raw_line in response:
                yield raw_line.decode("utf-8", "replace")
        except (socket.timeout, TimeoutError):  # pragma: no cover - network hiccup
            raise ProviderTimeoutError(f"{provider} stream timed out", provider=provider) from None
        finally:
            response.close()


class HttpxTransport(HttpTransport):
    """Transport backed by :mod:`httpx`, with native async I/O and pooling."""

    name = "httpx"

    def __init__(self, client: Any = None, async_client: Any = None) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - guarded by get_default_transport
            raise ProviderError(
                "httpx is not installed. Install it with `pip install pyagents[http]`."
            ) from exc
        self._httpx = httpx
        self._client = client or httpx.Client()
        self._async_client = async_client

    def _async(self) -> Any:
        if self._async_client is None:
            self._async_client = self._httpx.AsyncClient()
        return self._async_client

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Dict[str, Any]:
        """POST JSON and decode the response body."""
        try:
            response = self._client.post(url, json=payload, headers=dict(headers), timeout=timeout)
        except self._httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"{provider} request timed out after {timeout}s", provider=provider
            ) from exc
        except self._httpx.HTTPError as exc:
            raise ProviderError(f"{provider} request failed: {exc}", provider=provider) from exc
        if response.status_code >= 400:
            raise _status_to_error(provider, response.status_code, response.text)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - provider bug
            raise ProviderError(
                f"{provider} returned invalid JSON: {response.text[:200]}",
                provider=provider,
                body=response.text,
            ) from exc

    def post_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Iterator[str]:
        """POST JSON and yield raw SSE lines."""
        try:
            with self._client.stream(
                "POST", url, json=payload, headers=dict(headers), timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    raise _status_to_error(provider, response.status_code, response.read().decode())
                yield from response.iter_lines()
        except self._httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{provider} stream timed out", provider=provider) from exc

    async def apost_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> Dict[str, Any]:
        """Async POST JSON using :class:`httpx.AsyncClient`."""
        client = self._async()
        try:
            response = await client.post(
                url, json=dict(payload), headers=dict(headers), timeout=timeout
            )
        except self._httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"{provider} request timed out after {timeout}s", provider=provider
            ) from exc
        except self._httpx.HTTPError as exc:
            raise ProviderError(f"{provider} request failed: {exc}", provider=provider) from exc
        if response.status_code >= 400:
            raise _status_to_error(provider, response.status_code, response.text)
        return response.json() if response.content else {}

    async def apost_sse(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
        provider: str = "provider",
    ) -> AsyncIterator[str]:
        """Async SSE streaming using :class:`httpx.AsyncClient`."""
        client = self._async()
        try:
            async with client.stream(
                "POST", url, json=dict(payload), headers=dict(headers), timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise _status_to_error(provider, response.status_code, body)
                async for line in response.aiter_lines():
                    yield line
        except self._httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{provider} stream timed out", provider=provider) from exc

    def close(self) -> None:
        """Close the underlying httpx clients."""
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            self._client.close()
        if self._async_client is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - no running loop
                asyncio.get_event_loop().create_task(self._async_client.aclose())


_DEFAULT: Optional[HttpTransport] = None


def get_default_transport(prefer_httpx: bool = True) -> HttpTransport:
    """Return a process-wide default transport.

    Prefers :class:`HttpxTransport` when ``httpx`` is importable, otherwise
    falls back to the stdlib implementation. The instance is cached; call
    :func:`reset_default_transport` in tests that need a clean slate.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = _build_default_transport(prefer_httpx)
    return _DEFAULT


def _build_default_transport(prefer_httpx: bool) -> HttpTransport:
    if prefer_httpx:
        try:
            return HttpxTransport()
        except Exception:  # pragma: no cover - httpx missing or broken
            pass
    return UrllibTransport()


def reset_default_transport() -> None:
    """Drop the cached default transport (used by the test suite)."""
    global _DEFAULT
    if _DEFAULT is not None:
        _DEFAULT.close()
    _DEFAULT = None
