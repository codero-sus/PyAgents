"""Bridges between the sync and async APIs.

PyAgents is async-first: everything is implemented once, as a coroutine or an
async generator, and the synchronous API (``Agent.run``, ``Agent.stream``,
``Workflow.run``) is derived from it here.

When no event loop is running - a script, a REPL - the coroutine simply runs via
:func:`asyncio.run`. Inside a running loop (Jupyter, an async web handler) the
work is delegated to a short-lived worker thread with its own loop, so calling
``agent.run()`` from async code is safe instead of raising
``RuntimeError: This event loop is already running``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator, Optional, TypeVar

__all__ = ["has_running_loop", "iter_sync", "run_sync"]

T = TypeVar("T")

_SENTINEL = object()


def has_running_loop() -> bool:
    """Return ``True`` when called from inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def run_sync(
    factory: Callable[[], Coroutine[Any, Any, T]], *, timeout: Optional[float] = None
) -> T:
    """Run an async call from synchronous code.

    Args:
        factory: Zero-argument callable returning a fresh coroutine. Taking a
            factory (rather than a coroutine) lets the coroutine be created
            inside whichever event loop ends up running it.
        timeout: Optional wall-clock timeout in seconds.

    Returns:
        Whatever the coroutine returns.
    """
    if not has_running_loop():
        return asyncio.run(factory())

    box: dict = {}
    done = threading.Event()

    def worker() -> None:
        try:
            box["value"] = asyncio.run(factory())
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="pyagents-run-sync", daemon=True)
    thread.start()
    if not done.wait(timeout):
        raise TimeoutError("Timed out waiting for the run to finish")
    if "error" in box:
        raise box["error"]
    return box["value"]


def iter_sync(
    factory: Callable[[], AsyncIterator[T]],
    *,
    timeout: Optional[float] = None,
) -> Iterator[T]:
    """Iterate an async generator from synchronous code.

    The generator is created and driven on a worker thread with its own event
    loop; items are handed to the caller through an unbounded queue.
    """
    events: queue.Queue[Any] = queue.Queue()
    stopped = threading.Event()

    def worker() -> None:
        async def consume() -> None:
            async for item in factory():
                if stopped.is_set():  # consumer went away: stop early
                    break
                events.put(item)

        try:
            asyncio.run(consume())
        except BaseException as exc:
            events.put(exc)
        finally:
            events.put(_SENTINEL)

    thread = threading.Thread(target=worker, name="pyagents-stream", daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = events.get(timeout=timeout) if timeout else events.get()
            except queue.Empty as exc:  # pragma: no cover - only with timeout
                raise TimeoutError("Timed out waiting for the next event") from exc
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stopped.set()
