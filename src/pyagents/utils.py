"""Small internal helpers: JSON handling, token estimation, docstring parsing.

Nothing in here is part of the public API contract, but the functions are
stable enough to be useful when writing custom tools and providers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

__all__ = [
    "camel_to_snake",
    "collect_annotations",
    "estimate_message_tokens",
    "estimate_tokens",
    "is_async_callable",
    "json_dumps",
    "maybe_await",
    "parse_docstring",
    "safe_json_loads",
    "slugify",
    "truncate",
]

_WS_RE = re.compile(r"\s+")
_GOOGLE_SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?|Notes?)\s*:\s*$"
)


def json_dumps(value: Any, *, indent: Optional[int] = None) -> str:
    """Serialise ``value`` to JSON, falling back to ``str`` for odd objects.

    >>> json_dumps({"a": 1})
    '{"a": 1}'
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str, indent=indent)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return json.dumps(str(value))


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning ``default`` instead of raising on bad input."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


async def maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def is_async_callable(func: Callable[..., Any]) -> bool:
    """Return ``True`` when calling ``func`` produces an awaitable."""
    if inspect.iscoroutinefunction(func):
        return True
    call = getattr(func, "__call__", None)  # noqa: B004 - callable instances
    return bool(call is not None and inspect.iscoroutinefunction(call))


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Uses the widely-cited "~4 characters per token" heuristic. The estimate is
    deliberately cheap and dependency-free; pass a real tokenizer to
    :func:`pyagents.memory.TokenWindowMemory` if you need exact counts.

    >>> estimate_tokens("hello world")
    2
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: Iterable[Any]) -> int:
    """Estimate the token count of a chat transcript.

    Adds a small per-message overhead for role/formatting tokens, mirroring how
    chat models charge for message framing.
    """
    total = 0
    for message in messages:
        content = getattr(message, "text", message)
        total += estimate_tokens(str(content)) + 4
        for call in getattr(message, "tool_calls", []) or []:
            total += estimate_tokens(json_dumps(call.arguments)) + 8
    return total


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """Truncate ``text`` to ``limit`` characters, appending ``suffix``."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def slugify(value: str, *, fallback: str = "agent") -> str:
    """Turn arbitrary text into a snake_case identifier.

    >>> slugify("Research Agent #1")
    'research_agent_1'
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    if not slug or slug[0].isdigit():
        slug = f"{fallback}_{slug}" if slug else fallback
    return slug


def camel_to_snake(name: str) -> str:
    """Convert ``CamelCase`` to ``snake_case``."""
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


# --------------------------------------------------------------------------- #
# Docstring parsing
# --------------------------------------------------------------------------- #
def parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Split a docstring into a summary and per-argument descriptions.

    Supports the Google style (``Args:`` section), the Sphinx style
    (``:param name:``) and NumPy style (``name : type`` + indented text).

    Returns:
        ``(summary, {"arg_name": "description"})``

    >>> summary, params = parse_docstring("Add numbers.\\n\\nArgs:\\n    a: left\\n    b: right")
    >>> summary
    'Add numbers.'
    >>> params["b"]
    'right'
    """
    if not doc:
        return "", {}
    lines = textwrap.dedent(doc).strip().splitlines()

    summary_lines: List[str] = []
    params: Dict[str, str] = {}
    section: Optional[str] = None
    current: Optional[str] = None
    buffer: List[str] = []
    param_line = re.compile(
        r"^(?P<name>[*a-zA-Z_][\w.]*)\s*(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<desc>.*)$"
    )
    sphinx_line = re.compile(
        r"^:(?:param|parameter|arg|argument|key|keyword)\s+(?P<name>[*\w]+)\s*:\s*(?P<desc>.*)$"
    )

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            text = _WS_RE.sub(" ", " ".join(buffer)).strip()
            if text:
                params[current] = text
        current, buffer = None, []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if sphinx_match := sphinx_line.match(stripped):
            flush()
            params.setdefault(
                sphinx_match.group("name").lstrip("*"), sphinx_match.group("desc").strip()
            )
            continue
        if _GOOGLE_SECTION_RE.match(stripped):
            flush()
            section = stripped.rstrip(":")
            continue
        if section in {"Args", "Arguments", "Parameters"}:
            if match := param_line.match(stripped):
                flush()
                current = match.group("name").lstrip("*")
                buffer = [match.group("desc")]
                continue
            indented = line.startswith((" " * 4, "\t"))
            if current is not None and indented:
                buffer.append(stripped)
                continue
            # A NumPy style "name : type" header followed by indented text.
            if (
                stripped
                and not indented
                and re.match(r"^[a-zA-Z_][\w]*\s*:\s*[a-zA-Z_\[\], ]+$", stripped)
            ):
                flush()
                current = stripped.split(":", 1)[0].strip()
                buffer = []
                continue
        if section is not None:
            continue
        if current is not None:
            buffer.append(stripped)
        else:
            summary_lines.append(stripped)

    flush()
    summary = " ".join(summary_lines).strip()
    if not summary and params:
        summary = "Call the tool."
    if not summary:
        summary = "Call the tool."
    return summary, params


def collect_annotations(func: Callable[..., Any]) -> Dict[str, Any]:
    """Return resolved type hints for ``func``, ignoring unresolvable names.

    ``include_extras=True`` keeps ``Annotated[...]`` metadata, which is how
    argument descriptions attached with ``Annotated`` reach the JSON schema.
    """
    from typing import get_type_hints

    raw = dict(getattr(func, "__annotations__", {}))
    raw.pop("return", None)
    try:
        hints = dict(get_type_hints(func, include_extras=True))
        hints.pop("return", None)
        return hints
    except TypeError:  # pragma: no cover - Python < 3.9 without include_extras
        try:
            hints = dict(get_type_hints(func))
            hints.pop("return", None)
            return hints
        except Exception:
            pass
    except Exception:
        pass

    # Resolve what we can one annotation at a time; skip the rest rather than
    # handing raw strings to the schema builder.
    resolved: Dict[str, Any] = {}
    globalns = getattr(func, "__globals__", {})
    for name, annotation in raw.items():
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, globalns, None)
            except Exception:
                continue
        resolved[name] = annotation
    return resolved


async def gather_limited(
    coros: Sequence[Any],
    limit: int,
) -> List[Any]:
    """Run awaitables concurrently with a maximum in-flight count.

    ``limit <= 0`` means unbounded concurrency.
    """
    if limit <= 0 or limit >= len(coros):
        return list(await asyncio.gather(*coros))

    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(run(coro) for coro in coros)))


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield ``items`` in slices of at most ``size`` elements."""
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]


def ensure_list(value: Any) -> List[Any]:
    """Wrap a scalar in a list; pass lists/tuples/sets through as lists."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def typeof_name(cls: Type[Any]) -> str:
    """Return a readable name for a type."""
    return getattr(cls, "__name__", str(cls))
