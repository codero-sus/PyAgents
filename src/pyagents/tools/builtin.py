"""Batteries-included tools.

These cover the common cases (arithmetic, time, files, HTTP, code execution)
so examples and quick experiments work out of the box. They are deliberately
explicit about their capabilities - :func:`python_repl` and
:func:`http_request` execute untrusted content, so read their docstrings
before enabling them in production.
"""

from __future__ import annotations

import ast
import datetime as _dt
import io
import json
import math
import operator
import os
import textwrap
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..errors import ConfigurationError
from .base import Toolkit, tool

__all__ = [
    "calculator",
    "current_time",
    "default_toolkit",
    "default_tools",
    "http_get",
    "http_request",
    "list_directory",
    "python_repl",
    "read_file",
    "write_file",
]

# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
_BIN_OPS: Dict[Any, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
_UNARY_OPS: Dict[Any, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}
_SAFE_FUNCTIONS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _eval_node(node: ast.AST) -> Any:
    """Evaluate a restricted AST node (no attribute access, no calls to user code)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            raise ValueError("Only a whitelist of math functions can be called")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        return _SAFE_FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(item) for item in node.elts]
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool(tags=["math"])
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Supports ``+ - * / // % **``, parentheses and common math functions
    (``sqrt``, ``log``, ``sin``, ``factorial``, ...).

    Args:
        expression: Expression to evaluate, e.g. ``"(2 + 3) * sqrt(16)"``.
    """
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree)
    return f"{expression.strip()} = {result}"


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
@tool(tags=["time"])
def current_time(timezone: Optional[str] = None) -> str:
    """Return the current date and time.

    Args:
        timezone: IANA timezone name such as ``"Asia/Kolkata"`` or ``"UTC"``.
            Defaults to the system local time.
    """
    if timezone:
        try:
            from zoneinfo import ZoneInfo

            now = _dt.datetime.now(ZoneInfo(timezone))
        except Exception as exc:
            return f"Unknown timezone {timezone!r}: {exc}"
    else:
        now = _dt.datetime.now().astimezone()
    return now.isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _resolve(path: str, root: Optional[Path]) -> Path:
    """Resolve ``path`` and (optionally) confine it to ``root``."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = (root or Path.cwd()) / target
    resolved = target.resolve()
    if root is not None:
        root_resolved = root.resolve()
        if root_resolved not in resolved.parents and resolved != root_resolved:
            raise PermissionError(f"Path {path!r} escapes the allowed root {root_resolved}")
    return resolved


@tool(tags=["files"])
def read_file(path: str, max_bytes: int = 20000) -> str:
    """Read a UTF-8 text file from disk.

    Args:
        path: File path, absolute or relative to the current directory.
        max_bytes: Maximum number of bytes to read.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        return f"File not found: {path}"
    data = target.read_bytes()[:max_bytes]
    text = data.decode("utf-8", "replace")
    suffix = "" if len(data) < max_bytes else f"\n... [truncated at {max_bytes} bytes]"
    return text + suffix


@tool(tags=["files"])
def write_file(path: str, content: str) -> str:
    """Write text to a file, creating parent directories as needed.

    Args:
        path: Destination path.
        content: Text to write.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {target}"


@tool(tags=["files"])
def list_directory(path: str = ".") -> str:
    """List the entries of a directory.

    Args:
        path: Directory to list; defaults to the current directory.
    """
    target = Path(path).expanduser()
    if not target.is_dir():
        return f"Not a directory: {path}"
    entries: List[str] = []
    for item in sorted(target.iterdir()):
        if item.name.startswith("."):
            continue
        entries.append(
            f"{item.name}/" if item.is_dir() else f"{item.name} ({item.stat().st_size} bytes)"
        )
    return "\n".join(entries) or "(empty directory)"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Mapping[str, str]] = None,
    body: Optional[str] = None,
    timeout: float = 20.0,
    max_chars: int = 8000,
) -> str:
    """Perform an HTTP request and return a compact text rendering."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return f"Refusing to fetch unsupported URL scheme: {parsed.scheme or '(none)'}"
    data = body.encode("utf-8") if isinstance(body, str) else None
    request = urllib.request.Request(url, data=data, method=method.upper())
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    if data is not None and "Content-Type" not in (headers or {}):
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", "replace")
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except Exception as exc:
        return f"Request failed: {type(exc).__name__}: {exc}"
    truncated = payload[:max_chars] + ("\n... [truncated]" if len(payload) > max_chars else "")
    return f"HTTP {status} ({content_type})\n{truncated}"


@tool(tags=["web"])
def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    """Fetch a URL and return the response body as text.

    Args:
        url: Absolute http(s) URL.
        method: HTTP method, e.g. ``GET`` or ``POST``.
        headers: Optional request headers.
        body: Optional request body (sent as UTF-8).
        timeout: Request timeout in seconds.
    """
    return _http_request(url, method=method, headers=headers, body=body, timeout=timeout)


@tool(tags=["web"])
def http_get(url: str, timeout: float = 20.0) -> str:
    """Fetch a URL with a GET request.

    Args:
        url: Absolute http(s) URL.
        timeout: Request timeout in seconds.
    """
    return _http_request(url, method="GET", timeout=timeout)


# --------------------------------------------------------------------------- #
# Code execution
# --------------------------------------------------------------------------- #
@tool(tags=["code"])
def python_repl(code: str) -> str:
    """Execute Python code and return everything it printed.

    The code runs in-process with full privileges - only expose this tool to
    models and users you trust, or override it with a sandboxed executor.

    Args:
        code: Python source to execute. ``print`` calls become the result.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    scope: Dict[str, Any] = {"__name__": "__pyagents__", "json": json, "math": math, "os": os}
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(textwrap.dedent(code), scope)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}\n{stdout.getvalue()}".strip()
    output = stdout.getvalue()
    if stderr.getvalue():
        output += f"\n[stderr]\n{stderr.getvalue()}"
    return output.strip() or "(no output)"


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
def default_tools(include: Optional[Iterable[str]] = None) -> List[Any]:
    """Return a standard tool bundle.

    Args:
        include: Optional subset of names. Defaults to every safe tool
            (``calculator``, ``current_time``, ``read_file``, ``list_directory``
            and ``http_get``); pass ``["all"]`` to also add ``http_request``,
            ``write_file`` and ``python_repl``.
    """
    registry: Dict[str, Any] = {
        "calculator": calculator,
        "current_time": current_time,
        "read_file": read_file,
        "write_file": write_file,
        "list_directory": list_directory,
        "http_get": http_get,
        "http_request": http_request,
        "python_repl": python_repl,
    }
    default_names = ["calculator", "current_time", "read_file", "list_directory", "http_get"]
    if include is None:
        names = default_names
    else:
        requested = list(include)
        names = list(registry) if "all" in requested else requested
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ConfigurationError(
            f"Unknown built-in tools: {', '.join(unknown)}", known=sorted(registry)
        )
    return [registry[name] for name in names]


def default_toolkit(include: Optional[Iterable[str]] = None) -> Toolkit:
    """Return :func:`default_tools` wrapped in a :class:`Toolkit`."""
    return Toolkit(default_tools(include))
