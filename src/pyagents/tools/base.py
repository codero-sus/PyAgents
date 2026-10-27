"""Tool definition, JSON-schema inference and argument validation.

A tool is any callable in the broadest sense: a plain function, an async
function, a lambda, or a method. PyAgents reads the signature and the
docstring to build the schema the model sees, so most tools need no
configuration at all:

    >>> from pyagents.tools import tool
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two integers.'''
    ...     return a + b
    >>> add.parameters["properties"]["a"]["type"]
    'integer'
    >>> add.parameters["required"]
    ['a', 'b']
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import json
import re
import time
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from ..errors import ToolError, ToolNotFoundError, ToolValidationError
from ..types import ToolCall, ToolResult
from ..utils import collect_annotations, is_async_callable, json_dumps, maybe_await, parse_docstring

__all__ = [
    "FunctionTool",
    "Tool",
    "Toolkit",
    "coerce_arguments",
    "json_schema_from_function",
    "tool",
    "validate_arguments",
]

F = TypeVar("F", bound=Callable[..., Any])

_PY_TO_JSON: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
    type(None): "null",
}

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")


# --------------------------------------------------------------------------- #
# Schema inference
# --------------------------------------------------------------------------- #
def _unwrap_optional(annotation: Any) -> Tuple[Any, bool, List[Any]]:
    """Split ``Optional[X]``/``Union[...]`` into its members."""
    origin = get_origin(annotation)
    if origin is Union:  # includes Optional
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        nullable = len(members) != len(get_args(annotation))
        return (members[0] if len(members) == 1 else Union[tuple(members)], nullable, members)
    return annotation, False, [annotation]


def _annotation_to_schema(
    annotation: Any, *, descriptions: Mapping[str, str] | None = None, name: str = ""
) -> Dict[str, Any]:
    """Convert a Python annotation into a JSON-schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    # Annotated[int, "docs"] / Annotated[int, Field(...)]
    origin = get_origin(annotation)
    extra: Dict[str, Any] = {}
    if origin is not None and getattr(annotation, "__metadata__", None):
        meta = annotation.__metadata__
        annotation = get_args(annotation)[0]
        origin = get_origin(annotation)
        for item in meta:
            if isinstance(item, str):
                extra.setdefault("description", item)
            elif isinstance(item, Mapping):
                extra.update(item)
            else:  # pydantic FieldInfo and friends
                for attr, key in (
                    ("description", "description"),
                    ("title", "title"),
                    ("ge", "minimum"),
                    ("le", "maximum"),
                    ("gt", "exclusiveMinimum"),
                    ("lt", "exclusiveMaximum"),
                ):
                    value = getattr(item, attr, None)
                    if value is not None:
                        extra.setdefault(key, value)

    annotation, nullable, _members = _unwrap_optional(annotation)
    schema: Dict[str, Any]

    if annotation in _PY_TO_JSON:
        schema = {"type": _PY_TO_JSON[annotation]}
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        schema = {"type": "string", "enum": [member.value for member in annotation]}
        schema["type"] = _PY_TO_JSON.get(type(next(iter(annotation)).value), "string")
    elif origin is not None and str(origin) in {"typing.Literal", "typing_extensions.Literal"}:
        values = list(get_args(annotation))
        types = {_PY_TO_JSON.get(type(value), "string") for value in values}
        schema = {"enum": values, "type": types.pop() if len(types) == 1 else "string"}
    elif _is_pydantic_model(annotation):
        schema = dict(annotation.model_json_schema())
    elif dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        schema = _dataclass_schema(annotation)
    elif origin in (list, set, frozenset, tuple, Sequence, List, Set, Iterable):
        args = get_args(annotation)
        item_schema = _annotation_to_schema(args[0]) if args else {}
        schema = {"type": "array"}
        if item_schema:
            schema["items"] = item_schema
    elif origin in (dict, Mapping, Dict):
        schema = {"type": "object"}
    else:
        # Unknown/unsupported annotations degrade to a free-form string.
        schema = {"type": "string"}

    schema.update(extra)
    if nullable and schema.get("type") and schema["type"] != "null":
        schema["type"] = [schema["type"], "null"]
    if name and descriptions and name in descriptions and "description" not in schema:
        schema["description"] = descriptions[name]
    return schema


def _is_pydantic_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and hasattr(annotation, "model_json_schema")


def _dataclass_schema(cls: type) -> Dict[str, Any]:
    """Build a JSON schema for a dataclass."""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    hints = collect_annotations(cls) or getattr(cls, "__annotations__", {})
    for field in dataclasses.fields(cls):
        properties[field.name] = _annotation_to_schema(hints.get(field.name, Any), name=field.name)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def json_schema_from_function(func: Callable[..., Any]) -> Dict[str, Any]:
    """Infer a JSON schema for the arguments of ``func``.

    Uses type hints, defaults, the docstring's parameter descriptions and
    pydantic/dataclass parameter annotations.
    """
    signature = inspect.signature(func)
    _, param_docs = parse_docstring(inspect.getdoc(func))
    hints = collect_annotations(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    additional = False

    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            properties["args"] = {
                "type": "array",
                "items": {},
                "description": "Positional arguments.",
            }
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            additional = True
            continue
        annotation = hints.get(name, param.annotation)
        schema = _annotation_to_schema(annotation, descriptions=param_docs, name=name)
        if name in param_docs and "description" not in schema:
            schema["description"] = param_docs[name]
        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema

    doc_summary, _ = parse_docstring(inspect.getdoc(func))
    object_schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional,
    }
    if required:
        object_schema["required"] = required
    if doc_summary:
        object_schema["description"] = doc_summary
    return object_schema


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "null":
        return value is None
    return True


def coerce_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Best-effort coercion of LLM-supplied argument values.

    Models frequently send ``"5"`` where an integer is expected, or
    ``"true"``/``"false"`` for booleans. Coercion keeps runs alive instead of
    failing on cosmetic type mismatches.
    """
    properties = schema.get("properties") or {}
    result: Dict[str, Any] = {}
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        expected_types = expected if isinstance(expected, list) else [expected]
        if isinstance(value, str) and "string" not in expected_types:
            if "integer" in expected_types and re.fullmatch(r"-?\d+", value.strip()):
                result[key] = int(value.strip())
                continue
            if "number" in expected_types and re.fullmatch(
                r"-?\d*\.?\d+([eE][-+]?\d+)?", value.strip()
            ):
                result[key] = float(value.strip())
                continue
            if "boolean" in expected_types and value.strip().lower() in {"true", "false"}:
                result[key] = value.strip().lower() == "true"
                continue
            if {"array", "object"} & set(expected_types):
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, (list, dict)):
                    result[key] = parsed
                    continue
            if expected_types and expected_types[0] and "string" not in expected_types:
                # Cannot interpret it; leave as-is and let validation complain.
                result[key] = value
                continue
        result[key] = value
    return result


def validate_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    tool_name: str = "tool",
    strict: bool = False,
    coerce: bool = True,
) -> Dict[str, Any]:
    """Validate (and optionally coerce) tool arguments against ``schema``.

    Raises:
        ToolValidationError: when a required argument is missing, an argument
            has the wrong type, or unexpected arguments are supplied while
            ``strict`` is set.
    """
    if not isinstance(arguments, Mapping):
        raise ToolValidationError(f"{tool_name}: arguments must be an object", tool=tool_name)
    properties: Mapping[str, Any] = schema.get("properties") or {}
    prepared = coerce_arguments(schema, arguments) if coerce else dict(arguments)

    missing = [name for name in (schema.get("required") or []) if name not in prepared]
    if missing:
        raise ToolValidationError(
            f"{tool_name}: missing required argument(s): {', '.join(sorted(missing))}",
            tool=tool_name,
        )

    if strict and schema.get("additionalProperties") is False:
        unexpected = sorted(set(prepared) - set(properties))
        if unexpected:
            raise ToolValidationError(
                f"{tool_name}: unexpected argument(s): {', '.join(unexpected)}", tool=tool_name
            )

    for key, value in prepared.items():
        spec = properties.get(key)
        if not spec or value is None:
            continue
        expected = spec.get("type")
        options = expected if isinstance(expected, list) else [expected]
        if not options or options == [None]:
            continue
        if not any(_matches_type(value, option) for option in options if option):
            raise ToolValidationError(
                f"{tool_name}: argument '{key}' expected {expected}, got {type(value).__name__}",
                tool=tool_name,
                argument=key,
            )
        enum = spec.get("enum")
        if enum and value not in enum:
            raise ToolValidationError(
                f"{tool_name}: argument '{key}' must be one of {enum}", tool=tool_name, argument=key
            )
    return prepared


# --------------------------------------------------------------------------- #
# Tool objects
# --------------------------------------------------------------------------- #
class Tool(Generic[F]):
    """A callable exposed to a model as a JSON-schema function.

    Args:
        func: The underlying callable (sync or async).
        name: Override the inferred name.
        description: Override the docstring-derived description.
        parameters: Full JSON schema override; inferred from the signature
            when omitted.
        strict: Reject unknown arguments instead of ignoring them.
        output_errors: Return an error observation to the model instead of
            raising when the tool fails.
        tags: Free-form labels, useful for filtering in :class:`Toolkit`.
    """

    def __init__(
        self,
        func: F,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        strict: bool = False,
        output_errors: bool = True,
        tags: Iterable[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not callable(func):
            raise ToolError(f"{type(func).__name__} is not callable")
        self.func = func
        inferred_name = getattr(func, "__name__", type(func).__name__)
        self.name = name or inferred_name
        if not _NAME_RE.match(self.name):
            raise ToolError(
                f"Invalid tool name {self.name!r}: use letters, digits, '_' or '-' (max 64 chars)"
            )
        doc_summary, _ = parse_docstring(inspect.getdoc(func))
        self.description = description or doc_summary or f"Call the {self.name} function."
        self.parameters: Dict[str, Any] = (
            dict(parameters) if parameters is not None else json_schema_from_function(func)
        )
        if not self.parameters.get("description"):
            self.parameters["description"] = self.description
        self.strict = strict
        self.output_errors = output_errors
        self.tags: Tuple[str, ...] = tuple(tags)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.is_async = is_async_callable(func)

    # ---- introspection --------------------------------------------------- #
    @property
    def signature(self) -> inspect.Signature:
        """Signature of the wrapped callable."""
        return inspect.signature(self.func)

    @property
    def required_arguments(self) -> List[str]:
        """Names of arguments the model must supply."""
        return list(self.parameters.get("required") or [])

    def to_spec(self) -> Dict[str, Any]:
        """Return the canonical tool description used across providers.

        >>> @tool
        ... def ping() -> str:
        ...     '''Ping.'''
        ...     return "pong"
        >>> ping.to_spec()["name"]
        'ping'
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def to_openai_spec(self) -> Dict[str, Any]:
        """Return the tool in OpenAI's ``tools`` wire format."""
        return {"type": "function", "function": self.to_spec()}

    def to_anthropic_spec(self) -> Dict[str, Any]:
        """Return the tool in Anthropic's ``tools`` wire format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Alias of :meth:`to_spec` for symmetry with the rest of the API."""
        return self.to_spec()

    # ---- invocation ------------------------------------------------------ #
    def validate(self, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Validate and normalise ``arguments`` for this tool."""
        prepared = validate_arguments(
            self.parameters, dict(arguments or {}), tool_name=self.name, strict=self.strict
        )
        # Drop nulls for optional args so Python defaults apply.
        properties = self.parameters.get("properties") or {}
        return {
            key: value
            for key, value in prepared.items()
            if value is not None
            or key not in properties
            or "null" in _as_type_list(properties[key].get("type"))
        }

    def invoke(self, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        """Execute the tool synchronously and wrap the outcome in a ToolResult."""
        call = ToolCall(name=self.name, arguments=dict(arguments or {}))
        return run_tool_call_sync(self, call)

    async def ainvoke(self, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        """Execute the tool asynchronously and wrap the outcome in a ToolResult."""
        call = ToolCall(name=self.name, arguments=dict(arguments or {}))
        return await run_tool_call(self, call)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly (bypassing validation).

        Async functions return their coroutine; use :meth:`ainvoke` or await the
        result in an event loop.
        """
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<Tool {self.name} args={self.required_arguments}>"


class FunctionTool(Tool[F]):
    """Backwards-compatible explicit subclass of :class:`Tool`."""


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
async def run_tool_call(tool: Tool, call: ToolCall) -> ToolResult:
    """Validate and execute ``call`` against ``tool``, capturing failures.

    When ``tool.output_errors`` is true (the default) failures become error
    observations that the model can react to; otherwise the exception is
    re-raised.
    """
    started = time.perf_counter()
    try:
        arguments = tool.validate(call.arguments)
    except ToolValidationError as exc:
        return ToolResult(
            tool_call_id=call.id,
            name=tool.name,
            content=f"Invalid arguments: {exc}",
            is_error=True,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    try:
        output = await maybe_await(tool.func(**arguments))
    except Exception as exc:
        if not tool.output_errors:
            raise
        return ToolResult(
            tool_call_id=call.id,
            name=tool.name,
            content=f"{type(exc).__name__}: {exc}",
            is_error=True,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    return ToolResult(
        tool_call_id=call.id,
        name=tool.name,
        content=_render_output(output),
        output=output,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def run_tool_call_sync(tool: Tool, call: ToolCall) -> ToolResult:
    """Synchronous variant of :func:`run_tool_call`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_tool_call(tool, call))
    # Already inside an event loop: fall back to a direct, manual execution.
    started = time.perf_counter()
    try:
        arguments = tool.validate(call.arguments)
        output = tool.func(**arguments)
        if inspect.isawaitable(output):  # pragma: no cover - cannot block here
            raise ToolError(
                f"Tool '{tool.name}' is async; call it with `await tool.ainvoke(...)` from async code"
            )
    except Exception as exc:
        if not tool.output_errors:
            raise
        return ToolResult(
            tool_call_id=call.id,
            name=tool.name,
            content=f"{type(exc).__name__}: {exc}",
            is_error=True,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    return ToolResult(
        tool_call_id=call.id,
        name=tool.name,
        content=_render_output(output),
        output=output,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def _render_output(output: Any) -> str:
    """Render a tool's return value as the text the model will see."""
    if output is None:
        return "OK"
    if isinstance(output, str):
        return output
    if isinstance(output, (int, float, bool)):
        return str(output)
    return json_dumps(output)


def _as_type_list(value: Any) -> List[str]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


# --------------------------------------------------------------------------- #
# Toolkit
# --------------------------------------------------------------------------- #
class Toolkit:
    """An ordered collection of tools with lookup and filtering.

    >>> @tool
    ... def ping() -> str:
    ...     '''Ping the server.'''
    ...     return "pong"
    >>> kit = Toolkit([ping])
    >>> "ping" in kit, kit["ping"].description
    (True, 'Ping the server.')
    """

    def __init__(self, tools: Optional[Iterable[Any]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for entry in tools or []:
            self.add(entry)

    # ---- mutation -------------------------------------------------------- #
    def add(self, entry: Any) -> Tool:
        """Add a :class:`Tool`, a callable or a tool spec mapping."""
        tool_obj = ensure_tool(entry)
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def extend(self, entries: Iterable[Any]) -> Toolkit:
        """Add several tools at once and return ``self``."""
        for entry in entries:
            self.add(entry)
        return self

    def remove(self, name: str) -> None:
        """Remove a tool by name (no-op when absent)."""
        self._tools.pop(name, None)

    # ---- access ---------------------------------------------------------- #
    def get(self, name: str) -> Tool:
        """Return a tool by name.

        Raises:
            ToolNotFoundError: when the tool is unknown.
        """
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFoundError(
                f"Unknown tool '{name}'. Available tools: {known}", tool=name
            ) from exc

    def has(self, name: str) -> bool:
        """Return ``True`` when a tool with this name is registered."""
        return name in self._tools

    def filter(self, tags: Iterable[str]) -> Toolkit:
        """Return a new toolkit containing tools matching any of ``tags``."""
        wanted = {tag.lower() for tag in tags}
        return Toolkit(
            tool_obj
            for tool_obj in self._tools.values()
            if wanted & {tag.lower() for tag in tool_obj.tags}
        )

    def names(self) -> List[str]:
        """Sorted tool names."""
        return sorted(self._tools)

    def to_specs(self) -> List[Dict[str, Any]]:
        """Canonical specs for every tool, in insertion order."""
        return [tool_obj.to_spec() for tool_obj in self._tools.values()]

    # ---- dunder ---------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self.get(name)

    def __add__(self, other: Any) -> Toolkit:
        merged = Toolkit(self._tools.values())
        merged.extend(
            other
            if isinstance(other, Iterable) and not isinstance(other, (str, bytes))
            else [other]
        )
        return merged

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Toolkit({', '.join(self.names())})"


def ensure_tool(entry: Any) -> Tool:
    """Convert any supported tool representation into a :class:`Tool`."""
    if isinstance(entry, Tool):
        return entry
    if isinstance(entry, Mapping):
        spec = dict(entry)
        function = spec.pop("function", None)
        if function is None:
            function = spec.pop("func", None)
        if callable(function):
            return Tool(
                function,
                **{
                    k: v
                    for k, v in spec.items()
                    if k in {"name", "description", "parameters", "strict", "output_errors", "tags"}
                },
            )
        name = spec.get("name")
        if not name:
            raise ToolError("Tool spec mapping needs a 'name'")
        return Tool(
            _make_echo(name),
            name=name,
            description=spec.get("description", ""),
            parameters=spec.get("parameters"),
        )
    if callable(entry):
        return Tool(entry)
    raise ToolError(f"Cannot interpret {type(entry)!r} as a tool")


def _make_echo(name: str) -> Callable[..., Any]:
    def _tool(**kwargs: Any) -> Any:  # pragma: no cover - declarative tools
        return f"{name} called with {kwargs}"

    _tool.__name__ = name
    return _tool


@overload
def tool(func: F, /) -> Tool[F]: ...


@overload
def tool(
    func: None = None,
    /,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    output_errors: bool = True,
    tags: Iterable[str] = (),
) -> Callable[[F], Tool[F]]: ...


def tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    output_errors: bool = True,
    tags: Iterable[str] = (),
) -> Union[Tool[F], Callable[[F], Tool[F]]]:
    """Decorator that turns a function into a :class:`Tool`.

    Works bare (``@tool``) or configured (``@tool(name="search", tags=["web"])``).
    """

    def decorate(target: F) -> Tool[F]:
        wrapped = Tool(
            target,
            name=name,
            description=description,
            parameters=parameters,
            strict=strict,
            output_errors=output_errors,
            tags=tags,
        )
        functools.update_wrapper(wrapped, target, updated=())
        return wrapped

    if func is not None:
        return decorate(func)
    return decorate
