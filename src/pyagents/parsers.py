"""Output parsers for structured responses.

An agent with ``output_model`` needs to turn the model's final message into a
typed Python object. :class:`AutoParser` is the default choice: it asks for
JSON, then attempts progressively more forgiving strategies until something
validates.

    >>> from pyagents.parsers import AutoParser
    >>> import dataclasses
    >>> @dataclasses.dataclass
    ... class Person:
    ...     name: str
    ...     age: int
    >>> parser = AutoParser(Person)
    >>> parser.parse('{"name": "Ada", "age": 36}').name
    'Ada'
    >>> parser.parse('Sure! ```json\\n{"name": "Ada", "age": "36"}\\n```').age
    36
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Pattern, Sequence, Type

from .errors import StructuredOutputError
from .utils import json_dumps

__all__ = [
    "AutoParser",
    "DataclassParser",
    "JsonParser",
    "ListParser",
    "Parser",
    "PydanticParser",
    "RegexParser",
    "parse_json_loose",
    "schema_to_instructions",
]

_FENCE_RE = re.compile(r"```(?:json|python|yaml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_json_loose(text: str) -> Any:
    """Extract the first JSON value from arbitrary model output.

    Handles fenced code blocks, leading prose ("Here you go:") and trailing
    commentary. Returns ``None`` when nothing parseable is found.

    >>> parse_json_loose('Answer: {"a": 1} — done')
    {'a': 1}
    """
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    with contextlib.suppress(TypeError, ValueError):
        return json.loads(candidate)
    for match in _FENCE_RE.finditer(candidate):
        try:
            return json.loads(match.group(1).strip())
        except (TypeError, ValueError):
            continue
    for regex, opener in ((_OBJECT_RE, "{"), (_ARRAY_RE, "[")):
        found = regex.search(candidate)
        if not found:
            continue
        snippet = found.group(0)
        try:
            return json.loads(snippet)
        except (TypeError, ValueError):
            # Trim trailing junk one character at a time.
            for end in range(len(snippet) - 1, 0, -1):
                if snippet[end - 1] not in "}]":
                    continue
                try:
                    return json.loads(snippet[:end])
                except (TypeError, ValueError):
                    continue
            if opener == "[":
                continue
    return None


def schema_to_instructions(schema: Mapping[str, Any], *, name: str = "the response") -> str:
    """Render a JSON schema as plain-language instructions for a model."""
    properties: Mapping[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    lines: List[str] = []
    for field, spec in properties.items():
        field_type = spec.get("type", "any")
        if isinstance(field_type, list):
            field_type = " or ".join(str(item) for item in field_type)
        description = spec.get("description", "")
        marker = "required" if field in required else "optional"
        detail = f"- {field} ({field_type}, {marker})"
        if description:
            detail += f": {description}"
        if "enum" in spec:
            detail += f" — one of {spec['enum']}"
        lines.append(detail)
    header = f"Respond with a single JSON object describing {name}."
    body = "\n".join(lines) if lines else "The object may contain any fields you consider useful."
    return f"{header} Fields:\n{body}\nReturn JSON only, with no extra commentary."


class Parser(ABC):
    """Abstract output parser."""

    #: True when the parser expects a JSON object/array (agents then request JSON).
    expects_json: bool = True

    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse the model's final answer.

        Raises:
            StructuredOutputError: when parsing fails.
        """

    def format_instructions(self) -> str:
        """Instructions appended to the system prompt to steer the format."""
        return "Respond with valid JSON only."

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{type(self).__name__}()"


class JsonParser(Parser):
    """Parse JSON into dictionaries, optionally validating against a type.

    Args:
        target: Optional ``dataclass``, pydantic model or plain type to build.
        keys: Optional list of required top-level keys.
    """

    expects_json = True

    def __init__(
        self, target: Optional[Type[Any]] = None, *, keys: Optional[Sequence[str]] = None
    ) -> None:
        self.target = target
        self.keys = list(keys or [])

    def parse(self, text: str) -> Any:
        """Parse ``text`` as JSON and coerce it into ``target``."""
        data = parse_json_loose(text)
        if data is None:
            raise StructuredOutputError("Model output did not contain JSON", raw=text)
        if isinstance(data, Mapping) and self.keys:
            missing = [key for key in self.keys if key not in data]
            if missing:
                raise StructuredOutputError(
                    f"JSON is missing required key(s): {', '.join(missing)}", raw=text
                )
        return self._coerce(data, text)

    def _coerce(self, data: Any, raw: str) -> Any:
        target = self.target
        if target is None:
            return data
        if not isinstance(data, Mapping):
            raise StructuredOutputError(
                f"Expected a JSON object, got {type(data).__name__}", raw=raw
            )
        prepared = self._coerce_fields(data, raw)
        if _is_pydantic(target):
            try:
                return (
                    target(**prepared)
                    if not hasattr(target, "model_validate")
                    else target.model_validate(prepared)
                )
            except Exception as exc:
                raise StructuredOutputError(
                    f"Validation failed: {exc}", raw=raw, errors=exc
                ) from exc
        if dataclasses.is_dataclass(target):
            try:
                return target(**prepared)
            except Exception as exc:
                raise StructuredOutputError(
                    f"Validation failed: {exc}", raw=raw, errors=exc
                ) from exc
        if isinstance(target, type) and issubclass(target, dict):
            return prepared
        try:  # pydantic v1 style .parse_obj
            parse_obj = target.parse_obj
            return parse_obj(prepared)
        except Exception as exc:
            raise StructuredOutputError(
                f"Cannot coerce JSON into {getattr(target, '__name__', target)}: {exc}", raw=raw
            ) from exc

    def _coerce_fields(self, data: Mapping[str, Any], raw: str) -> Dict[str, Any]:
        """Coerce obvious type mismatches before validation."""
        target = self.target
        hints: Dict[str, Any] = _resolve_hints(target) if dataclasses.is_dataclass(target) else {}
        if _is_pydantic(target):
            fields = getattr(target, "model_fields", None) or getattr(target, "__fields__", {})
            hints = {name: getattr(field, "annotation", Any) for name, field in fields.items()}
        prepared: Dict[str, Any] = dict(data)
        for name, annotation in hints.items():
            if name not in prepared:
                continue
            value = prepared[name]
            if (
                annotation is int
                and isinstance(value, str)
                and re.fullmatch(r"-?\d+", value.strip())
            ):
                prepared[name] = int(value)
            elif annotation is float and isinstance(value, str):
                with contextlib.suppress(ValueError):
                    prepared[name] = float(value)
            elif annotation is bool and isinstance(value, str):
                if value.strip().lower() in {"true", "false"}:
                    prepared[name] = value.strip().lower() == "true"
            elif annotation is str and not isinstance(value, str) and value is not None:
                prepared[name] = str(value)
            elif annotation is list and isinstance(value, str):
                parsed = parse_json_loose(value)
                if isinstance(parsed, list):
                    prepared[name] = parsed
        return prepared

    def format_instructions(self) -> str:
        if dataclasses.is_dataclass(self.target):
            schema = _dataclass_schema_for_instructions(self.target)
            return schema_to_instructions(schema)
        target = self.target
        if _is_pydantic(target):
            try:
                return schema_to_instructions(target.model_json_schema())  # type: ignore[union-attr]
            except Exception:
                pass
        return super().format_instructions()


def _dataclass_schema_for_instructions(cls: type) -> Dict[str, Any]:
    """Build a minimal schema for prompt instructions from a dataclass."""
    from .tools.base import _dataclass_schema

    return _dataclass_schema(cls)


def _resolve_hints(target: Any) -> Dict[str, Any]:
    """Return resolved field annotations for a dataclass-like target.

    Dataclasses declared under ``from __future__ import annotations`` store
    their annotations as strings, so they are resolved here before the value
    coercion rules look at them.
    """
    try:
        from typing import get_type_hints

        return dict(get_type_hints(target))
    except Exception:
        return dict(getattr(target, "__annotations__", {}) or {})


def _is_pydantic(target: Any) -> bool:
    return isinstance(target, type) and (
        hasattr(target, "model_validate") or hasattr(target, "parse_obj")
    )


class PydanticParser(JsonParser):
    """Explicit parser for pydantic models (v1 or v2)."""

    def __init__(self, model: Type[Any]) -> None:
        if not _is_pydantic(model):
            raise StructuredOutputError(f"{model!r} is not a pydantic model")
        super().__init__(model)


class DataclassParser(JsonParser):
    """Explicit parser for dataclasses."""

    def __init__(self, model: Type[Any]) -> None:
        if not dataclasses.is_dataclass(model):
            raise StructuredOutputError(f"{model!r} is not a dataclass")
        super().__init__(model)


class RegexParser(Parser):
    """Extract a capture group from the model's answer.

    Args:
        pattern: Regular expression to search for.
        flags: Compilation flags (case-insensitive by default).
        group: Which capture group to return; ``0`` returns the whole match.

    >>> RegexParser(r"Score:\\s*(\\d+)", group=1).parse("Score: 42")
    '42'
    >>> RegexParser(r"Score\\s*\\d+").parse("Your total — Score 42 points")
    'Score 42'
    """

    expects_json = False

    def __init__(self, pattern: str, *, flags: int = re.IGNORECASE, group: int = 0) -> None:
        self.pattern: Pattern[str] = re.compile(pattern, flags)
        self.group = group

    def parse(self, text: str) -> Any:
        match = self.pattern.search(text or "")
        if not match:
            raise StructuredOutputError(f"Output did not match {self.pattern.pattern!r}", raw=text)
        try:
            return match.group(self.group)
        except IndexError as exc:  # pragma: no cover - misconfiguration
            raise StructuredOutputError(str(exc), raw=text) from exc

    def format_instructions(self) -> str:
        return f"Your answer must match the pattern: {self.pattern.pattern}"


class ListParser(Parser):
    """Parse a JSON array, or fall back to bullet/numbered list extraction."""

    expects_json = True

    def __init__(self, item_type: Optional[Type[Any]] = None, *, min_items: int = 0) -> None:
        self.item_type = item_type
        self.min_items = min_items

    def parse(self, text: str) -> List[Any]:
        data = parse_json_loose(text)
        items: List[Any]
        if isinstance(data, list):
            items = list(data)
        elif isinstance(data, Mapping):
            items = next((value for value in data.values() if isinstance(value, list)), [])
        else:
            items = [
                re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
                for line in (text or "").splitlines()
                if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)
            ]
        if self.item_type is str:
            items = [item if isinstance(item, str) else json_dumps(item) for item in items]
        if len(items) < self.min_items:
            raise StructuredOutputError(
                f"Expected at least {self.min_items} items, got {len(items)}", raw=text
            )
        return items

    def format_instructions(self) -> str:
        return "Respond with a JSON array only."


class AutoParser(Parser):
    """Pick the best parser for a target type.

    ``dict``/``list`` targets and ``None`` use :class:`JsonParser`,
    dataclasses use :class:`DataclassParser`, pydantic models use
    :class:`PydanticParser`.
    """

    expects_json = True

    def __init__(self, target: Optional[Type[Any]] = None, **json_options: Any) -> None:
        self.target = target
        if target is None or target in (dict, Dict):
            self._delegate: Parser = JsonParser(None, **json_options)
        elif dataclasses.is_dataclass(target):
            self._delegate = DataclassParser(target)
        elif _is_pydantic(target):
            self._delegate = PydanticParser(target)
        elif isinstance(target, type) and issubclass(target, list):
            self._delegate = ListParser()
        else:
            self._delegate = JsonParser(target, **json_options)

    def parse(self, text: str) -> Any:
        """Parse using the selected strategy."""
        return self._delegate.parse(text)

    def format_instructions(self) -> str:
        """Return instructions for the selected strategy."""
        return self._delegate.format_instructions()

    @property
    def schema(self) -> Optional[Dict[str, Any]]:
        """JSON schema of the target when one can be derived."""
        target = self.target
        if target is None:
            return None
        if _is_pydantic(target):
            try:
                return dict(target.model_json_schema())
            except Exception:
                return None
        if dataclasses.is_dataclass(target):
            return _dataclass_schema_for_instructions(target)
        return None


def parser_for(target: Any) -> Parser:
    """Return an appropriate parser for ``target`` (a parser passes through)."""
    if isinstance(target, Parser):
        return target
    return AutoParser(target)
