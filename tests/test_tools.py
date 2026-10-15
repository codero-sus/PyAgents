"""Tests for tool definition, schema inference, validation and built-ins."""

from __future__ import annotations

import asyncio
import dataclasses
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional

import pytest

from pyagents.errors import ConfigurationError, ToolError, ToolNotFoundError, ToolValidationError
from pyagents.tools import (
    Tool,
    Toolkit,
    coerce_arguments,
    default_tools,
    ensure_tool,
    json_schema_from_function,
    tool,
    validate_arguments,
)
from pyagents.tools.base import run_tool_call, run_tool_call_sync
from pyagents.tools.builtin import (
    calculator,
    current_time,
    list_directory,
    python_repl,
    read_file,
    write_file,
)
from pyagents.types import ToolCall


class Colour(Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class Settings:
    name: str
    retries: int = 3


class TestSchemaInference:
    def test_basic_types_and_required(self):
        def fn(a: int, b: str, c: float, d: bool) -> None:
            """Do a thing.

            Args:
                a: first
                b: second
            """

        schema = json_schema_from_function(fn)
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "string"
        assert schema["properties"]["c"]["type"] == "number"
        assert schema["properties"]["d"]["type"] == "boolean"
        assert schema["required"] == ["a", "b", "c", "d"]
        assert schema["properties"]["a"]["description"] == "first"

    def test_defaults_make_arguments_optional(self):
        def fn(a: int, b: str = "x") -> None:
            pass

        schema = json_schema_from_function(fn)
        assert schema["required"] == ["a"]
        assert schema["properties"]["b"]["default"] == "x"

    def test_optional_and_containers(self):
        def fn(items: List[str], mapping: Dict[str, int], maybe: Optional[int] = None) -> None:
            pass

        schema = json_schema_from_function(fn)
        assert schema["properties"]["items"] == {"type": "array", "items": {"type": "string"}}
        assert schema["properties"]["mapping"]["type"] == "object"
        assert schema["properties"]["maybe"]["type"] == ["integer", "null"]

    def test_enum_and_literal(self):
        def fn(colour: Colour, mode: Literal["fast", "slow"]) -> None:
            pass

        schema = json_schema_from_function(fn)
        assert schema["properties"]["colour"]["enum"] == ["red", "blue"]
        assert schema["properties"]["mode"]["enum"] == ["fast", "slow"]

    def test_dataclass_and_varargs(self):
        def fn(settings: Settings, *args, **kwargs) -> None:
            pass

        schema = json_schema_from_function(fn)
        assert schema["properties"]["settings"]["type"] == "object"
        assert schema["properties"]["settings"]["properties"]["name"]["type"] == "string"
        assert "args" in schema["properties"]
        assert schema["additionalProperties"] is True

    def test_annotated_description(self):
        def fn(count: Annotated[int, "how many"]) -> None:
            pass

        schema = json_schema_from_function(fn)
        assert schema["properties"]["count"]["description"] == "how many"


class TestValidation:
    def test_missing_required_raises(self):
        with pytest.raises(ToolValidationError, match="missing required"):
            validate_arguments(
                {"type": "object", "properties": {"a": {}}, "required": ["a"]}, {}, tool_name="t"
            )

    def test_wrong_type_raises(self):
        with pytest.raises(ToolValidationError, match="expected integer"):
            validate_arguments(
                {"type": "object", "properties": {"a": {"type": "integer"}}},
                {"a": "abc"},
                tool_name="t",
            )

    def test_coercion_of_string_values(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "boolean"},
                "c": {"type": "number"},
            },
        }
        assert coerce_arguments(schema, {"a": "5", "b": "true", "c": "1.5"}) == {
            "a": 5,
            "b": True,
            "c": 1.5,
        }

    def test_coercion_of_json_strings_for_containers(self):
        schema = {"type": "object", "properties": {"items": {"type": "array"}}}
        assert coerce_arguments(schema, {"items": "[1, 2]"}) == {"items": [1, 2]}

    def test_enum_validation(self):
        with pytest.raises(ToolValidationError, match="must be one of"):
            validate_arguments(
                {"type": "object", "properties": {"c": {"type": "string", "enum": ["red"]}}},
                {"c": "blue"},
            )

    def test_strict_rejects_unknown_arguments(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ToolValidationError, match="unexpected argument"):
            validate_arguments(schema, {"a": "x", "b": "y"}, strict=True, tool_name="t")
        assert validate_arguments(schema, {"a": "x", "b": "y"}) == {"a": "x", "b": "y"}


class TestToolObject:
    def test_decorator_bare_and_configured(self):
        @tool
        def ping() -> str:
            """Ping the server."""
            return "pong"

        assert isinstance(ping, Tool)
        assert ping.name == "ping"
        assert ping.description == "Ping the server."
        assert ping() == "pong"

        @tool(name="custom", description="Custom desc", tags=["x"])
        def other() -> str:
            return "ok"

        assert other.name == "custom"
        assert other.description == "Custom desc"
        assert other.tags == ("x",)

    def test_bad_name_rejected(self):
        with pytest.raises(ToolError):
            Tool(lambda: None, name="not a name")

    def test_specs(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        assert add.to_spec()["parameters"]["required"] == ["a", "b"]
        assert add.to_openai_spec()["type"] == "function"
        assert add.to_anthropic_spec()["input_schema"] == add.parameters

    def test_invoke_returns_tool_result(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        result = add.invoke({"a": 1, "b": 2})
        assert result.content == "3"
        assert result.output == 3
        assert not result.is_error

    def test_async_tool(self):
        @tool
        async def slow(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        assert asyncio.run(slow.ainvoke({"x": 2})).output == 4

    def test_error_becomes_observation(self):
        @tool
        def boom() -> None:
            """Fail."""
            raise ValueError("nope")

        result = asyncio.run(run_tool_call(boom, ToolCall(name="boom")))
        assert result.is_error
        assert "ValueError: nope" in result.content

    def test_output_errors_false_reraises(self):
        @tool(output_errors=False)
        def boom() -> None:
            """Fail."""
            raise ValueError("nope")

        with pytest.raises(ValueError):
            asyncio.run(run_tool_call(boom, ToolCall(name="boom")))

    def test_validation_failure_is_error_result(self):
        @tool
        def add(a: int) -> int:
            """Add."""
            return a

        result = asyncio.run(run_tool_call(add, ToolCall(name="add", arguments={"a": "abc"})))
        assert result.is_error
        assert "Invalid arguments" in result.content

    def test_sync_invocation_helper(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        assert run_tool_call_sync(add, ToolCall(name="add", arguments={"a": 2, "b": 3})).output == 5

    def test_output_rendering(self):
        @tool
        def returns_none() -> None:
            """Nothing."""

        @tool
        def returns_dict() -> dict:
            """A dict."""
            return {"a": 1}

        assert returns_none.invoke({}).content == "OK"
        assert returns_dict.invoke({}).content == '{"a": 1}'


class TestToolkit:
    def test_add_get_and_membership(self):
        kit = Toolkit([calculator])
        assert "calculator" in kit
        assert kit.get("calculator") is calculator
        assert kit["calculator"].name == "calculator"
        assert kit.names() == ["calculator"]
        with pytest.raises(ToolNotFoundError, match="Unknown tool"):
            kit.get("nope")

    def test_accepts_mappings_and_callables(self):
        def plain(x: int) -> int:
            return x

        kit = Toolkit([plain])
        assert kit.names() == ["plain"]
        kit.add(
            {
                "name": "declared",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            }
        )
        assert kit.get("declared").description == "d"

    def test_remove_filter_and_add(self):
        kit = Toolkit(default_tools())
        kit.remove("http_get")
        assert "http_get" not in kit
        tagged = kit.filter(["math"])
        assert tagged.names() == ["calculator"]
        combined = kit + Toolkit([calculator])
        assert set(combined.names()) == set(kit.names())

    def test_ensure_tool_rejects_garbage(self):
        with pytest.raises(ToolError):
            ensure_tool(42)


class TestBuiltinTools:
    def test_calculator(self):
        assert calculator.invoke({"expression": "(2 + 3) * 4"}).output == "(2 + 3) * 4 = 20"
        assert calculator.invoke({"expression": "sqrt(16)"}).output.endswith("4.0")

    def test_calculator_rejects_code(self):
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        assert result.is_error

    def test_current_time(self):
        output = current_time.invoke({"timezone": "UTC"}).output
        assert "T" in output
        assert current_time.invoke({"timezone": "Nowhere/Nothing"}).output.startswith(
            "Unknown timezone"
        )

    def test_file_tools(self, tmp_path):
        target = tmp_path / "sub" / "note.txt"
        assert write_file.invoke({"path": str(target), "content": "hello"}).content.startswith(
            "Wrote 5"
        )
        assert read_file.invoke({"path": str(target)}).output == "hello"
        assert "sub/" in list_directory.invoke({"path": str(tmp_path)}).output
        assert "note.txt" in list_directory.invoke({"path": str(target.parent)}).output
        assert read_file.invoke({"path": str(tmp_path / "missing.txt")}).output.startswith(
            "File not found"
        )

    def test_python_repl(self):
        assert python_repl.invoke({"code": "print(2 + 2)"}).output == "4"
        assert "NameError" in python_repl.invoke({"code": "raise NameError('x')"}).output

    def test_default_tools_selection(self):
        safe = [item.name for item in default_tools()]
        assert "calculator" in safe
        assert "python_repl" not in safe
        everything = [item.name for item in default_tools(["all"])]
        assert "python_repl" in everything
        assert [item.name for item in default_tools(["calculator"])] == ["calculator"]
        with pytest.raises(ConfigurationError):
            default_tools(["nope"])
