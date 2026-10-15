"""Pydantic integration (skipped when pydantic is not installed).

These models live at module scope on purpose: annotations declared under
``from __future__ import annotations`` are resolved lazily, and only names
reachable from the module globals can be resolved.
"""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from pyagents import Agent  # noqa: E402
from pyagents.parsers import AutoParser, PydanticParser  # noqa: E402
from pyagents.testing import text_response  # noqa: E402
from pyagents.tools.base import json_schema_from_function  # noqa: E402


class Address(pydantic.BaseModel):
    city: str
    postcode: int


class Profile(pydantic.BaseModel):
    name: str
    age: int
    address: Address | None = None


def takes_model(address: Address) -> str:
    """Use a pydantic model as a parameter.

    Args:
        address: Where the user lives.
    """
    return address.city


class TestPydanticSchema:
    def test_model_parameter_schema(self):
        schema = json_schema_from_function(takes_model)
        assert schema["properties"]["address"]["properties"]["city"]["type"] == "string"
        assert schema["properties"]["address"]["properties"]["postcode"]["type"] == "integer"
        assert schema["properties"]["address"]["description"] == "Where the user lives."

    def test_tool_from_pydantic_parameter(self):
        from pyagents.tools import tool

        @tool
        def store(address: Address) -> str:
            """Store an address."""
            return address.city

        assert store.parameters["properties"]["address"]["properties"]["city"]["type"] == "string"


class TestPydanticParsing:
    def test_nested_validation(self):
        parser = AutoParser(Profile)
        profile = parser.parse(
            '{"name": "Ada", "age": "36", "address": {"city": "London", "postcode": "1234"}}'
        )
        assert profile.age == 36
        assert profile.address.postcode == 1234

    def test_validation_error_surfaces(self):
        from pyagents.errors import StructuredOutputError

        with pytest.raises(StructuredOutputError, match="Validation failed"):
            PydanticParser(Profile).parse('{"name": "Ada"}')

    def test_rejects_non_models(self):
        from pyagents.errors import StructuredOutputError

        with pytest.raises(StructuredOutputError):
            PydanticParser(dict)  # type: ignore[arg-type]


class TestPydanticAgent:
    def test_structured_output(self, scripted):
        provider = scripted(text_response('```json\n{"name": "Ada", "age": 36}\n```'))
        agent = Agent("extract", provider=provider, output_model=Profile)
        result = agent.run("Who?")
        assert isinstance(result.structured, Profile)
        assert result.structured.name == "Ada"

    def test_format_instructions_include_fields(self, scripted):
        agent = Agent(
            "extract",
            provider=scripted(text_response('{"name": "A", "age": 1}')),
            output_model=Profile,
        )
        assert "name" in agent.system_message().text
