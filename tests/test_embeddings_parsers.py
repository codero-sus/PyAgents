"""Tests for embeddings and the parsers."""

from __future__ import annotations

import dataclasses

import pytest

from pyagents.embeddings import (
    HashEmbedder,
    OpenAIEmbedder,
    TfidfEmbedder,
    cosine_similarity,
    dot_product,
    embeddings_from_config,
    normalize,
    top_k_indices,
)
from pyagents.errors import ConfigurationError, ProviderError, StructuredOutputError
from pyagents.parsers import (
    AutoParser,
    DataclassParser,
    JsonParser,
    ListParser,
    RegexParser,
    parse_json_loose,
    parser_for,
    schema_to_instructions,
)
from pyagents.testing import FakeTransport


class TestVectorMaths:
    def test_dot_and_normalise(self):
        assert dot_product([1, 2], [3, 4]) == 11
        assert normalize([3, 4]) == [0.6, 0.8]
        assert normalize([0, 0]) == [0, 0]

    def test_cosine(self):
        assert cosine_similarity([1, 0], [1, 0]) == 1
        assert cosine_similarity([1, 0], [0, 1]) == 0
        assert cosine_similarity([], [1, 0]) == 0

    def test_top_k(self):
        ranked = top_k_indices([1, 0], [[0, 1], [1, 0], [0.9, 0.1]], k=2)
        assert [index for index, _ in ranked] == [1, 2]


class TestHashEmbedder:
    def test_deterministic_and_normalised(self):
        embedder = HashEmbedder(dimensions=64)
        first = embedder.embed(["hello world"])[0]
        second = HashEmbedder(dimensions=64).embed(["hello world"])[0]
        assert first == second
        assert len(first) == 64
        assert abs(sum(value * value for value in first) - 1) < 1e-9

    def test_similar_texts_score_higher(self):
        embedder = HashEmbedder()
        texts = ["the cat sat on the mat", "the dog sat on the log", "quantum chromodynamics"]
        vectors = embedder.embed(texts)
        assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(vectors[0], vectors[2])

    def test_embed_one_and_config(self):
        assert HashEmbedder(dimensions=8).embed_one("x")
        assert embeddings_from_config({"type": "hash", "dimensions": 16}).dimensions == 16
        assert embeddings_from_config(None) is None
        with pytest.raises(ConfigurationError):
            embeddings_from_config({"type": "nope"})

    def test_validation(self):
        with pytest.raises(ConfigurationError):
            HashEmbedder(dimensions=0)


class TestTfidfEmbedder:
    def test_fit_then_embed(self):
        embedder = TfidfEmbedder().fit(["the cat sat", "the dog barked"])
        vectors = embedder.embed(["cat", "dog"])
        assert len(vectors[0]) == embedder.dimensions
        # "cat" is closer to the first document than to the second one.
        documents = embedder.embed(["the cat sat", "the dog barked"])
        assert cosine_similarity(vectors[0], documents[0]) > cosine_similarity(
            vectors[0], documents[1]
        )

    def test_unfitted_raises(self):
        with pytest.raises(ConfigurationError):
            TfidfEmbedder().embed(["x"])

    def test_fit_transform(self):
        vectors = TfidfEmbedder().fit_transform(["a b", "b c"])
        assert len(vectors) == 2


class TestOpenAIEmbedder:
    def test_uses_transport(self):
        transport = FakeTransport(
            [
                {
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": 1, "embedding": [0.3, 0.4]},
                    ]
                }
            ]
        )
        embedder = OpenAIEmbedder(api_key="sk-test", transport=transport)
        vectors = embedder.embed(["a", "b"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert transport.last_payload["model"] == "text-embedding-3-small"

    def test_requires_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            OpenAIEmbedder()

    def test_count_mismatch_raises(self):
        transport = FakeTransport([{"data": [{"index": 0, "embedding": [0.1]}]}])
        with pytest.raises(ProviderError):
            OpenAIEmbedder(api_key="k", transport=transport).embed(["a", "b"])

    def test_async_embed(self):
        import asyncio

        transport = FakeTransport([{"data": [{"index": 0, "embedding": [1.0]}]}])
        embedder = OpenAIEmbedder(api_key="k", transport=transport)
        assert asyncio.run(embedder.aembed(["a"])) == [[1.0]]


class TestParseJsonLoose:
    def test_plain_json(self):
        assert parse_json_loose('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert parse_json_loose('Here:\n```json\n{"a": 1}\n```\nDone') == {"a": 1}

    def test_embedded(self):
        assert parse_json_loose('Sure! {"a": 1} — hope that helps') == {"a": 1}
        assert parse_json_loose("Results: [1, 2, 3].") == [1, 2, 3]

    def test_nothing(self):
        assert parse_json_loose("no json at all") is None
        assert parse_json_loose("") is None


@dataclasses.dataclass
class Person:
    name: str
    age: int


class TestParsers:
    def test_json_parser(self):
        parser = JsonParser(keys=["a"])
        assert parser.parse('{"a": 1}') == {"a": 1}
        with pytest.raises(StructuredOutputError, match="missing required key"):
            parser.parse('{"b": 1}')
        with pytest.raises(StructuredOutputError, match="did not contain JSON"):
            parser.parse("nope")

    def test_json_parser_rejects_non_objects_for_typed_targets(self):
        with pytest.raises(StructuredOutputError, match="Expected a JSON object"):
            JsonParser(Person).parse("[1, 2]")

    def test_dataclass_parser_coerces(self):
        person = DataclassParser(Person).parse('{"name": "Ada", "age": "36"}')
        assert person == Person(name="Ada", age=36)

    def test_dataclass_parser_rejects_non_dataclass(self):
        with pytest.raises(StructuredOutputError):
            DataclassParser(dict)  # type: ignore[arg-type]

    def test_validation_error_surfaces(self):
        with pytest.raises(StructuredOutputError, match="Validation failed"):
            JsonParser(Person).parse('{"name": "Ada"}')

    def test_regex_parser(self):
        parser = RegexParser(r"Score:\s*(\d+)", group=1)
        assert parser.parse("Score: 42") == "42"
        assert "pattern" in parser.format_instructions()
        with pytest.raises(StructuredOutputError):
            parser.parse("nothing")

    def test_list_parser(self):
        assert ListParser().parse('["a", "b"]') == ["a", "b"]
        assert ListParser().parse("- one\n- two\n") == ["one", "two"]
        assert ListParser().parse('{"items": [1, 2]}') == [1, 2]
        assert ListParser(str).parse("[1, 2]") == ["1", "2"]
        with pytest.raises(StructuredOutputError):
            ListParser(min_items=3).parse('["a"]')

    def test_auto_parser_dispatch(self):
        assert isinstance(AutoParser(None)._delegate, JsonParser)
        assert isinstance(AutoParser(Person)._delegate, DataclassParser)
        assert isinstance(AutoParser(list)._delegate, ListParser)

    def test_auto_parser_schema(self):
        schema = AutoParser(Person).schema
        assert schema is not None
        assert schema["properties"]["age"]["type"] == "integer"

    def test_parser_for_passthrough(self):
        parser = RegexParser("x")
        assert parser_for(parser) is parser

    def test_schema_to_instructions(self):
        text = schema_to_instructions(
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        )
        assert "a (string, required)" in text

    def test_format_instructions_default(self):
        assert "JSON" in JsonParser().format_instructions()
        assert (
            JsonParser(Person).format_instructions().startswith("Respond with a single JSON object")
        )
