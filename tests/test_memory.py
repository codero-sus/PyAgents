"""Tests for memory strategies."""

from __future__ import annotations

import pytest

from pyagents import (
    BufferMemory,
    CompositeMemory,
    Message,
    SummaryMemory,
    TokenWindowMemory,
    VectorMemory,
    WindowMemory,
    memory_from_dict,
)
from pyagents.embeddings import HashEmbedder
from pyagents.errors import ConfigurationError
from pyagents.memory import Memory
from pyagents.testing import ScriptedProvider, text_response


def make_history(count: int) -> list:
    return [Message.user(f"m{index}") for index in range(count)]


class TestBufferMemory:
    def test_keeps_the_last_n(self):
        memory = BufferMemory(max_messages=2)
        memory.extend(make_history(4))
        assert [m.text for m in memory.messages()] == ["m2", "m3"]

    def test_unbounded(self):
        memory = BufferMemory(max_messages=None)
        memory.extend(make_history(50))
        assert len(memory) == 50

    def test_rejects_zero(self):
        with pytest.raises(ConfigurationError):
            BufferMemory(max_messages=0)

    def test_clear(self):
        memory = BufferMemory()
        memory.add(Message.user("x"))
        memory.clear()
        assert memory.messages() == []

    def test_round_trip(self):
        memory = BufferMemory(max_messages=5)
        memory.extend(make_history(3))
        restored = BufferMemory.from_dict(memory.to_dict())
        assert [m.text for m in restored.messages()] == ["m0", "m1", "m2"]
        assert restored.max_messages == 5


class TestWindowMemory:
    def test_pins_system_and_first_user(self):
        memory = WindowMemory(max_messages=3)
        memory.add(Message.system("sys"))
        memory.add(Message.user("first"))
        memory.extend(make_history(5))
        texts = [m.text for m in memory.messages()]
        assert texts[0] == "sys"
        assert texts[1] == "first"
        assert texts[-1] == "m4"
        assert len(texts) == 3

    def test_can_disable_pinning(self):
        memory = WindowMemory(max_messages=2, keep_system=False, keep_first_user=False)
        memory.add(Message.system("sys"))
        memory.extend(make_history(3))
        assert [m.text for m in memory.messages()] == ["m1", "m2"]


class TestTokenWindowMemory:
    def test_respects_budget(self):
        counter = lambda messages: sum(len(m.text) for m in messages)  # noqa: E731
        memory = TokenWindowMemory(max_tokens=5, token_counter=counter)
        memory.extend([Message.user("aaa"), Message.user("bbb"), Message.user("c")])
        assert [m.text for m in memory.messages()] == ["bbb", "c"]

    def test_always_keeps_system(self):
        counter = lambda messages: sum(10 for _ in messages)  # noqa: E731
        memory = TokenWindowMemory(max_tokens=1, token_counter=counter)
        memory.add(Message.system("sys"))
        memory.add(Message.user("big"))
        assert [m.text for m in memory.messages()] == ["sys"]


class TestSummaryMemory:
    def test_summarises_when_threshold_crossed(self):
        provider = ScriptedProvider([text_response("They discussed m0 and m1.")])
        memory = SummaryMemory(provider, summary_threshold=2, keep_recent=2)
        memory.extend(make_history(5))
        texts = [m.text for m in memory.messages()]
        assert texts[0].startswith(SummaryMemory.SUMMARY_PREFIX)
        assert "m0" in texts[0]
        assert texts[-2:] == ["m3", "m4"]

    def test_needs_a_provider(self):
        memory = SummaryMemory(None)
        memory.extend(make_history(5))
        with pytest.raises(ConfigurationError):
            memory.summarize()

    def test_async_summarisation(self):
        import asyncio

        provider = ScriptedProvider([text_response("Summary.")])
        memory = SummaryMemory(provider, summary_threshold=1, keep_recent=1)

        async def run():
            for message in make_history(3):
                await memory.aadd(message)

        asyncio.run(run())
        assert memory.summary

    def test_clear_resets_summary(self):
        memory = SummaryMemory(provider=ScriptedProvider([text_response("s")]))
        memory.summary = "old"
        memory.clear()
        assert memory.summary == ""


class TestVectorMemory:
    def test_retrieves_relevant_history(self):
        embedder = HashEmbedder()
        memory = VectorMemory(embedder, top_k=1, always_keep=0)
        memory.add(Message.user("I love python programming"))
        memory.add(Message.user("the weather in Paris is rainy"))
        memory.set_query("python programming question")
        assert "python" in memory.messages()[0].text

    def test_without_query_returns_recent(self):
        memory = VectorMemory(HashEmbedder(), top_k=2, always_keep=1)
        memory.extend(make_history(5))
        assert [m.text for m in memory.messages()] == ["m3", "m4"]

    def test_requires_an_embedder(self):
        with pytest.raises(ConfigurationError):
            VectorMemory(None)

    def test_always_keeps_recent(self):
        memory = VectorMemory(HashEmbedder(), top_k=1, always_keep=1)
        memory.add(Message.user("python"))
        memory.add(Message.user("weather"))
        memory.set_query("python")
        texts = [m.text for m in memory.messages()]
        assert "weather" in texts  # most recent always included

    def test_async_add(self):
        import asyncio

        memory = VectorMemory(HashEmbedder(dimensions=16), top_k=1, always_keep=1)

        async def run():
            await memory.aadd(Message.user("hello world"))
            return memory.messages()

        assert len(asyncio.run(run())) == 1


class TestCompositeMemory:
    def test_merges_and_deduplicates(self):
        first = BufferMemory(max_messages=2)
        second = BufferMemory(max_messages=1)
        memory = CompositeMemory([first, second])
        memory.extend(make_history(3))
        assert [m.text for m in memory.messages()] == ["m1", "m2"]

    def test_requires_members(self):
        with pytest.raises(ConfigurationError):
            CompositeMemory([])

    def test_round_trip(self):
        memory = CompositeMemory([BufferMemory(max_messages=2), WindowMemory(max_messages=2)])
        memory.add(Message.user("hi"))
        restored = memory_from_dict(memory.to_dict())
        assert isinstance(restored, CompositeMemory)
        assert len(restored.messages()) >= 1


class TestRegistry:
    def test_memory_from_dict_dispatch(self):
        restored = memory_from_dict({"kind": "buffer", "max_messages": 3, "messages": []})
        assert isinstance(restored, BufferMemory)
        assert restored.max_messages == 3

    def test_unknown_kind(self):
        with pytest.raises(ConfigurationError):
            memory_from_dict({"kind": "nope"})

    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            Memory()  # type: ignore[abstract]

    def test_iteration_and_truthiness(self):
        memory = BufferMemory()
        assert bool(memory) is True
        assert list(memory) == []
        assert "BufferMemory" in repr(memory)
