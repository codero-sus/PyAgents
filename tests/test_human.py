"""Tests for the human-in-the-loop :class:`~pyagents.human.HumanAgent`."""

from __future__ import annotations

from typing import List

import pytest

from pyagents.errors import ConfigurationError
from pyagents.human import HumanAgent
from pyagents.memory import BufferMemory
from pyagents.types import EventType, Message, RunStatus


class _FakeInput:
    """A callable that returns scripted answers in order."""

    def __init__(self, answers: List[str]) -> None:
        self.answers = list(answers)
        self.calls: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answers.pop(0)


def test_human_run_uses_scripted_input():
    human = HumanAgent("editor", input=_FakeInput(["Looks good to me."]))
    result = human.run("Review this draft.")
    assert result.output == "Looks good to me."
    assert result.status is RunStatus.COMPLETED
    assert result.agent == "editor"


def test_human_stream_emits_expected_events():
    human = HumanAgent("editor", input=_FakeInput(["Approved."]))
    events = list(human.stream("Review this."))
    types = [event.type for event in events]
    assert EventType.RUN_START in types
    assert EventType.TEXT in types
    assert EventType.MESSAGE in types
    assert EventType.RUN_END in types
    assert events[-1].type is EventType.RUN_END
    assert events[-1].raw is not None
    assert events[-1].raw.output == "Approved."


def test_human_uses_instructions_in_prompt():
    captured: List[str] = []

    def fake_input(prompt: str) -> str:
        captured.append(prompt)
        return "ok"

    human = HumanAgent(
        "editor",
        instructions="Please check the facts carefully.",
        input=fake_input,
    )
    human.run("Is Paris the capital of France?")
    assert any("check the facts carefully" in call for call in captured)
    assert any("Is Paris the capital of France?" in call for call in captured)


def test_human_with_memory_includes_history():
    captured: List[str] = []

    def fake_input(prompt: str) -> str:
        captured.append(prompt)
        return "acknowledged"

    memory = BufferMemory(max_messages=20)
    memory.add(Message.user("First turn"))
    memory.add(Message.assistant("First answer"))

    human = HumanAgent("editor", input=fake_input, memory=memory)
    human.run("Second turn")

    assert any("First turn" in call for call in captured)
    assert any("First answer" in call for call in captured)


def test_human_call_returns_text():
    human = HumanAgent("editor", input=_FakeInput(["yes"]))
    assert human.call("Approve?") == "yes"


def test_human_requires_task():
    human = HumanAgent("editor", input=_FakeInput(["ok"]))
    with pytest.raises(ConfigurationError):
        human.run(None)


def test_human_result_has_run_id_and_duration():
    human = HumanAgent("editor", input=_FakeInput(["done"]))
    result = human.run("Task")
    assert result.run_id.startswith("run_")
    assert result.duration_ms >= 0
    assert result.finished_at is not None


def test_human_can_be_cloned():
    human = HumanAgent("editor", instructions="Check this.")
    clone = human.clone(name="reviewer", instructions="Review that.")
    assert clone.name == "reviewer"
    assert clone.instructions == "Review that."
    assert human.name == "editor"  # original unchanged


def test_human_to_dict_roundtrips():
    human = HumanAgent("editor", instructions="Check facts.")
    data = human.to_dict()
    assert data["name"] == "editor"
    assert data["instructions"] == "Check facts."
    assert data["interactive"] is True
