"""Tests for the core data model."""

from __future__ import annotations

import json

import pytest

from pyagents.errors import RunFailed
from pyagents.types import (
    AgentEvent,
    EventType,
    Handoff,
    LLMResponse,
    Message,
    Role,
    RunResult,
    RunStatus,
    Step,
    ToolCall,
    ToolResult,
    Usage,
    new_id,
)


class TestToolCall:
    def test_from_raw_parses_json(self):
        call = ToolCall.from_raw("add", '{"a": 1, "b": 2}')
        assert call.arguments == {"a": 1, "b": 2}
        assert call.raw_arguments == '{"a": 1, "b": 2}'

    def test_from_raw_tolerates_bad_json(self):
        call = ToolCall.from_raw("add", "{not json")
        assert call.arguments == {}
        assert call.raw_arguments == "{not json"

    def test_from_raw_wraps_scalars(self):
        assert ToolCall.from_raw("echo", '"hello"').arguments == {"value": "hello"}

    def test_round_trip(self):
        call = ToolCall(name="add", arguments={"a": 1}, id="call_1")
        restored = ToolCall.from_dict(call.to_dict())
        assert restored.name == "add"
        assert restored.arguments == {"a": 1}
        assert restored.id == "call_1"


class TestMessage:
    def test_constructors(self):
        assert Message.system("s").role is Role.SYSTEM
        assert Message.user("u").text == "u"
        assert Message.assistant("a", tool_calls=[ToolCall("t")]).tool_calls[0].name == "t"

    def test_user_with_images_builds_blocks(self):
        message = Message.user("look", images=["data:image/png;base64,AAA"])
        assert isinstance(message.content, list)
        assert message.content[1]["image_url"]["url"].startswith("data:")

    def test_tool_result_serialises_payloads(self):
        message = Message.tool_result({"ok": True}, tool_call_id="c1", name="t", is_error=True)
        assert message.content == '{"ok": true}'
        assert message.metadata["is_error"] is True

    def test_coerce_accepts_strings_dicts_and_messages(self):
        assert Message.coerce("hi").role is Role.USER
        assert Message.coerce({"role": "assistant", "content": "yo"}).role is Role.ASSISTANT
        original = Message.user("x")
        assert Message.coerce(original) is original
        with pytest.raises(TypeError):
            Message.coerce(42)  # type: ignore[arg-type]

    def test_text_from_content_blocks(self):
        message = Message(Role.USER, [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        assert message.text == "a\nb"

    def test_copy_is_independent(self):
        message = Message.assistant("hi", tool_calls=[ToolCall("t", {"a": 1})])
        clone = message.copy()
        clone.tool_calls[0].arguments["a"] = 2
        clone.metadata["x"] = 1
        assert message.tool_calls[0].arguments["a"] == 1
        assert "x" not in message.metadata

    def test_round_trip(self):
        message = Message.assistant("hi", tool_calls=[ToolCall("t", {"a": 1})])
        restored = Message.from_dict(message.to_dict())
        assert restored.text == "hi"
        assert restored.tool_calls[0].arguments == {"a": 1}

    def test_to_dict_is_json_serialisable(self):
        message = Message.tool_result("payload", tool_call_id="c1")
        assert json.loads(json.dumps(message.to_dict()))["role"] == "tool"


class TestUsage:
    def test_addition(self):
        total = Usage(1, 2, 3) + Usage(4, 5, 9)
        assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (5, 7, 12)

    def test_inplace_addition(self):
        usage = Usage()
        usage += Usage(2, 3, 5)
        assert usage.total_tokens == 5

    def test_from_dict_infers_total(self):
        assert Usage.from_dict({"prompt_tokens": 2, "completion_tokens": 3}).total_tokens == 5
        assert Usage.from_dict(None).total_tokens == 0


class TestRunResult:
    def _result(self) -> RunResult:
        step = Step(index=1, message=Message.assistant("done"), usage=Usage(1, 1, 2))
        return RunResult(
            output="done",
            status=RunStatus.COMPLETED,
            agent="a",
            messages=[Message.user("hi"), Message.assistant("done")],
            steps=[step],
            usage=Usage(1, 1, 2),
            handoffs=[Handoff("a", "b", "task")],
            finished_at=1.0,
        )

    def test_helpers(self):
        result = self._result()
        assert result.ok
        assert result.iterations == 1
        assert result.last_message().text == "done"
        assert [m.role for m in result.transcript()] == [Role.USER, Role.ASSISTANT]
        assert [m.role for m in result.transcript(include_system=True)] == [
            Role.USER,
            Role.ASSISTANT,
        ]

    def test_raise_for_status(self):
        result = self._result()
        assert result.raise_for_status() is result
        result.status = RunStatus.FAILED
        result.error = "boom"
        with pytest.raises(RunFailed, match="boom"):
            result.raise_for_status()

    def test_serialisation_round_trip(self):
        result = self._result()
        restored = RunResult.from_dict(result.to_dict())
        assert restored.status is RunStatus.COMPLETED
        assert restored.output == "done"
        assert restored.steps[0].message.text == "done"
        assert restored.handoffs[0].to_agent == "b"

    def test_status_accepts_legacy_strings(self):
        data = self._result().to_dict()
        data["status"] = "RunStatus.RUNNING"
        assert RunResult.from_dict(data).status is RunStatus.RUNNING

    def test_tool_call_aggregation(self):
        step = Step(
            index=1,
            message=Message.assistant("", tool_calls=[ToolCall("t", {"x": 1})]),
            tool_results=[ToolResult("t", "t", "ok")],
        )
        result = RunResult(steps=[step])
        assert result.tool_calls[0].name == "t"
        assert result.tool_results[0].content == "ok"


class TestEventsAndIds:
    def test_event_to_dict_is_json_safe(self):
        event = AgentEvent(type=EventType.TEXT, agent="a", text="hi")
        assert json.loads(json.dumps(event.to_dict()))["type"] == "text"

    def test_event_str(self):
        assert str(AgentEvent(type=EventType.TEXT, text="hi")) == "hi"
        assert "run_start" in str(AgentEvent(type=EventType.RUN_START, agent="a"))

    def test_new_id(self):
        assert new_id("run").startswith("run_")
        assert new_id() != new_id()

    def test_llm_response_helpers(self):
        response = LLMResponse(message=Message.assistant("hi", tool_calls=[ToolCall("t")]))
        assert response.text == "hi"
        assert response.tool_calls[0].name == "t"
