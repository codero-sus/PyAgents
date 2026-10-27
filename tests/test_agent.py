"""Tests for the agent reasoning loop."""

from __future__ import annotations

import dataclasses

import pytest

from pyagents import Agent, BufferMemory, MemoryStore, Role, RunStatus
from pyagents.errors import ConfigurationError, MaxStepsExceeded
from pyagents.testing import ScriptedProvider, text_response, tool_response
from pyagents.tools import Tool, calculator, tool
from pyagents.types import EventType, Message, ToolCall


@tool
def echo(text: str) -> str:
    """Echo text back."""
    return text


@tool
def explode() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


@tool
def double(value: int) -> int:
    """Double a number."""
    return value * 2


class TestBasicRun:
    def test_simple_answer(self, agent):
        result = agent().run("hi")
        assert result.output == "hello"
        assert result.status is RunStatus.COMPLETED
        assert result.ok
        assert [message.role for message in result.messages] == [
            Role.SYSTEM,
            Role.USER,
            Role.ASSISTANT,
        ]
        assert result.usage.total_tokens == 15
        assert result.finished_at is not None
        assert result.metadata["task"] == "hi"

    def test_call_helper(self, agent):
        assert agent().call("hi") == "hello"

    def test_stream_helper(self, agent):
        texts = [event.text for event in agent().stream("hi") if event.type is EventType.TEXT]
        assert "".join(texts) == "hello"

    def test_async_api(self, agent):
        import asyncio

        result = asyncio.run(agent().arun("hi"))
        assert result.output == "hello"

    def test_run_inside_event_loop(self, agent):
        import asyncio

        async def main():
            # run() is sync but must work inside a loop (thread bridge).
            return agent().run("hi").output

        assert asyncio.run(main()) == "hello"

    def test_message_input(self, agent):
        result = agent().run(Message.user("hi"))
        assert result.output == "hello"

    def test_requires_a_task(self):
        with pytest.raises(ConfigurationError):
            Agent("a", provider=ScriptedProvider([text_response("x")])).run()

    def test_context_fills_instructions(self, scripted):
        agent = Agent(
            "a", instructions="You are {persona}.", provider=scripted(text_response("ok"))
        )
        result = agent.run("hi", context={"persona": "a pirate"})
        assert result.messages[0].text == "You are a pirate."

    def test_unknown_placeholder_is_left_alone(self, scripted):
        agent = Agent(
            "a", instructions="You are {persona}.", provider=scripted(text_response("ok"))
        )
        assert agent.run("hi").messages[0].text == "You are {persona}."

    def test_verbose_run(self, scripted, capsys):
        agent = Agent("chatty", provider=scripted(text_response("hi")), verbose=True)
        agent.run("hello")
        captured = capsys.readouterr()
        assert "run started" in captured.out
        assert "run finished" in captured.out

    def test_repr_and_to_dict(self, agent):
        instance = agent(tools=[calculator])
        assert "test-agent" in repr(instance)
        data = instance.to_dict()
        assert data["provider"] == "scripted"
        assert [entry["name"] for entry in data["tools"]] == ["calculator"]

    def test_clone_shares_tools(self, agent):
        instance = agent(tools=[calculator])
        clone = instance.clone("copy")
        assert clone.name == "copy"
        assert clone.toolkit.names() == ["calculator"]


class TestToolLoop:
    def test_tool_call_then_answer(self, scripted):
        provider = scripted(
            tool_response("calculator", {"expression": "2 + 2"}),
            text_response("The answer is 4."),
        )
        agent = Agent("math", provider=provider, tools=[calculator])
        result = agent.run("what is 2+2?")
        assert result.output == "The answer is 4."
        assert len(result.steps) == 2
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_results[0].content == "2 + 2 = 4"
        # The observation is part of the transcript sent to the model.
        assert result.messages[2].tool_calls[0].name == "calculator"
        assert result.messages[3].role is Role.TOOL

    def test_parallel_tool_calls(self, scripted):
        provider = scripted(
            tool_response("double", {"value": 1}),
            text_response("done"),
        )
        agent = Agent("m", provider=provider, tools=[double])
        result = agent.run("go")
        assert result.tool_results[0].output == 2

    def test_multiple_calls_in_one_step(self, scripted):
        from pyagents.testing import response
        from pyagents.types import ToolCall

        provider = scripted(
            response(
                "", tool_calls=[ToolCall("double", {"value": 1}), ToolCall("double", {"value": 2})]
            ),
            text_response("done"),
        )
        agent = Agent("m", provider=provider, tools=[double])
        result = agent.run("go")
        assert len(result.steps[0].tool_results) == 2
        assert result.output == "done"

    def test_unknown_tool_becomes_observation(self, scripted):
        provider = scripted(tool_response("does_not_exist", {}), text_response("recovered"))
        agent = Agent("m", provider=provider, tools=[calculator])
        result = agent.run("go")
        assert result.steps[0].tool_results[0].is_error
        assert "does_not_exist" in result.steps[0].tool_results[0].content
        assert result.output == "recovered"

    def test_failing_tool_becomes_observation(self, scripted):
        provider = scripted(tool_response("explode", {}), text_response("recovered"))
        agent = Agent("m", provider=provider, tools=[explode])
        result = agent.run("go")
        assert result.steps[0].tool_results[0].is_error
        assert "kaboom" in result.steps[0].tool_results[0].content

    def test_tool_specs_are_offered_to_the_model(self, scripted):
        provider = scripted(text_response("done"))
        agent = Agent("m", provider=provider, tools=[calculator])
        agent.run("go")
        assert [spec["name"] for spec in provider.tools_seen[-1]] == ["calculator"]

    def test_no_tools_means_no_tool_payload(self, scripted):
        provider = scripted(text_response("done"))
        Agent("m", provider=provider).run("go")
        assert provider.tools_seen[-1] == []


class TestLimits:
    def test_max_steps(self, scripted):
        provider = ScriptedProvider(
            [tool_response("calculator", {"expression": "1 + 1"})] * 5, default_text=""
        )
        agent = Agent("loop", provider=provider, tools=[calculator], max_steps=3)
        result = agent.run("go")
        assert result.status is RunStatus.MAX_STEPS
        assert not result.ok
        assert len(result.steps) == 3
        assert "maximum" in (result.error or "")

    def test_raise_on_max_steps(self, scripted):
        provider = ScriptedProvider(
            [tool_response("calculator", {"expression": "1 + 1"})] * 5, default_text=""
        )
        agent = Agent(
            "loop", provider=provider, tools=[calculator], max_steps=2, raise_on_max_steps=True
        )
        with pytest.raises(MaxStepsExceeded):
            agent.run("go")

    def test_provider_errors_propagate_with_partial_result(self, scripted):
        from pyagents.errors import ProviderError

        def boom(_messages):
            raise ProviderError("upstream exploded", provider="scripted")

        agent = Agent("m", provider=ScriptedProvider([boom], default_text=""))
        with pytest.raises(ProviderError) as info:
            agent.run("go")
        assert info.value.run_result.status is RunStatus.FAILED

    def test_events_include_error_then_reraise(self, scripted):
        from pyagents.errors import ProviderError

        def boom(_messages):
            raise ProviderError("nope", provider="scripted")

        agent = Agent("m", provider=ScriptedProvider([boom], default_text=""))
        types = []
        with pytest.raises(ProviderError):
            for event in agent.stream("go"):
                types.append(event.type)
        assert EventType.ERROR in types


class TestStructuredOutput:
    @dataclasses.dataclass
    class Person:
        name: str
        age: int

    def test_dataclass_output(self, scripted):
        agent = Agent(
            "x",
            provider=scripted(text_response('{"name": "Ada", "age": "36"}')),
            output_model=self.Person,
        )
        result = agent.run("go")
        assert isinstance(result.structured, self.Person)
        assert result.structured.age == 36
        assert result.metadata["structured"] is True

    def test_repair_attempt(self, scripted):
        agent = Agent(
            "x",
            provider=scripted(
                text_response("not json"), text_response('{"name": "Bob", "age": 1}')
            ),
            output_model=self.Person,
        )
        result = agent.run("go")
        assert result.structured.name == "Bob"

    def test_failure_is_recorded(self, scripted):
        agent = Agent(
            "x",
            provider=scripted(text_response("nope"), text_response("still nope")),
            output_model=self.Person,
        )
        result = agent.run("go")
        assert result.structured is None
        assert result.metadata["structured"] is False
        assert "structured_error" in result.metadata

    def test_instructions_mention_the_format(self, scripted):
        agent = Agent(
            "x", provider=scripted(text_response('{"name":"a","age":1}')), output_model=self.Person
        )
        result = agent.run("go")
        assert "name" in result.messages[0].text

    def test_custom_parser(self, scripted):
        from pyagents.parsers import RegexParser

        agent = Agent(
            "x",
            provider=scripted(text_response("Score: 42")),
            output_parser=RegexParser(r"Score:\s*(\d+)", group=1),
        )
        assert agent.run("go").structured == "42"


class TestMemoryIntegration:
    def test_task_and_answer_are_remembered(self, scripted):
        memory = BufferMemory(max_messages=10)
        agent = Agent(
            "m", provider=scripted(text_response("hi!"), text_response("again")), memory=memory
        )
        agent.run("hello")
        agent.run("you there?")
        assert [(m.role, m.text) for m in memory.messages()] == [
            (Role.USER, "hello"),
            (Role.ASSISTANT, "hi!"),
            (Role.USER, "you there?"),
            (Role.ASSISTANT, "again"),
        ]
        # Second call replays the first exchange before the new question.
        texts = [message.text for message in agent.provider.calls[-1]]
        assert texts[1:3] == ["hello", "hi!"]
        assert texts[-1] == "you there?"

    def test_per_run_memory_override(self, scripted):
        memory = BufferMemory()
        memory.add(Message.user("previous"))
        agent = Agent("m", provider=scripted(text_response("ok")))
        result = agent.run("now", memory=memory)
        assert any(message.text == "previous" for message in result.messages)

    def test_dangling_tool_calls_are_dropped(self, scripted):
        memory = BufferMemory(max_messages=10)
        memory.add(Message.assistant("", tool_calls=[ToolCall("calculator", {})]))
        agent = Agent("m", provider=scripted(text_response("ok")), memory=memory)
        result = agent.run("hi")
        assert all(message.role is not Role.TOOL for message in result.messages)
        assert len(result.messages) == 3

    def test_memory_failures_do_not_break_runs(self, scripted):
        class Broken(BufferMemory):
            def add(self, _message):
                raise RuntimeError("memory offline")

        agent = Agent("m", provider=scripted(text_response("ok")), memory=Broken())
        result = agent.run("hi")
        assert result.output == "ok"
        assert "memory offline" in result.metadata["memory_errors"][0]


class TestPersistenceAndHooks:
    def test_checkpoints_every_step(self, scripted):
        store = MemoryStore()
        provider = scripted(
            tool_response("calculator", {"expression": "1 + 1"}), text_response("two")
        )
        agent = Agent("m", provider=provider, tools=[calculator], store=store)
        result = agent.run("go")
        stored = store.load(result.run_id)
        assert stored.status is RunStatus.COMPLETED
        assert stored.output == "two"
        assert len(stored.steps) == 2

    def test_failed_run_is_stored(self, scripted):
        from pyagents.errors import ProviderError

        store = MemoryStore()

        def boom(_messages):
            raise ProviderError("bad", provider="scripted")

        agent = Agent("m", provider=ScriptedProvider([boom], default_text=""), store=store)
        with pytest.raises(ProviderError):
            agent.run("go")
        assert len(store) == 1
        assert next(iter(store._runs.values())).status is RunStatus.FAILED

    def test_sqlite_store_across_threads(self, scripted, tmp_path):
        from pyagents import SQLiteRunStore

        store = SQLiteRunStore(tmp_path / "runs.db")
        agent = Agent("m", provider=scripted(text_response("stored")), store=store)
        result = agent.run("go")
        assert store.load(result.run_id).output == "stored"
        store.close()

    def test_hooks_run_in_order(self, scripted):
        calls = []

        def before_run(agent, **payload):
            calls.append("before_run")

        def before_model(agent, **payload):
            calls.append("before_model")

        def after_model(agent, response, **payload):
            calls.append("after_model")

        def after_tool(agent, tool, result, **payload):
            calls.append("after_tool")

        def after_run(agent, **payload):
            calls.append("after_run")

        provider = scripted(
            tool_response("calculator", {"expression": "1 + 1"}), text_response("2")
        )
        agent = Agent(
            "m",
            provider=provider,
            tools=[calculator],
            hooks={
                "before_run": before_run,
                "before_model": before_model,
                "after_model": after_model,
                "after_tool": after_tool,
                "after_run": after_run,
            },
        )
        agent.run("go")
        assert calls == [
            "before_run",
            "before_model",
            "after_model",
            "after_tool",
            "before_model",
            "after_model",
            "after_run",
        ]

    def test_after_model_can_replace_the_response(self, scripted):

        def after_model(agent, response, **payload):
            if not response.message.tool_calls:
                return text_response("replaced")
            return None

        agent = Agent(
            "m", provider=scripted(text_response("original")), hooks={"after_model": after_model}
        )
        assert agent.run("go").output == "replaced"

    def test_async_hooks_are_awaited(self, scripted):
        seen = []

        async def before_run(agent, **payload):
            seen.append("async")

        agent = Agent("m", provider=scripted(text_response("ok")), hooks={"before_run": before_run})
        agent.run("go")
        assert seen == ["async"]

    def test_unknown_hook_rejected(self, scripted):
        with pytest.raises(ConfigurationError):
            Agent("m", provider=scripted(), hooks={"nope": lambda *_args, **_kwargs: None})

    def test_events_include_checkpoints(self, scripted):
        store = MemoryStore()
        agent = Agent("m", provider=scripted(text_response("ok")), store=store)
        types = [event.type for event in agent.stream("go")]
        assert EventType.CHECKPOINT in types
        assert types[0] is EventType.RUN_START
        assert types[-1] is EventType.RUN_END

    def test_run_id_can_be_provided(self, scripted):
        agent = Agent("m", provider=scripted(text_response("ok")))
        assert agent.run("go", run_id="custom-1").run_id == "custom-1"


class TestHandoffs:
    def _team(self, scripted):
        worker = Agent(
            "worker", description="Counts things.", provider=scripted(text_response("There are 3."))
        )
        manager = Agent(
            "manager",
            provider=scripted(
                tool_response(
                    "transfer_to_worker", {"task": "count the apples", "reason": "specialist"}
                ),
                text_response("Final: 3 apples."),
            ),
            handoffs={"worker": worker},
        )
        return manager, worker

    def test_handoff_tool_is_exposed(self, scripted):
        manager, _ = self._team(scripted)
        names = [spec["name"] for spec in manager.tool_specs()]
        assert "transfer_to_worker" in names

    def test_handoff_runs_the_target_agent(self, scripted):
        manager, worker = self._team(scripted)
        result = manager.run("how many apples?")
        assert result.output == "Final: 3 apples."
        assert result.steps[0].tool_results[0].content == "There are 3."
        assert [(h.from_agent, h.to_agent) for h in result.handoffs] == [("manager", "worker")]
        assert worker.provider.calls  # the worker really ran

    def test_handoff_events(self, scripted):
        manager, _ = self._team(scripted)
        types = [event.type for event in manager.stream("how many apples?")]
        assert EventType.HANDOFF in types
        assert EventType.AGENT_START in types
        assert EventType.AGENT_END in types

    def test_handoff_failure_is_an_error_observation(self, scripted):
        from pyagents.errors import ProviderError

        def boom(_messages):
            raise ProviderError("worker down", provider="scripted")

        worker = Agent("worker", provider=ScriptedProvider([boom], default_text=""))
        manager = Agent(
            "manager",
            provider=scripted(
                tool_response("transfer_to_worker", {"task": "x"}), text_response("I'll handle it")
            ),
            handoffs={"worker": worker},
        )
        result = manager.run("go")
        assert result.steps[0].tool_results[0].is_error
        assert result.output == "I'll handle it"

    def test_cycles_are_rejected(self, scripted):
        a = Agent("a", provider=scripted())
        b = Agent("b", provider=scripted(), handoffs={"a": a})
        a.handoffs["b"] = b
        with pytest.raises(ConfigurationError, match="cycle"):
            a._validate_handoffs()


class TestComposition:
    def test_as_tool(self, scripted):
        inner = Agent(
            "summariser", description="Summarises text.", provider=scripted(text_response("short"))
        )
        outer = Agent(
            "writer",
            provider=scripted(
                tool_response("summariser", {"task": "summarise this"}), text_response("final")
            ),
        )
        outer.add_tool(inner.as_tool())
        result = outer.run("please summarise")
        assert result.output == "final"
        assert result.steps[0].tool_results[0].content == "short"

    def test_tool_instances_are_accepted(self, scripted):
        agent = Agent(
            "m", provider=scripted(text_response("ok")), tools=[Tool(echo.func, name="renamed")]
        )
        assert [spec["name"] for spec in agent.tool_specs()] == ["renamed"]
