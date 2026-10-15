"""Tests for multi-agent teams and workflows."""

from __future__ import annotations

import asyncio

import pytest

from pyagents import Agent, BufferMemory, MemoryStore, RunStatus, Team, Workflow
from pyagents.errors import ConfigurationError, OrchestrationError
from pyagents.teams import TeamResult, parallel, sequential
from pyagents.testing import ScriptedProvider, text_response, tool_response
from pyagents.types import EventType
from pyagents.workflows import WorkflowContext, WorkflowResult, node


def make_agent(name: str, *texts: str, **kwargs) -> Agent:
    responses = [text_response(text) for text in texts] or [text_response(f"{name} done")]
    kwargs.setdefault("provider", ScriptedProvider(responses))
    return Agent(name, **kwargs)


class TestSequentialTeam:
    def test_pipeline_passes_context_forward(self):
        first = make_agent("first", "first answer")
        second = make_agent("second", "second answer")
        team = sequential(first, second)
        result = team.run("do the thing")
        assert isinstance(result, TeamResult)
        assert result.output == "second answer"
        assert result.mode == "sequential"
        assert [member.agent for member in result.agent_results] == ["first", "second"]
        assert result.outputs == {"first": "first answer", "second": "second answer"}
        # The second agent saw the first agent's output.
        assert "first answer" in second.provider.calls[-1][-1].text

    def test_without_context_passing(self):
        first = make_agent("first", "first answer")
        second = make_agent("second", "second answer")
        team = sequential(first, second, pass_context=False)
        team.run("original task")
        assert second.provider.calls[-1][-1].text == "original task"

    def test_multiple_turns(self):
        agent = make_agent("a", "one", "two")
        result = sequential(agent, turns=2).run("task")
        assert len(result.agent_results) == 2
        assert result.output == "two"

    def test_handoffs_are_recorded(self):
        result = sequential(make_agent("a", "x"), make_agent("b", "y")).run("task")
        assert [(handoff.from_agent, handoff.to_agent) for handoff in result.handoffs] == [
            ("a", "b")
        ]

    def test_usage_is_aggregated(self):
        result = sequential(make_agent("a", "x"), make_agent("b", "y")).run("task")
        assert result.usage.total_tokens == 30

    def test_events_include_agent_boundaries(self):
        team = sequential(make_agent("a", "x"), make_agent("b", "y"))
        types = [event.type for event in team.stream("task")]
        assert types[0] is EventType.RUN_START
        assert types[-1] is EventType.RUN_END
        assert types.count(EventType.AGENT_START) == 2
        assert types.count(EventType.AGENT_END) == 2

    def test_shared_memory(self):
        memory = BufferMemory(max_messages=50)
        team = sequential(
            make_agent("a", "x", memory=memory), make_agent("b", "y", memory=memory), memory=memory
        )
        team.run("task")
        assert any(message.text == "task" for message in memory.messages())

    def test_async_run(self):
        team = sequential(make_agent("a", "x"))
        assert asyncio.run(team.arun("task")).output == "x"

    def test_call_helper(self):
        assert sequential(make_agent("a", "x")).call("task") == "x"

    def test_requires_agents(self):
        with pytest.raises(ConfigurationError):
            Team([], mode="sequential")

    def test_unknown_mode(self):
        with pytest.raises(ConfigurationError):
            Team([make_agent("a")], mode="telepathy")

    def test_store_persists(self):
        store = MemoryStore()
        result = sequential(make_agent("a", "x"), store=store).run("task")
        assert store.load(result.run_id).output == "x"

    def test_repr_and_dict(self):
        team = sequential(make_agent("a"))
        assert "sequential" in repr(team)
        assert team.to_dict()["agents"] == ["a"]


class TestParallelTeam:
    def test_all_agents_run_and_outputs_are_joined(self):
        team = parallel(make_agent("a", "Alpha"), make_agent("b", "Beta"))
        result = team.run("task")
        assert result.outputs == {"a": "Alpha", "b": "Beta"}
        assert "## a" in result.output and "Alpha" in result.output
        assert "## b" in result.output and "Beta" in result.output

    def test_aggregator_synthesises(self):
        aggregator = make_agent("chair", "Synthesis complete.")
        team = parallel(make_agent("a", "Alpha"), make_agent("b", "Beta"), aggregator=aggregator)
        result = team.run("task")
        assert result.output == "Synthesis complete."
        assert "Alpha" in aggregator.provider.calls[-1][-1].text
        assert [member.agent for member in result.agent_results] == ["a", "b", "chair"]

    def test_member_failure_is_recorded(self):
        from pyagents.errors import ProviderError

        def boom(_messages):
            raise ProviderError("down", provider="scripted")

        broken = Agent("broken", provider=ScriptedProvider([boom], default_text=""))
        team = parallel(make_agent("ok", "fine"), broken)
        result = team.run("task")
        assert result.status is RunStatus.COMPLETED
        assert "broken" in result.metadata["errors"]
        assert "fine" in result.output

    def test_events_pass_through_with_agent_attribution(self):
        team = parallel(make_agent("a", "Alpha"), make_agent("b", "Beta"))
        agents = {event.agent for event in team.stream("task") if event.type is EventType.TEXT}
        assert agents == {"a", "b"}

    def test_concurrency_cap_is_honoured(self):
        team = parallel(make_agent("a"), make_agent("b"), max_concurrency=1)
        assert team.max_concurrency == 1
        assert team.run("task").status is RunStatus.COMPLETED


class TestOrchestratorTeam:
    def _team(self, **kwargs):
        manager = Agent(
            "manager",
            provider=ScriptedProvider(
                [
                    tool_response(
                        "transfer_to_specialist", {"task": "count apples", "reason": "expert"}
                    ),
                    text_response("Final answer: 3 apples."),
                ]
            ),
        )
        specialist = make_agent("specialist", "There are 3 apples.")
        return Team([manager, specialist], mode="orchestrator", **kwargs), manager, specialist

    def test_manager_delegates(self):
        team, manager, specialist = self._team()
        result = team.run("how many apples?")
        assert result.output == "Final answer: 3 apples."
        assert "transfer_to_specialist" in [spec["name"] for spec in manager.tool_specs()]
        assert specialist.provider.calls

    def test_member_results_include_worker_run(self):
        team, _, _ = self._team()
        result = team.run("how many apples?")
        assert [member.agent for member in result.agent_results] == ["manager", "specialist"]
        assert result.handoffs[0].to_agent == "specialist"

    def test_explicit_orchestrator(self):
        manager = Agent("boss", provider=ScriptedProvider([text_response("I'll answer myself.")]))
        worker = make_agent("worker")
        team = Team([worker], mode="orchestrator", orchestrator=manager)
        result = team.run("task")
        assert result.output == "I'll answer myself."
        assert "worker" in manager.handoffs

    def test_needs_workers(self):
        with pytest.raises(OrchestrationError):
            Team([make_agent("only")], mode="orchestrator")

    def test_failure_propagates_with_result(self):
        from pyagents.errors import ProviderError

        def boom(_messages):
            raise ProviderError("manager down", provider="scripted")

        manager = Agent("manager", provider=ScriptedProvider([boom], default_text=""))
        team = Team([manager, make_agent("worker")], mode="orchestrator")
        with pytest.raises(ProviderError) as info:
            team.run("task")
        assert info.value.run_result.status is RunStatus.FAILED


class TestWorkflow:
    def test_linear_pipeline(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("draft", "draft text"), name="draft")
        workflow.add(make_agent("polish", "polished"), name="polish", depends_on=["draft"])
        result = workflow.run("write something")
        assert isinstance(result, WorkflowResult)
        assert result.output == "polished"
        assert result.outputs == {"draft": "draft text", "polish": "polished"}
        assert result.status is RunStatus.COMPLETED

    def test_input_template_is_formatted(self):
        writer = make_agent("writer", "final")
        workflow = Workflow("flow")
        workflow.add(make_agent("draft", "the draft"), name="draft")
        workflow.add(
            writer,
            name="writer",
            depends_on=["draft"],
            input_template="Task: {task}\nDraft: {draft}",
        )
        workflow.run("do it")
        assert writer.provider.calls[-1][-1].text == "Task: do it\nDraft: the draft"

    def test_function_nodes_receive_context(self):
        seen = {}

        def summarise(context: WorkflowContext) -> str:
            seen["task"] = context.task
            seen["draft"] = context.get("draft")
            context.shared["marker"] = True
            return "summary"

        def plain(task) -> str:
            return f"plain:{task}"

        workflow = Workflow("flow")
        workflow.add(make_agent("draft", "draft"), name="draft")
        workflow.add(summarise, name="summary", depends_on=["draft"])
        workflow.add(plain, name="plain", depends_on=["summary"])
        result = workflow.run("task text")
        assert seen == {"task": "task text", "draft": "draft"}
        assert result.outputs["summary"] == "summary"
        assert result.outputs["plain"] == "plain:task text"

    def test_async_function_nodes(self):
        async def double(context: WorkflowContext) -> str:
            await asyncio.sleep(0)
            return context.task * 2

        workflow = Workflow("flow")
        workflow.add(double, name="double")
        assert workflow.run("ab").output == "abab"

    def test_node_returning_run_result(self):
        from pyagents.types import RunResult

        def returns_result(task) -> RunResult:
            return RunResult(output="from result", status=RunStatus.COMPLETED)

        workflow = Workflow("flow")
        workflow.add(returns_result, name="node")
        assert workflow.run("x").output == "from result"

    def test_output_node_selection(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("first", "one"), name="first")
        workflow.add(make_agent("second", "two"), name="second", depends_on=["first"])
        workflow.set_output("first")
        assert workflow.run("x").output == "one"

    def test_unknown_output_node(self):
        with pytest.raises(ConfigurationError):
            Workflow("flow").set_output("nope")

    def test_unknown_dependency(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("a"), name="a", depends_on=["ghost"])
        with pytest.raises(ConfigurationError, match="unknown node"):
            workflow.order()

    def test_cycle_detection(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("a"), name="a", depends_on=["b"])
        workflow.add(make_agent("b"), name="b", depends_on=["a"])
        with pytest.raises(ConfigurationError, match="cycle"):
            workflow.order()

    def test_duplicate_names_and_unnamed_nodes(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("a"), name="a")
        with pytest.raises(ConfigurationError, match="Duplicate"):
            workflow.add(make_agent("a2"), name="a")

    def test_events_and_streaming(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("a", "one"), name="a")
        events = list(workflow.stream("task"))
        assert events[0].type is EventType.RUN_START
        assert events[-1].type is EventType.RUN_END
        assert any(
            event.type is EventType.MESSAGE and event.data.get("node") == "a" for event in events
        )

    def test_agent_events_are_forwarded(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("a", "one"), name="a")
        types = [event.type for event in workflow.stream("task")]
        assert EventType.TEXT in types  # the agent's tokens stream through

    def test_store_and_shared_context(self):
        store = MemoryStore()
        workflow = Workflow("flow", context={"brand": "Acme"}, store=store)

        def read_shared(context: WorkflowContext) -> str:
            return context.shared["brand"]

        workflow.add(read_shared, name="brand")
        result = workflow.run("x")
        assert result.outputs["brand"] == "Acme"
        assert store.load(result.run_id).output == "Acme"

    def test_node_decorator_attaches_spec(self):
        @node(name="custom", depends_on=["start"])
        def fn(context):
            return "x"

        assert fn.workflow_node["name"] == "custom"
        assert fn.workflow_node["depends_on"] == ["start"]

    def test_failure_is_reported(self):
        from pyagents.errors import ProviderError

        def boom(_messages):
            raise ProviderError("node failed", provider="scripted")

        workflow = Workflow("flow")
        workflow.add(
            Agent("broken", provider=ScriptedProvider([boom], default_text="")), name="broken"
        )
        with pytest.raises(ProviderError) as info:
            workflow.run("x")
        assert info.value.run_result.status is RunStatus.FAILED

    def test_unsupported_runner(self):
        workflow = Workflow("flow")
        workflow.add(42, name="bad")  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            workflow.run("x")

    def test_repr_and_dict(self):
        workflow = Workflow("flow")
        workflow.add(make_agent("a"), name="a")
        assert "flow" in repr(workflow)
        assert workflow.to_dict()["nodes"][0]["name"] == "a"

    def test_agent_as_tool_composition(self):
        """An agent exposed as a tool can be used by another agent."""
        expert = make_agent("expert", "expert answer")
        caller = Agent(
            "caller",
            provider=ScriptedProvider(
                [tool_response("expert", {"task": "help"}), text_response("wrapped")]
            ),
        )
        caller.add_tool(expert.as_tool())
        assert caller.run("go").steps[0].tool_results[0].content == "expert answer"
