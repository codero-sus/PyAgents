"""Deterministic multi-step workflows.

Teams are about *agents working together*; workflows are about *steps in a
fixed order*, where each step may be an agent, a plain function, or anything
else. Use them when you know the shape of the work up front and want explicit
control flow::

    from pyagents import Agent
    from pyagents.workflows import Workflow

    workflow = Workflow("report")
    workflow.add(research_agent, name="research")
    workflow.add(summarise, name="summary", depends_on=["research"])
    workflow.add(writer_agent, name="write", depends_on=["summary"],
                 input_template="Brief:\\n{summary}\\n\\nWrite the final report.")

    result = workflow.run("The state of open-source LLM tooling")
    print(result.output)

Every node receives a :class:`WorkflowContext` with the original ``task`` and an
``outputs`` mapping of node name to result, so custom nodes can read whatever
they need.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from ._sync import iter_sync, run_sync
from .agent import Agent
from .errors import ConfigurationError, PyAgentsError, attach_run_result
from .types import (
    AgentEvent,
    EventType,
    Message,
    RunResult,
    RunStatus,
    new_id,
    utc_now,
)
from .utils import maybe_await

__all__ = ["Workflow", "WorkflowContext", "WorkflowNode", "WorkflowResult", "node"]

NodeRunner = Union[Agent, Callable[..., Any]]


@dataclass
class WorkflowContext:
    """Execution context handed to every workflow node.

    Attributes:
        task: The original task text.
        outputs: Results of nodes that have already run, keyed by node name.
        shared: Mutable scratch space that survives the whole workflow.
        node: Name of the node currently running.
    """

    task: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)
    shared: Dict[str, Any] = field(default_factory=dict)
    node: str = ""

    def get(self, name: str, default: Any = None) -> Any:
        """Return a previous node's output."""
        return self.outputs.get(name, default)

    def as_text(self) -> str:
        """Render previous outputs as a compact text block."""
        return "\n\n".join(f"## {name}\n{value}" for name, value in self.outputs.items() if value)


@dataclass
class WorkflowNode:
    """A single unit of work inside a :class:`Workflow`."""

    name: str
    runner: NodeRunner
    depends_on: List[str] = field(default_factory=list)
    input_template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Populated after the node runs.
    result: Optional[RunResult] = None

    def prompt(self, context: WorkflowContext) -> str:
        """Build the prompt for an agent node."""
        if self.input_template:
            values = {"task": context.task, **context.outputs, **context.shared}
            try:
                return self.input_template.format(**values)
            except (KeyError, IndexError, ValueError):
                return self.input_template
        if context.outputs:
            return f"{context.task}\n\nContext from earlier steps:\n{context.as_text()}"
        return context.task

    async def execute(self, context: WorkflowContext) -> Any:
        """Run the node against ``context`` and return its output."""
        context.node = self.name
        runner = self.runner
        if isinstance(runner, Agent):
            result = await runner.arun(self.prompt(context))
            self.result = result
            return result.output
        if callable(runner):
            wants_context = _accepts_context(runner)
            payload = context if wants_context else context.task
            output = await maybe_await(runner(payload))
            if isinstance(output, RunResult):
                self.result = output
                return output.output
            return output
        raise ConfigurationError(
            f"Node '{self.name}' has an unsupported runner", runner=type(runner).__name__
        )


def _accepts_context(func: Callable[..., Any]) -> bool:
    """Return ``True`` when ``func`` takes a WorkflowContext (or *args)."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    parameters = list(signature.parameters.values())
    if not parameters:
        return False
    if any(param.kind is param.VAR_KEYWORD for param in parameters):
        return True
    first = parameters[0]
    if first.kind is first.VAR_POSITIONAL:
        return True
    return first.annotation in (WorkflowContext, "WorkflowContext") or first.name in {
        "context",
        "ctx",
    }


@dataclass
class WorkflowResult(RunResult):
    """Result of a workflow run, including every node's output."""

    outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data = super().to_dict()
        data["outputs"] = {key: str(value) for key, value in self.outputs.items()}
        return data


class Workflow:
    """A directed acyclic graph of agents and functions.

    Args:
        name: Workflow name (used in events).
        context: Optional shared mutable dictionary for the run.
        store: Optional run store.
    """

    def __init__(
        self,
        name: str = "workflow",
        *,
        context: Optional[Mapping[str, Any]] = None,
        store: Any = None,
    ) -> None:
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[tuple] = []
        self.shared: Dict[str, Any] = dict(context or {})
        self.store = store
        self.output_node: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def add(
        self,
        runner: NodeRunner,
        *,
        name: Optional[str] = None,
        depends_on: Optional[Sequence[str]] = None,
        input_template: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WorkflowNode:
        """Register a node.

        Args:
            runner: An :class:`~pyagents.agent.Agent` or any callable.
            name: Node name; defaults to the agent/callable name.
            depends_on: Names of nodes that must finish first.
            input_template: Format string used to build an agent's prompt from
                ``{task}`` and previous node outputs.
            metadata: Extra fields stored on the node.

        Returns:
            The created :class:`WorkflowNode`.
        """
        node_name = name or getattr(runner, "name", None) or getattr(runner, "__name__", None)
        if not node_name:
            raise ConfigurationError("Workflow nodes need a name", runner=type(runner).__name__)
        if node_name in self.nodes:
            raise ConfigurationError(f"Duplicate workflow node '{node_name}'")
        node = WorkflowNode(
            name=node_name,
            runner=runner,
            depends_on=list(depends_on or []),
            input_template=input_template,
            metadata=dict(metadata or {}),
        )
        self.nodes[node_name] = node
        for dependency in node.depends_on:
            self.edges.append((dependency, node_name))
        return node

    def set_output(self, name: str) -> Workflow:
        """Mark which node's output becomes the workflow output."""
        if name not in self.nodes:
            raise ConfigurationError(f"Unknown workflow node '{name}'")
        self.output_node = name
        return self

    def mark_output(self, name: str) -> Workflow:
        """Alias of :meth:`set_output`."""
        return self.set_output(name)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def order(self) -> List[str]:
        """Return node names in a valid execution order.

        Raises:
            ConfigurationError: on unknown dependencies or cycles.
        """
        pending = {name: set(node.depends_on) for name, node in self.nodes.items()}
        for name, dependencies in pending.items():
            missing = dependencies - set(self.nodes)
            if missing:
                raise ConfigurationError(
                    f"Node '{name}' depends on unknown node(s): {', '.join(sorted(missing))}"
                )
        ordered: List[str] = []
        while pending:
            ready = sorted(name for name, dependencies in pending.items() if not dependencies)
            if not ready:
                raise ConfigurationError("Workflow has a dependency cycle", nodes=sorted(pending))
            for name in ready:
                ordered.append(name)
                pending.pop(name)
            for dependencies in pending.values():
                dependencies.difference_update(ready)
        return ordered

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(self, task: Union[str, Message] = "") -> WorkflowResult:
        """Execute the workflow synchronously."""
        return run_sync(lambda: self.arun(task))

    async def arun(self, task: Union[str, Message] = "") -> WorkflowResult:
        """Execute the workflow and return a :class:`WorkflowResult`."""
        result: Optional[WorkflowResult] = None
        async for event in self.astream(task):
            if event.type is EventType.RUN_END and isinstance(event.raw, WorkflowResult):
                result = event.raw
        if result is None:  # pragma: no cover - defensive
            raise PyAgentsError("Workflow produced no result")
        return result

    def call(self, task: str, **kwargs: Any) -> str:
        """Run the workflow and return only the final text."""
        return self.run(task, **kwargs).output

    def stream(self, task: Union[str, Message] = "") -> Iterator[AgentEvent]:
        """Iterate workflow events synchronously."""
        return iter_sync(lambda: self.astream(task))

    async def astream(self, task: Union[str, Message] = "") -> AsyncIterator[AgentEvent]:
        """Async generator of workflow events."""
        task_text = task.text if isinstance(task, Message) else str(task)
        workflow_result = WorkflowResult(
            run_id=new_id("flow"),
            status=RunStatus.RUNNING,
            agent=self.name,
            metadata={"task": task_text, "nodes": self.order()},
        )
        context = WorkflowContext(task=task_text, shared=self.shared)
        yield AgentEvent(
            type=EventType.RUN_START,
            agent=self.name,
            run_id=workflow_result.run_id,
            data={"nodes": self.order()},
            text=task_text,
        )
        try:
            for node_name in self.order():
                node = self.nodes[node_name]
                yield AgentEvent(
                    type=EventType.STEP_START,
                    agent=self.name,
                    run_id=workflow_result.run_id,
                    data={"node": node_name},
                )
                if isinstance(node.runner, Agent):
                    agent_result: Optional[RunResult] = None
                    async for event in node.runner.astream(node.prompt(context)):
                        if event.type is EventType.RUN_END:
                            agent_result = event.raw
                            continue
                        yield event
                    output: Any = agent_result.output if agent_result else ""
                    node.result = agent_result
                    if agent_result is not None:
                        workflow_result.usage += agent_result.usage
                        workflow_result.messages.extend(agent_result.messages)
                else:
                    output = await node.execute(context)
                context.outputs[node_name] = output
                workflow_result.outputs[node_name] = output
                yield AgentEvent(
                    type=EventType.MESSAGE,
                    agent=self.name,
                    run_id=workflow_result.run_id,
                    text=_as_text(output),
                    data={"node": node_name, "output": _as_text(output)[:2000]},
                )
        except asyncio.CancelledError:
            workflow_result.status = RunStatus.CANCELLED
            workflow_result.error = "Workflow cancelled"
            workflow_result.finished_at = utc_now()
            if self.store is not None:
                self.store.save(workflow_result, event="cancelled")
            raise
        except PyAgentsError as exc:
            workflow_result.status = RunStatus.FAILED
            workflow_result.error = str(exc)
            workflow_result.finished_at = utc_now()
            attach_run_result(exc, workflow_result)
            if self.store is not None:
                self.store.save(workflow_result, event="failed")
            yield AgentEvent(
                type=EventType.ERROR,
                agent=self.name,
                run_id=workflow_result.run_id,
                data={"error": str(exc)},
            )
            raise

        final_key = self.output_node or self.order()[-1]
        workflow_result.output = _as_text(context.outputs.get(final_key, ""))
        workflow_result.metadata["output_node"] = final_key
        workflow_result.status = RunStatus.COMPLETED
        workflow_result.finished_at = utc_now()
        if self.store is not None:
            self.store.save(workflow_result, event="finished")
        yield AgentEvent(
            type=EventType.RUN_END,
            agent=self.name,
            run_id=workflow_result.run_id,
            text=workflow_result.output,
            data={"status": str(workflow_result.status), "output_node": final_key},
            raw=workflow_result,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the workflow topology."""
        return {
            "name": self.name,
            "nodes": [
                {
                    "name": node.name,
                    "depends_on": node.depends_on,
                    "type": type(node.runner).__name__,
                }
                for node in self.nodes.values()
            ],
            "output_node": self.output_node,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Workflow(name={self.name!r}, nodes={self.order()})"


def _as_text(value: Any) -> str:
    """Render any node output as text."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def node(
    name: Optional[str] = None,
    *,
    depends_on: Optional[Sequence[str]] = None,
    input_template: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that annotates a function for use as a workflow node.

    >>> @node(name="double", depends_on=["start"])
    ... def double(context):
    ...     return "4"
    >>> double.workflow_node["name"]
    'double'
    """
    spec = {
        "name": name,
        "depends_on": list(depends_on or []),
        "input_template": input_template,
    }

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        func.workflow_node = spec  # type: ignore[attr-defined]
        return func

    return decorate
