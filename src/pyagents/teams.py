"""Multi-agent teams: sequential, parallel and orchestrator patterns.

A team coordinates several :class:`~pyagents.agent.Agent` objects and returns a
single :class:`TeamResult` that contains every member's run, so you always keep
the full audit trail::

    from pyagents import Agent, Team

    researcher = Agent("researcher", instructions="Find primary-source facts.")
    writer = Agent("writer", instructions="Turn notes into crisp prose.")

    team = Team([researcher, writer], mode="sequential")
    result = team.run("Explain how HTTPS works")
    print(result.output)
    print(result.agent_results[0].output[:80])

Patterns:

``sequential``
    Each agent receives the previous agent's answer as context (a pipeline).
``parallel``
    Every agent works on the same task at once; results are concatenated or
    summarised by an optional aggregator agent.
``orchestrator``
    A manager agent decides who works on what, delegating through generated
    ``transfer_to_*`` handoff tools.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from ._sync import iter_sync, run_sync
from .agent import Agent
from .errors import ConfigurationError, OrchestrationError, PyAgentsError, attach_run_result
from .memory import Memory
from .stores import RunStore
from .types import (
    AgentEvent,
    EventType,
    Handoff,
    Message,
    RunResult,
    RunStatus,
    new_id,
    utc_now,
)

__all__ = ["Team", "TeamMode", "TeamResult", "parallel", "sequential"]

MODES = ("sequential", "parallel", "orchestrator")
TeamMode = str


@dataclass
class TeamResult(RunResult):
    """A :class:`~pyagents.types.RunResult` extended with per-agent results."""

    mode: str = "sequential"
    agent_results: List[RunResult] = field(default_factory=list)

    @property
    def outputs(self) -> Dict[str, str]:
        """Mapping of agent name to that agent's final answer."""
        return {result.agent: result.output for result in self.agent_results}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation, members included."""
        data = super().to_dict()
        data["mode"] = self.mode
        data["agent_results"] = [result.to_dict() for result in self.agent_results]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TeamResult:
        """Rebuild a team result from :meth:`to_dict` output."""
        base = RunResult.from_dict(data)
        return cls(
            run_id=base.run_id,
            output=base.output,
            status=base.status,
            agent=base.agent,
            messages=base.messages,
            steps=base.steps,
            usage=base.usage,
            handoffs=base.handoffs,
            error=base.error,
            structured=base.structured,
            metadata=base.metadata,
            started_at=base.started_at,
            finished_at=base.finished_at,
            mode=data.get("mode", "sequential"),
            agent_results=[RunResult.from_dict(item) for item in data.get("agent_results") or []],
        )


class Team:
    """Coordinate multiple agents on one task.

    Args:
        agents: Member agents (at least one). For ``orchestrator`` mode the first
            agent is the manager unless ``orchestrator`` is given explicitly.
        mode: ``"sequential"``, ``"parallel"`` or ``"orchestrator"``.
        name: Team name used in events and the result.
        orchestrator: Manager agent for orchestrator mode.
        aggregator: Agent that merges parallel outputs; without it the outputs
            are concatenated.
        turns: Passes over the member list in sequential mode (default 1).
        pass_context: Forward previous outputs to the next agent (sequential).
        max_concurrency: Cap on concurrent agents in parallel mode.
        memory: Optional shared memory passed to every member run.
        store: Optional run store for checkpointing the team result.
        verbose: Print agent boundaries and text as they happen.
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        mode: str = "sequential",
        name: str = "team",
        orchestrator: Optional[Agent] = None,
        aggregator: Optional[Agent] = None,
        turns: int = 1,
        pass_context: bool = True,
        max_concurrency: int = 4,
        memory: Optional[Memory] = None,
        store: Optional[RunStore] = None,
        verbose: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ConfigurationError(f"Unknown team mode '{mode}'", known=list(MODES))
        if not agents and not orchestrator:
            raise ConfigurationError("A team needs at least one agent")
        self.mode = mode
        self.name = name
        self.agents: List[Agent] = list(agents)
        self.orchestrator = orchestrator
        self.aggregator = aggregator
        self.turns = max(1, int(turns))
        self.pass_context = pass_context
        self.max_concurrency = max(1, int(max_concurrency))
        self.memory = memory
        self.store = store
        self.verbose = verbose
        if mode == "orchestrator":
            self._configure_orchestrator()

    # ------------------------------------------------------------------ #
    # Orchestrator wiring
    # ------------------------------------------------------------------ #
    def _configure_orchestrator(self) -> None:
        """Give the manager a handoff tool per worker."""
        manager = self.orchestrator or (self.agents[0] if self.agents else None)
        workers = self.agents if self.orchestrator is not None else self.agents[1:]
        if manager is None:
            raise OrchestrationError("Orchestrator mode needs a manager agent")
        if not workers:
            raise OrchestrationError("Orchestrator mode needs at least one worker agent")
        for worker in workers:
            manager.add_handoff(worker.name, worker)
        self.orchestrator = manager
        self.workers = workers
        manager.description = manager.description or f"Coordinates the {self.name} team"
        manager.instructions = (
            f"{manager.instructions}\n\n"
            "You lead a team of specialists. Decide which specialist to delegate to using the "
            "transfer_to_* tools, then synthesise their findings into the final answer. "
            "Delegate only when it genuinely helps; you may also answer directly."
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(
        self, task: Union[str, Message], *, context: Optional[Mapping[str, Any]] = None
    ) -> TeamResult:
        """Execute the team synchronously."""
        return run_sync(lambda: self.arun(task, context=context))

    async def arun(
        self, task: Union[str, Message], *, context: Optional[Mapping[str, Any]] = None
    ) -> TeamResult:
        """Execute the team and return a :class:`TeamResult`."""
        result: Optional[TeamResult] = None
        async for event in self.astream(task, context=context):
            if event.type is EventType.RUN_END and isinstance(event.raw, TeamResult):
                result = event.raw
        if result is None:  # pragma: no cover - defensive
            raise OrchestrationError("Team run produced no result")
        return result

    def call(self, task: str, **kwargs: Any) -> str:
        """Run the team and return only the final text."""
        return self.run(task, **kwargs).output

    def stream(
        self, task: Union[str, Message], *, context: Optional[Mapping[str, Any]] = None
    ) -> Iterator[AgentEvent]:
        """Iterate team events synchronously."""
        return iter_sync(lambda: self.astream(task, context=context))

    async def astream(
        self, task: Union[str, Message], *, context: Optional[Mapping[str, Any]] = None
    ) -> AsyncIterator[AgentEvent]:
        """Async generator of team events, including every member's events."""
        task_text = task.text if isinstance(task, Message) else str(task)
        team_result = TeamResult(
            run_id=new_id("team"),
            status=RunStatus.RUNNING,
            agent=self.name,
            mode=self.mode,
            metadata={"task": task_text, "mode": self.mode},
        )
        yield AgentEvent(
            type=EventType.RUN_START,
            agent=self.name,
            run_id=team_result.run_id,
            data={"mode": self.mode, "agents": [agent.name for agent in self._members()]},
            text=task_text,
        )
        try:
            if self.mode == "sequential":
                async for event in self._run_sequential(task, team_result, context):
                    yield event
            elif self.mode == "parallel":
                async for event in self._run_parallel(task, team_result, context):
                    yield event
            else:
                async for event in self._run_orchestrator(task, team_result, context):
                    yield event
        except asyncio.CancelledError:
            team_result.status = RunStatus.CANCELLED
            team_result.error = "Team cancelled"
            team_result.finished_at = utc_now()
            if self.store is not None:
                self.store.save(team_result, event="cancelled")
            raise
        except PyAgentsError as exc:
            team_result.status = RunStatus.FAILED
            team_result.error = str(exc)
            team_result.finished_at = utc_now()
            attach_run_result(exc, team_result)
            if self.store is not None:
                self.store.save(team_result, event="failed")
            yield AgentEvent(
                type=EventType.ERROR,
                agent=self.name,
                run_id=team_result.run_id,
                data={"error": str(exc)},
            )
            raise

        team_result.status = RunStatus.COMPLETED
        team_result.finished_at = utc_now()
        if self.store is not None:
            self.store.save(team_result, event="finished")
        yield AgentEvent(
            type=EventType.RUN_END,
            agent=self.name,
            run_id=team_result.run_id,
            text=team_result.output,
            data={"status": str(team_result.status), "mode": self.mode},
            raw=team_result,
        )

    # ------------------------------------------------------------------ #
    # Strategies
    # ------------------------------------------------------------------ #
    def _members(self) -> List[Agent]:
        if self.mode == "orchestrator":
            return [self.orchestrator, *getattr(self, "workers", [])]
        return list(self.agents) + ([self.aggregator] if self.aggregator else [])

    async def _run_sequential(
        self,
        task: Union[str, Message],
        team_result: TeamResult,
        context: Optional[Mapping[str, Any]],
    ) -> AsyncIterator[AgentEvent]:
        """Run agents in order, threading each answer into the next prompt."""
        base_task = task.text if isinstance(task, Message) else str(task)
        prompt = base_task
        previous: Optional[RunResult] = None
        for turn in range(self.turns):
            for agent in self.agents:
                if previous is not None:
                    team_result.handoffs.append(
                        Handoff(
                            from_agent=previous.agent,
                            to_agent=agent.name,
                            task=prompt[:200],
                            reason=f"sequential step {turn + 1}",
                        )
                    )
                async for event in self._run_member(agent, prompt, context):
                    if event.type is EventType.AGENT_END and event.raw is not None:
                        previous = event.raw
                    yield event
                assert previous is not None
                team_result.agent_results.append(previous)
                team_result.usage += previous.usage
                team_result.messages.extend(previous.messages)
                team_result.steps.extend(_renumber(previous.steps, len(team_result.steps)))
                if self.pass_context:
                    prompt = (
                        f"{base_task}\n\n"
                        f"Work already done by {previous.agent}:\n{previous.output}\n\n"
                        f"Continue from there and produce the best possible result."
                    )
                else:
                    prompt = base_task
        team_result.output = previous.output if previous else ""
        team_result.metadata["pipeline"] = [result.agent for result in team_result.agent_results]

    async def _run_parallel(
        self,
        task: Union[str, Message],
        team_result: TeamResult,
        context: Optional[Mapping[str, Any]],
    ) -> AsyncIterator[AgentEvent]:
        """Run every agent on the same task, then aggregate."""
        base_task = task.text if isinstance(task, Message) else str(task)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def worker(agent: Agent) -> None:
            async with semaphore:
                result: Optional[RunResult] = None
                try:
                    async for event in agent.astream(base_task, context=context):
                        if event.type is EventType.RUN_END:
                            result = event.raw
                        else:
                            await queue.put(("event", event))
                except Exception as exc:
                    await queue.put(("error", (agent.name, exc)))
                await queue.put(("done", (agent, result)))

        tasks = [asyncio.create_task(worker(agent)) for agent in self.agents]
        completed = 0
        for _ in range(len(tasks)):
            while True:
                kind, payload = await queue.get()
                if kind == "event":
                    yield payload
                    continue
                if kind == "error":
                    name, exc = payload
                    team_result.metadata.setdefault("errors", {})[name] = str(exc)
                    break
                _agent, result = payload
                completed += 1
                if result is not None:
                    team_result.agent_results.append(result)
                    team_result.usage += result.usage
                    team_result.messages.extend(result.messages)
                    team_result.steps.extend(_renumber(result.steps, len(team_result.steps)))
                break
        await asyncio.gather(*tasks, return_exceptions=True)

        outputs = team_result.outputs
        if self.aggregator is not None:
            digest = "\n\n".join(f"## {name}\n{text}" for name, text in outputs.items())
            synthesis_prompt = (
                f"{base_task}\n\nSynthesise the following independent contributions into one "
                f"coherent answer. Prefer agreement, flag conflicts, and keep it tight.\n\n{digest}"
            )
            final: Optional[RunResult] = None
            async for event in self._run_member(self.aggregator, synthesis_prompt, context):
                if event.type is EventType.AGENT_END and event.raw is not None:
                    final = event.raw
                yield event
            if final is not None:
                team_result.agent_results.append(final)
                team_result.usage += final.usage
                team_result.messages.extend(final.messages)
                team_result.output = final.output
            else:  # pragma: no cover - defensive
                team_result.output = "\n\n".join(outputs.values())
        else:
            team_result.output = "\n\n".join(
                f"## {name}\n{text}" for name, text in outputs.items() if text
            )
        team_result.metadata["completed_agents"] = completed

    async def _run_orchestrator(
        self,
        task: Union[str, Message],
        team_result: TeamResult,
        context: Optional[Mapping[str, Any]],
    ) -> AsyncIterator[AgentEvent]:
        """Run the manager agent, which delegates through handoff tools."""
        manager = self.orchestrator
        assert manager is not None  # guaranteed by _configure_orchestrator
        final: Optional[RunResult] = None
        async for event in self._run_member(manager, task, context):
            if event.type is EventType.AGENT_END and event.raw is not None:
                final = event.raw
            yield event
        if final is None:  # pragma: no cover - defensive
            raise OrchestrationError("Orchestrator produced no result")
        team_result.agent_results.append(final)
        team_result.usage += final.usage
        team_result.messages.extend(final.messages)
        team_result.steps.extend(_renumber(final.steps, len(team_result.steps)))
        team_result.handoffs.extend(final.handoffs)
        team_result.output = final.output
        for step in final.steps:
            for tool_result in step.tool_results:
                if isinstance(tool_result.output, RunResult) and not any(
                    existing.run_id == tool_result.output.run_id
                    for existing in team_result.agent_results
                ):
                    team_result.agent_results.append(tool_result.output)

    async def _run_member(
        self,
        agent: Agent,
        task: Union[str, Message],
        context: Optional[Mapping[str, Any]],
    ) -> AsyncIterator[AgentEvent]:
        """Run one member, bracketing its events with AGENT_START/AGENT_END."""
        task_text = task.text if isinstance(task, Message) else str(task)
        if self.verbose:  # pragma: no cover - developer convenience
            print(f"[{self.name}] → {agent.name}: {task_text[:80]}")
        yield AgentEvent(
            type=EventType.AGENT_START,
            agent=agent.name,
            data={"task": task_text},
        )
        started = time.perf_counter()
        result: Optional[RunResult] = None
        async for event in agent.astream(task, context=context, memory=self.memory):
            if event.type is EventType.RUN_END:
                result = event.raw
                continue
            yield event
        yield AgentEvent(
            type=EventType.AGENT_END,
            agent=agent.name,
            text=result.output if result else "",
            data={
                "status": str(result.status) if result else "failed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            raw=result,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the team configuration."""
        return {
            "name": self.name,
            "mode": self.mode,
            "agents": [agent.name for agent in self._members()],
            "turns": self.turns,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Team(name={self.name!r}, mode={self.mode!r}, agents={[a.name for a in self._members()]})"


def _renumber(steps: Sequence[Any], offset: int) -> List[Any]:
    """Copy steps from another run, re-indexing them into the merged list."""
    from .types import Step

    copied: List[Any] = []
    for index, step in enumerate(steps):
        clone = Step.from_dict(step.to_dict())
        clone.index = offset + index + 1
        copied.append(clone)
    return copied


def sequential(*agents: Agent, **kwargs: Any) -> Team:
    """Build a sequential team (pipeline)."""
    return Team(list(agents), mode="sequential", **kwargs)


def parallel(*agents: Agent, **kwargs: Any) -> Team:
    """Build a parallel team."""
    return Team(list(agents), mode="parallel", **kwargs)
