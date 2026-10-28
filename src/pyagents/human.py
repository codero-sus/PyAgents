"""Human-in-the-loop support for multi-agent workflows.

:class:`HumanAgent` lets a person be part of an agent team instead of (or in
addition to) an LLM-backed agent. It exposes the same interface as
:class:`~pyagents.agent.Agent` (``run``, ``arun``, ``stream``, ``astream``,
``call``, ``acall``) but instead of calling a model it prompts the human for
input::

    from pyagents import Agent, Team, sequential
    from pyagents.human import HumanAgent

    researcher = Agent("researcher", instructions="Find the facts.")
    reviewer = HumanAgent("editor", instructions="Approve or rewrite the draft.")

    team = sequential(researcher, reviewer)
    result = team.run("Write a brief on HTTPS.")

The human's response becomes the agent's output and flows into the next team
member exactly like any other agent result.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from .errors import ConfigurationError
from .memory import Memory
from .stores import RunStore
from .types import (
    AgentEvent,
    EventType,
    Message,
    Role,
    RunResult,
    RunStatus,
    new_id,
    utc_now,
)

__all__ = ["HumanAgent"]


class HumanAgent:
    """An agent whose "model" is a person.

    When run, the agent presents the task (and any preceding context) to the
    human and waits for a response. That response becomes the agent's output,
    so it slots into teams and workflows exactly like an LLM-backed agent.

    Args:
        name: Identifier used in events and results.
        instructions: Prompt shown to the human before they respond.
        input: A callable that returns the human's response as a string.
            When ``None`` (the default) input is read from ``sys.stdin``.
        interactive: When ``True`` (the default for stdin mode) a ``you>``
            prompt is printed before reading.
        store: Optional run store for checkpointing.
        metadata: Arbitrary metadata attached to every run result.
    """

    def __init__(
        self,
        name: str = "human",
        *,
        instructions: str = "Please review the work above and respond.",
        input: Optional[Callable[[str], str]] = None,
        interactive: bool = True,
        memory: Optional[Memory] = None,
        store: Optional[RunStore] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = f"{name} (human-in-the-loop)"
        self.instructions = instructions
        self._input = input
        self.interactive = interactive
        self.memory = memory
        self.store = store
        self.metadata: Dict[str, Any] = dict(metadata or {})

    # -- public API (mirrors Agent) --------------------------------------- #

    def run(
        self,
        task: Union[str, Message, None] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """Execute synchronously and return the :class:`~pyagents.types.RunResult`."""
        return asyncio.run(self.arun(task, context=context, memory=memory, run_id=run_id))

    async def arun(
        self,
        task: Union[str, Message, None] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """Async variant of :meth:`run`."""
        result: Optional[RunResult] = None
        async for event in self.astream(task, context=context, memory=memory, run_id=run_id):
            if event.type is EventType.RUN_END and event.raw is not None:
                result = event.raw
        if result is None:  # pragma: no cover - defensive
            raise ConfigurationError("HumanAgent produced no result")
        return result

    def call(self, task: str, **kwargs: Any) -> str:
        """Run and return only the human's response text."""
        return self.run(task, **kwargs).output

    async def acall(self, task: str, **kwargs: Any) -> str:
        """Async variant of :meth:`call`."""
        return (await self.arun(task, **kwargs)).output

    def stream(
        self,
        task: Union[str, Message, None] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> Iterator[AgentEvent]:
        """Iterate events synchronously."""
        return asyncio_run_generator(
            self.astream(task, context=context, memory=memory, run_id=run_id)
        )

    async def astream(
        self,
        task: Union[str, Message, None] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> AsyncIterator[AgentEvent]:
        memory = memory if memory is not None else self.memory
        """Async generator of events: prompt, then collect the human's answer."""
        task_text = _task_text(task)
        if not task_text and not _has_user_history(memory):
            raise ConfigurationError("HumanAgent.run() needs a task (a string or Message)")

        run = RunResult(
            run_id=run_id or new_id("run"),
            status=RunStatus.RUNNING,
            agent=self.name,
            metadata={"task": task_text, **self.metadata},
        )
        messages = _build_messages(task, context, memory)
        run.messages = [message.copy() for message in messages]
        for task_message in _task_messages(task):
            await self._remember(memory, task_message, run)

        yield self._event(EventType.RUN_START, run, data={"task": task_text})

        # Present the task + instructions to the human and collect a response.
        prompt = _render_prompt(self.instructions, task_text, messages)
        response_text = await self._ask(prompt)

        assistant = Message.assistant(response_text, agent=self.name)
        messages.append(assistant)
        run.messages.append(assistant.copy())
        await self._remember(memory, assistant, run)

        step_output = f"{self.name}: {response_text}"
        step_index = 1
        yield self._event(EventType.STEP_START, run, step=step_index, text=step_output)
        yield self._event(
            EventType.TEXT,
            run,
            step=step_index,
            text=response_text,
        )
        yield self._event(
            EventType.MESSAGE,
            run,
            step=step_index,
            message=assistant,
            raw=assistant,
        )

        run.output = response_text
        run.status = RunStatus.COMPLETED
        run.finished_at = utc_now()
        if self.store is not None:
            self.store.save(run, event="finished")

        yield self._event(
            EventType.RUN_END,
            run,
            data={"status": str(run.status)},
            text=run.output,
            raw=run,
        )

    # -- composition helpers --------------------------------------------- #

    def clone(self, name: Optional[str] = None, **overrides: Any) -> HumanAgent:
        """Return a copy, optionally renaming it and overriding settings."""
        options: Dict[str, Any] = {
            "instructions": self.instructions,
            "input": self._input,
            "interactive": self.interactive,
            "memory": self.memory,
            "store": self.store,
            "metadata": self.metadata,
        }
        options.update(overrides)
        return HumanAgent(name or self.name, **options)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the agent's configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "interactive": self.interactive,
            "has_memory": self.memory is not None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"HumanAgent(name={self.name!r})"

    # -- internals -------------------------------------------------------- #

    def _event(
        self,
        event_type: EventType,
        run: RunResult,
        *,
        step: int = 0,
        text: str = "",
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentEvent:
        return AgentEvent(
            type=event_type,
            agent=self.name,
            run_id=run.run_id,
            step=step,
            text=text,
            data=dict(data or {}),
            **kwargs,
        )

    async def _ask(self, prompt: str) -> str:
        """Collect the human's response, via the callable or stdin."""
        if self._input is not None:
            return str(self._input(prompt))
        if self.interactive and sys.stdin.isatty():
            print(prompt, end="")
            if not prompt.endswith("\n"):
                print()
            return input("you> ")
        # Non-interactive: read everything remaining on stdin.
        return sys.stdin.read().strip()

    async def _remember(
        self,
        memory: Optional[Memory],
        message: Message,
        run: Optional[RunResult] = None,
    ) -> None:
        """Write a message to memory; failures are recorded, not raised."""
        if memory is None:
            return
        try:
            adder = getattr(memory, "aadd", None)
            if adder is not None:
                await adder(message)
            else:
                memory.add(message)
        except Exception as exc:
            if run is not None:
                run.metadata.setdefault("memory_errors", []).append(f"{type(exc).__name__}: {exc}")


def asyncio_run_generator(
    agen: AsyncIterator[AgentEvent],
) -> Iterator[AgentEvent]:
    """Drive an async generator from sync code (mirrors ``Agent.stream``)."""

    async def drain() -> List[AgentEvent]:
        return [event async for event in agen]

    return iter(asyncio.run(drain()))


def _task_text(task: Union[str, Message, None]) -> str:
    if task is None:
        return ""
    if isinstance(task, str):
        return task
    return task.text


def _task_messages(task: Union[str, Message, None]) -> List[Message]:
    if task is None:
        return []
    if isinstance(task, Message):
        return [task]
    return [Message.user(task)]


def _has_user_history(memory: Optional[Memory]) -> bool:
    if memory is None:
        return False
    return any(message.role is Role.USER for message in memory.messages())


def _build_messages(
    task: Union[str, Message, None],
    _context: Optional[Dict[str, Any]],
    memory: Optional[Memory],
) -> List[Message]:
    messages: List[Message] = []
    history = list(memory.messages()) if memory is not None else []
    messages.extend(history)
    messages.extend(_task_messages(task))
    return messages


def _render_prompt(instructions: str, task_text: str, messages: List[Message]) -> str:
    """Render the text shown to the human before they respond."""
    lines: List[str] = []
    if instructions:
        lines.append(instructions)
    if task_text:
        lines.append("")
        lines.append("Task:")
        lines.append(task_text)
    if messages:
        lines.append("")
        lines.append("Preceding transcript:")
        for message in messages:
            role = str(message.role).upper()
            lines.append(f"  {role}: {message.text}")
    lines.append("")
    lines.append("Your response:")
    return "\n".join(lines)
