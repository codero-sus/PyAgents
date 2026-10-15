"""The :class:`Agent` - a single reasoning loop with tools and memory.

An agent owns four things:

* **instructions** - the system prompt
* **a provider** - which model to talk to (see :mod:`pyagents.llm`)
* **tools** - functions the model may call (see :mod:`pyagents.tools`)
* **memory** - what it remembers between runs (see :mod:`pyagents.memory`)

It runs a ReAct-style loop: ask the model, execute any tool calls it requests,
feed the observations back, and repeat until the model answers without tools or
the step budget runs out. Everything the loop does is observable as a stream of
:class:`~pyagents.types.AgentEvent` objects::

    from pyagents import Agent
    from pyagents.tools import calculator

    agent = Agent("math", instructions="You are precise.", tools=[calculator])
    result = agent.run("What is (2 + 3) * 4?")
    print(result.output, result.status, result.usage.total_tokens)

    for event in agent.stream("Now double it"):
        if event.type == "text":
            print(event.text, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from ._sync import iter_sync, run_sync
from .errors import (
    AgentError,
    ConfigurationError,
    MaxStepsExceeded,
    PyAgentsError,
    StructuredOutputError,
    ToolNotFoundError,
    attach_run_result,
)
from .llm.base import BaseProvider
from .memory import Memory, VectorMemory
from .parsers import Parser, parser_for
from .stores import RunStore
from .tools.base import Tool, Toolkit, run_tool_call
from .types import (
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
    new_id,
    utc_now,
)
from .utils import maybe_await, slugify

__all__ = ["HOOK_NAMES", "Agent", "HookFn"]

#: Names of the supported lifecycle hooks.
HOOK_NAMES = ("before_run", "after_run", "before_model", "after_model", "before_tool", "after_tool")

HookFn = Callable[..., Any]

DEFAULT_INSTRUCTIONS = (
    "You are a capable assistant. Think step by step, use the available tools "
    "when they help, and answer the user's request directly and completely."
)


class Agent:
    """A tool-using LLM agent.

    Args:
        name: Identifier used in events, logs and handoff tools.
        instructions: System prompt. May contain ``{placeholders}`` filled from
            the ``context`` mapping passed to :meth:`run`.
        description: One-line summary; used when the agent is exposed as a tool
            or member of a team.
        provider: A provider instance, a registered provider name
            (``"openai"``, ``"anthropic"``, ``"ollama"``) or ``None`` to pick one
            from the environment.
        model: Model identifier; ignored when ``provider`` is an instance that
            already carries a model.
        tools: Callables, :class:`~pyagents.tools.Tool` objects or tool specs.
        memory: Conversation memory shared across runs.
        output_model: Dataclass or pydantic model used to parse the final answer.
        output_parser: Custom :class:`~pyagents.parsers.Parser`; overrides
            ``output_model``.
        max_steps: Maximum model calls per run (each tool round trip is a step).
        handoffs: Mapping of agent name to :class:`Agent` this agent may delegate
            to. Each entry becomes a ``transfer_to_<name>`` tool.
        tool_choice: Provider-specific tool-choice hint.
        parallel_tools: Execute tool calls from one step concurrently.
        max_tool_concurrency: Upper bound on concurrent tool executions.
        store: :class:`~pyagents.stores.RunStore` for checkpointing.
        checkpoint: Persist the run after every step when a store is configured.
        raise_on_max_steps: Raise :class:`~pyagents.errors.MaxStepsExceeded`
            instead of returning a ``max_steps`` result.
        structured_retries: Extra attempts to obtain parseable structured output.
        verbose: Print events to stdout while running (handy in notebooks).
        metadata: Arbitrary metadata attached to every run result.
    """

    def __init__(
        self,
        name: str = "agent",
        *,
        instructions: Optional[str] = None,
        description: str = "",
        provider: Optional[Union[str, BaseProvider]] = None,
        model: Optional[str] = None,
        tools: Optional[Iterable[Any]] = None,
        memory: Optional[Memory] = None,
        output_model: Optional[Type[Any]] = None,
        output_parser: Optional[Parser] = None,
        max_steps: int = 10,
        handoffs: Optional[Mapping[str, Agent]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tools: bool = True,
        max_tool_concurrency: int = 8,
        store: Optional[RunStore] = None,
        checkpoint: bool = True,
        raise_on_max_steps: bool = False,
        structured_retries: int = 1,
        verbose: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        hooks: Optional[Mapping[str, HookFn]] = None,
        stream_model: bool = True,
    ) -> None:
        self.name = name
        self.description = description or f"{name} agent"
        self.instructions = instructions if instructions is not None else DEFAULT_INSTRUCTIONS
        self.model = model
        self._provider_arg = provider
        self._provider: Optional[BaseProvider] = None
        self.memory = memory
        self.output_model = output_model
        self.parser: Optional[Parser] = (
            parser_for(output_parser or output_model) if (output_model or output_parser) else None
        )
        self.max_steps = max(1, int(max_steps))
        self.tool_choice = tool_choice
        self.parallel_tools = parallel_tools
        self.max_tool_concurrency = max(1, int(max_tool_concurrency))
        self.store = store
        self.checkpoint = checkpoint
        self.raise_on_max_steps = raise_on_max_steps
        self.structured_retries = max(0, int(structured_retries))
        self.verbose = verbose
        self.stream_model = stream_model
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.toolkit = Toolkit(tools or [])
        self.handoffs: Dict[str, Agent] = dict(handoffs or {})
        self.hooks: Dict[str, HookFn] = {
            name: fn for name, fn in (hooks or {}).items() if fn is not None
        }
        unknown_hooks = set(self.hooks) - set(HOOK_NAMES)
        if unknown_hooks:
            raise ConfigurationError(
                f"Unknown hook(s): {', '.join(sorted(unknown_hooks))}", known=list(HOOK_NAMES)
            )
        self._validate_handoffs()

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    @property
    def provider(self) -> BaseProvider:
        """The resolved provider (created on first use)."""
        if self._provider is None:
            self._provider = resolve_provider(self._provider_arg, self.model)
        return self._provider

    @provider.setter
    def provider(self, value: Union[str, BaseProvider]) -> None:
        self._provider_arg = value
        self._provider = None

    @property
    def tools(self) -> List[Tool]:
        """Tools available to the model (handoffs excluded)."""
        return list(self.toolkit)

    @property
    def all_tools(self) -> List[Tool]:
        """Tools available to the model, including generated handoff tools."""
        return self.tools + self.handoff_tools()

    def add_tool(self, entry: Any) -> Tool:
        """Register a tool and return it."""
        return self.toolkit.add(entry)

    def add_handoff(self, name: str, agent: Agent) -> None:
        """Allow this agent to delegate to ``agent`` via a generated tool."""
        if not isinstance(agent, Agent):
            raise ConfigurationError("Handoffs must point at Agent instances", target=name)
        self.handoffs[name] = agent

    def tool_specs(self) -> List[Dict[str, Any]]:
        """Canonical specs for every tool the model can call."""
        return [tool_obj.to_spec() for tool_obj in self.all_tools]

    def handoff_tools(self) -> List[Tool]:
        """Build the ``transfer_to_*`` tools for configured handoffs."""
        tools: List[Tool] = []
        for key, target in self.handoffs.items():
            tools.append(self._build_handoff_tool(key, target))
        return tools

    def _handoff_tool_name(self, key: str) -> str:
        """Normalise a handoff key into a tool name."""
        slug = slugify(key)
        return slug if slug.startswith("transfer_to_") else f"transfer_to_{slug}"

    def _build_handoff_tool(self, key: str, target: Agent) -> Tool:
        tool_name = self._handoff_tool_name(key)

        def _handoff(task: str, reason: str = "") -> str:
            suffix = f" (reason: {reason})" if reason else ""
            return f"Handing off to {target.name}: {task}{suffix}"

        _handoff.__name__ = tool_name
        description = (
            f"Delegate work to the '{target.name}' agent. {target.description} "
            "Use this when the task is better handled by that specialist."
        )
        return Tool(
            _handoff,
            name=tool_name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Self-contained instructions for the agent.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this agent is the right choice.",
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            metadata={"handoff": target.name},
        )

    def _validate_handoffs(self) -> None:
        """Guard against handoff cycles that would recurse forever."""
        seen: List[str] = []

        def walk(current: Agent) -> None:
            if current.name in seen:
                raise ConfigurationError("Handoff cycle detected", chain=[*seen, current.name])
            seen.append(current.name)
            for target in current.handoffs.values():
                walk(target)
            seen.pop()

        walk(self)

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #
    def system_message(self, context: Optional[Mapping[str, Any]] = None) -> Message:
        """Render the system prompt, filling ``{placeholders}`` from ``context``."""
        text = self.instructions or DEFAULT_INSTRUCTIONS
        if context:
            with contextlib.suppress(KeyError, IndexError, ValueError):
                # Leave unknown placeholders untouched rather than failing.
                text = text.format(**context)
        if self.parser is not None:
            text = f"{text}\n\n{self.parser.format_instructions()}"
        return Message.system(text, agent=self.name)

    def build_messages(
        self,
        task: Union[str, Message, Sequence[Message], None] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        memory: Optional[Memory] = None,
        system: bool = True,
    ) -> List[Message]:
        """Assemble the message list sent to the model.

        The system prompt is first, followed by recalled memory, followed by
        the new task.
        """
        messages: List[Message] = []
        if system:
            messages.append(self.system_message(context))
        history = list(memory.messages()) if memory is not None else []
        messages.extend(self._sanitize_history(history, system=system))
        messages.extend(self._task_messages(task))
        return messages

    @staticmethod
    def _task_messages(task: Union[str, Message, Sequence[Message], None]) -> List[Message]:
        """Normalise the ``task`` argument into a list of messages."""
        if task is None:
            return []
        if isinstance(task, Message):
            return [task]
        if isinstance(task, str):
            return [Message.user(task)]
        return [Message.coerce(item) for item in task]

    @staticmethod
    def _sanitize_history(history: Sequence[Message], *, system: bool) -> List[Message]:
        """Keep tool calls and their observations together.

        Memory policies can truncate the transcript in the middle of a tool
        round trip, which most providers reject. This drops assistant messages
        whose tool calls have no matching result and results with no request.
        """
        result: List[Message] = []
        pending: Dict[str, int] = {}
        for message in history:
            if system and message.role is Role.SYSTEM:
                continue
            if message.role is Role.ASSISTANT and message.tool_calls:
                pending = {call.id: len(result) for call in message.tool_calls}
                result.append(message)
                continue
            if message.role is Role.TOOL:
                call_id = message.tool_call_id or ""
                if call_id in pending:
                    pending.pop(call_id, None)
                    result.append(message)
                continue
            result.append(message)
        if pending:
            # Drop assistant messages whose tool results never arrived.
            drop_indices = set(pending.values())
            result = [message for index, message in enumerate(result) if index not in drop_indices]
        return result

    # ------------------------------------------------------------------ #
    # Public execution API
    # ------------------------------------------------------------------ #
    def run(
        self,
        task: Union[str, Message, Sequence[Message], None] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """Execute the agent to completion and return the :class:`RunResult`."""
        return run_sync(lambda: self.arun(task, context=context, memory=memory, run_id=run_id))

    async def arun(
        self,
        task: Union[str, Message, Sequence[Message], None] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """Async variant of :meth:`run`."""
        result: Optional[RunResult] = None
        async for event in self.astream(task, context=context, memory=memory, run_id=run_id):
            if event.type is EventType.RUN_END and event.raw is not None:
                result = event.raw
        if result is None:  # pragma: no cover - defensive
            raise AgentError("Agent run produced no result")
        return result

    def call(self, task: str, **kwargs: Any) -> str:
        """Run the agent and return only the final text."""
        return self.run(task, **kwargs).output

    async def acall(self, task: str, **kwargs: Any) -> str:
        """Async variant of :meth:`call`."""
        return (await self.arun(task, **kwargs)).output

    def stream(
        self,
        task: Union[str, Message, Sequence[Message], None] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> Iterator[AgentEvent]:
        """Iterate :class:`AgentEvent` objects as the run progresses (sync)."""
        factory = lambda: self.astream(task, context=context, memory=memory, run_id=run_id)  # noqa: E731
        return iter_sync(factory)

    async def astream(
        self,
        task: Union[str, Message, Sequence[Message], None] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        memory: Optional[Memory] = None,
        run_id: Optional[str] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Async generator of :class:`AgentEvent` objects.

        The final event is always ``RUN_END``; ``event.raw`` holds the
        :class:`~pyagents.types.RunResult` for programmatic use.
        """
        memory = memory if memory is not None else self.memory
        task_text = self._task_text(task)
        if not task_text and not self._has_user_history(memory):
            raise ConfigurationError(
                "Agent.run() needs a task (a string, Message or message sequence)"
            )
        if isinstance(memory, VectorMemory) and task_text:
            memory.set_query(task_text)

        run = RunResult(
            run_id=run_id or new_id("run"),
            status=RunStatus.RUNNING,
            agent=self.name,
            metadata={"task": task_text, **self.metadata},
        )
        messages = self.build_messages(task, context=context, memory=memory)
        run.messages = [message.copy() for message in messages]
        # Remember the new task *after* the transcript was assembled, so it is
        # not replayed twice on the next turn.
        for task_message in self._task_messages(task):
            await self._remember(memory, task_message, run)

        await self._invoke_hook("before_run", messages=messages, run=run)

        if self.verbose:  # pragma: no cover - developer convenience
            print(
                f"[{self.name}] run started ({getattr(self.provider, 'name', '?')}/{getattr(self.provider, 'model', '?')})"
            )

        yield self._event(EventType.RUN_START, run, data={"task": task_text})

        status = RunStatus.RUNNING
        steps_taken = 0
        try:
            while steps_taken < self.max_steps:
                steps_taken += 1
                yield self._event(EventType.STEP_START, run, step=steps_taken)

                response: Optional[LLMResponse] = None
                async for event in self._model_events(
                    messages, run=run, step=steps_taken, stream=self.stream_model
                ):
                    if event.type is EventType.MESSAGE:
                        response = event.raw
                    yield event
                if response is None:  # pragma: no cover - defensive
                    raise AgentError("Provider returned no response")

                assistant = response.message
                run.usage += response.usage
                messages.append(assistant)
                run.messages.append(assistant.copy())
                await self._remember(memory, assistant, run)

                step = Step(
                    index=steps_taken,
                    message=assistant,
                    usage=response.usage,
                )

                if assistant.tool_calls:
                    async for event, tool_result in self._run_tools(
                        assistant.tool_calls, run=run, step=steps_taken
                    ):
                        if tool_result is None:
                            yield event
                            continue
                        step.tool_results.append(tool_result)
                        observation = tool_result.to_message()
                        messages.append(observation)
                        run.messages.append(observation.copy())
                        await self._remember(memory, observation, run)
                        yield event

                step.duration_ms = (utc_now() - step.started_at) * 1000
                run.steps.append(step)
                if self.store is not None and self.checkpoint:
                    self.store.save(run, event=f"step_{steps_taken}")
                    yield self._event(
                        EventType.CHECKPOINT, run, step=steps_taken, data={"event": "step"}
                    )

                if not assistant.tool_calls:
                    status = RunStatus.COMPLETED
                    break
            else:
                status = RunStatus.MAX_STEPS
                run.error = f"Reached the maximum of {self.max_steps} steps without a final answer"

            if status is RunStatus.COMPLETED:
                output, structured = await self._finalize_output(run, messages, memory)
                run.output = output
                if structured is not None:
                    run.structured = structured
                    if isinstance(structured, str):
                        run.output = structured
        except asyncio.CancelledError:
            run.status = status = RunStatus.CANCELLED
            run.error = "Run cancelled"
            run.finished_at = utc_now()
            if self.store is not None:
                self.store.save(run, event="cancelled")
            yield self._event(
                EventType.ERROR, run, data={"error": run.error, "type": "CancelledError"}
            )
            raise
        except PyAgentsError as exc:
            run.status = status = RunStatus.FAILED
            run.error = str(exc)
            run.finished_at = utc_now()
            attach_run_result(exc, run)
            if self.store is not None:
                self.store.save(run, event="failed")
            yield self._event(
                EventType.ERROR, run, data={"error": str(exc), "type": type(exc).__name__}
            )
            raise
        except Exception as exc:
            run.status = status = RunStatus.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            run.finished_at = utc_now()
            attach_run_result(exc, run)
            if self.store is not None:
                self.store.save(run, event="failed")
            yield self._event(
                EventType.ERROR, run, data={"error": run.error, "type": type(exc).__name__}
            )
            raise

        run.status = status
        run.finished_at = utc_now()
        if self.store is not None:
            self.store.save(run, event="finished")
        await self._invoke_hook("after_run", result=run)

        if self.verbose:  # pragma: no cover
            print(f"[{self.name}] run finished: {status} ({run.duration_ms:.0f} ms)")
        if status is RunStatus.MAX_STEPS and self.raise_on_max_steps:
            raise MaxStepsExceeded(
                run.error or "Max steps exceeded", run_id=run.run_id, steps=steps_taken
            )

        yield self._event(
            EventType.RUN_END, run, data={"status": str(status)}, text=run.output, raw=run
        )

    # ------------------------------------------------------------------ #
    # Internals: model calls
    # ------------------------------------------------------------------ #
    async def _model_events(
        self, messages: List[Message], *, run: RunResult, step: int, stream: bool = True
    ) -> AsyncIterator[AgentEvent]:
        """Call the provider and yield TEXT chunks plus a final MESSAGE event."""
        specs = self.tool_specs()
        await self._invoke_hook("before_model", messages=messages, run=run)
        kwargs: Dict[str, Any] = {}
        if specs:
            kwargs["tools"] = specs
            if self.tool_choice is not None:
                kwargs["tool_choice"] = self.tool_choice
        agent_name = self.name
        response: Optional[LLMResponse] = None
        if stream and hasattr(self.provider, "astream"):
            async for chunk in self.provider.astream(messages, **kwargs):
                if chunk.kind == "text" and chunk.text:
                    if self.verbose:  # pragma: no cover
                        print(chunk.text, end="", flush=True)
                    yield AgentEvent(
                        type=EventType.TEXT,
                        agent=agent_name,
                        text=chunk.text,
                        run_id=run.run_id,
                        step=step,
                    )
                elif chunk.kind == "response" and chunk.response is not None:
                    response = chunk.response
        if response is None:
            # Non-streaming providers (and streaming ones that returned nothing).
            response = await self._acomplete(messages, kwargs)

        replacement = await self._invoke_hook("after_model", response=response, run=run)
        response = replacement or response
        yield AgentEvent(
            type=EventType.MESSAGE,
            agent=agent_name,
            message=response.message,
            run_id=run.run_id,
            step=step,
            data={"usage": response.usage.to_dict(), "finish_reason": response.finish_reason or ""},
            raw=response,
        )

    async def _acomplete(self, messages: List[Message], kwargs: Mapping[str, Any]) -> LLMResponse:
        """Await a provider completion, supporting sync-only custom providers."""
        acomplete = getattr(self.provider, "acomplete", None)
        if acomplete is not None:
            return await maybe_await(acomplete(messages, **kwargs))
        complete = getattr(self.provider, "complete", None)
        if complete is None:
            raise ConfigurationError(
                "Provider exposes neither complete() nor acomplete()",
                provider=type(self.provider).__name__,
            )
        return await asyncio.to_thread(complete, messages, **dict(kwargs))

    async def _run_tools(
        self, calls: Sequence[ToolCall], *, run: RunResult, step: int
    ) -> AsyncIterator[Tuple[AgentEvent, Optional[ToolResult]]]:
        """Execute tool calls, yielding events as they happen.

        Yields ``(event, None)`` for progress events and ``(event, result)``
        when a result is ready, so the caller can append observations in order.
        """
        queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(self.max_tool_concurrency if self.parallel_tools else 1)

        async def emit(event: AgentEvent) -> None:
            await queue.put(("event", event))

        async def worker(call: ToolCall) -> None:
            async with semaphore:
                try:
                    result = await self._execute_tool(call, emit=emit, run=run, step=step)
                except Exception as exc:
                    result = ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=f"{type(exc).__name__}: {exc}",
                        is_error=True,
                    )
                await queue.put(("result", result))

        for call in calls:
            yield (
                AgentEvent(
                    type=EventType.TOOL_CALL,
                    agent=self.name,
                    tool_call=call,
                    run_id=run.run_id,
                    step=step,
                ),
                None,
            )

        if not calls:
            return

        tasks = [asyncio.create_task(worker(call)) for call in calls]
        try:
            for _ in range(len(tasks)):
                while True:
                    kind, payload = await queue.get()
                    if kind == "event":
                        yield (payload, None)
                        continue
                    yield (
                        AgentEvent(
                            type=EventType.TOOL_RESULT,
                            agent=self.name,
                            tool_result=payload,
                            run_id=run.run_id,
                            step=step,
                            text=payload.text,
                        ),
                        payload,
                    )
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_tool(
        self,
        call: ToolCall,
        *,
        emit: Callable[[AgentEvent], Awaitable[None]],
        run: RunResult,
        step: int,
    ) -> ToolResult:
        """Run a single tool call, resolving handoffs to nested agent runs."""
        handoff_target = self._handoff_target(call.name)
        if handoff_target is not None:
            return await self._run_handoff(call, handoff_target, emit=emit, run=run, step=step)

        try:
            tool_obj = self.toolkit.get(call.name)
        except ToolNotFoundError as exc:
            available = ", ".join(self.toolkit.names()) or "none"
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"{exc}. Available tools: {available}",
                is_error=True,
            )
        await self._invoke_hook("before_tool", tool=tool_obj, call=call, run=run)
        result = await run_tool_call(tool_obj, call)
        await self._invoke_hook("after_tool", tool=tool_obj, call=call, result=result, run=run)
        return result

    def _handoff_target(self, tool_name: str) -> Optional[Agent]:
        """Return the agent a tool name refers to, if it is a handoff."""
        for key, target in self.handoffs.items():
            if self._handoff_tool_name(key) == tool_name:
                return target
        return None

    async def _run_handoff(
        self,
        call: ToolCall,
        target: Agent,
        *,
        emit: Callable[[AgentEvent], Awaitable[None]],
        run: RunResult,
        step: int,
    ) -> ToolResult:
        """Delegate to another agent and feed its answer back as the observation."""
        arguments = call.arguments or {}
        task = str(arguments.get("task") or run.metadata.get("task") or "")
        reason = str(arguments.get("reason") or "")
        handoff = Handoff(from_agent=self.name, to_agent=target.name, task=task, reason=reason)
        run.handoffs.append(handoff)
        await emit(
            AgentEvent(
                type=EventType.HANDOFF,
                agent=self.name,
                run_id=run.run_id,
                step=step,
                text=f"{self.name} → {target.name}",
                data=handoff.to_dict(),
            )
        )
        await emit(
            AgentEvent(
                type=EventType.AGENT_START,
                agent=target.name,
                run_id=run.run_id,
                step=step,
                data={"task": task},
            )
        )
        started = time.perf_counter()
        sub_result: Optional[RunResult] = None
        async for event in target.astream(task):
            if event.type is EventType.RUN_END:
                sub_result = event.raw
                continue
            # Re-emit the sub-agent's activity so observers see the whole tree.
            await emit(event)
        duration_ms = (time.perf_counter() - started) * 1000
        output = sub_result.output if sub_result else ""
        await emit(
            AgentEvent(
                type=EventType.AGENT_END,
                agent=target.name,
                run_id=run.run_id,
                step=step,
                text=output,
                data={
                    "status": str(sub_result.status) if sub_result else "failed",
                    "duration_ms": round(duration_ms, 2),
                    "run_id": sub_result.run_id if sub_result else "",
                },
            )
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=output or f"{target.name} produced no output",
            output=sub_result,
            is_error=bool(sub_result is None or sub_result.status is not RunStatus.COMPLETED),
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------ #
    # Internals: output handling
    # ------------------------------------------------------------------ #
    async def _finalize_output(
        self, run: RunResult, messages: List[Message], memory: Optional[Memory]
    ) -> Tuple[str, Any]:
        """Extract the final answer, parsing structured output when configured."""
        output = run.messages[-1].text if run.messages else ""
        if self.parser is None:
            return output, None

        attempts = self.structured_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                parsed = self.parser.parse(output)
            except StructuredOutputError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                repair = Message.user(
                    f"That response could not be parsed ({exc}). "
                    f"Reply with valid JSON only.\n\n{self.parser.format_instructions()}"
                )
                messages.append(repair)
                run.messages.append(repair.copy())
                response = await self.provider.acomplete(messages, tools=None)
                run.usage += response.usage
                messages.append(response.message)
                run.messages.append(response.message.copy())
                run.steps.append(
                    Step(index=len(run.steps) + 1, message=response.message, usage=response.usage)
                )
                await self._remember(memory, response.message, run)
                output = response.message.text
                continue
            run.metadata["structured"] = True
            return output, parsed

        run.metadata["structured_error"] = str(last_error)
        run.metadata["structured"] = False
        return output, None

    # ------------------------------------------------------------------ #
    # Internals: plumbing
    # ------------------------------------------------------------------ #
    def _event(
        self,
        event_type: EventType,
        run: RunResult,
        *,
        step: int = 0,
        text: str = "",
        data: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentEvent:
        """Build an :class:`AgentEvent` bound to ``run``."""
        return AgentEvent(
            type=event_type,
            agent=self.name,
            run_id=run.run_id,
            step=step,
            text=text,
            data=dict(data or {}),
            **kwargs,
        )

    async def _invoke_hook(self, hook: str, **payload: Any) -> Any:
        """Call a lifecycle hook if configured, awaiting it when necessary."""
        function = self.hooks.get(hook)
        if function is None:
            return None
        try:
            return await maybe_await(function(self, **payload))
        except TypeError:
            # Allow hooks that take only the payload they care about.
            signature = inspect.signature(function)
            accepted = {key: value for key, value in payload.items() if key in signature.parameters}
            return await maybe_await(function(self, **accepted))

    async def _remember(
        self, memory: Optional[Memory], message: Message, run: Optional[RunResult] = None
    ) -> None:
        """Write a message to memory using the async path when available.

        A failing memory backend must not destroy an otherwise good run, so
        errors are recorded on ``run.metadata["memory_errors"]`` instead of
        being raised.
        """
        if memory is None:
            return
        try:
            adder = getattr(memory, "aadd", None)
            if adder is not None:
                await maybe_await(adder(message))
            else:
                memory.add(message)
        except Exception as exc:
            if run is not None:
                run.metadata.setdefault("memory_errors", []).append(f"{type(exc).__name__}: {exc}")

    def _task_text(self, task: Union[str, Message, Sequence[Message], None]) -> str:
        """Render the task as plain text for metadata and handoffs."""
        if task is None:
            return ""
        if isinstance(task, str):
            return task
        if isinstance(task, Message):
            return task.text
        return "\n".join(Message.coerce(item).text for item in task)

    def _has_user_history(self, memory: Optional[Memory]) -> bool:
        """Return ``True`` when memory already holds a user turn."""
        if memory is None:
            return False
        return any(message.role is Role.USER for message in memory.messages())

    # ------------------------------------------------------------------ #
    # Composition helpers
    # ------------------------------------------------------------------ #
    def as_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
        """Expose this agent as a tool other agents can call.

        The tool takes a single ``task`` argument and returns the agent's answer,
        which makes "agent as a tool" composition trivial::

            researcher = Agent("researcher", instructions="Find facts.")
            writer = Agent("writer", tools=[researcher.as_tool()])
        """
        tool_name = name or slugify(self.name)
        tool_description = (
            description or f"{self.description} Provide a self-contained task description."
        )

        def _run_agent(task: str) -> str:
            return self.run(task).output

        _run_agent.__name__ = tool_name
        return Tool(
            _run_agent,
            name=tool_name,
            description=tool_description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What this agent should do."}
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            metadata={"agent": self.name},
        )

    def clone(self, name: Optional[str] = None, **overrides: Any) -> Agent:
        """Return a copy of this agent, optionally renaming it.

        Provider, tools, memory and handoffs are shared with the original;
        prompts, limits and metadata are copied.
        """
        options: Dict[str, Any] = {
            "instructions": self.instructions,
            "description": self.description,
            "provider": self._provider_arg,
            "model": self.model,
            "tools": self.tools,
            "memory": self.memory,
            "output_model": self.output_model,
            "output_parser": self.parser,
            "max_steps": self.max_steps,
            "handoffs": self.handoffs,
            "tool_choice": self.tool_choice,
            "parallel_tools": self.parallel_tools,
            "max_tool_concurrency": self.max_tool_concurrency,
            "store": self.store,
            "checkpoint": self.checkpoint,
            "raise_on_max_steps": self.raise_on_max_steps,
            "structured_retries": self.structured_retries,
            "verbose": self.verbose,
            "metadata": self.metadata,
            "hooks": self.hooks,
            "stream_model": self.stream_model,
        }
        options.update(overrides)
        return Agent(name or self.name, **options)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the agent's configuration (tools are reported by name)."""
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "model": self.model,
            "provider": self.provider.name,
            "tools": [tool_obj.to_spec() for tool_obj in self.all_tools],
            "handoffs": sorted(self.handoffs),
            "max_steps": self.max_steps,
            "output_model": getattr(self.output_model, "__name__", None),
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Agent(name={self.name!r}, tools={self.toolkit.names()}, model={self.model or 'default'})"


def resolve_provider(
    provider: Optional[Union[str, BaseProvider]], model: Optional[str] = None
) -> BaseProvider:
    """Turn the ``provider`` argument into a usable provider instance.

    Accepts a provider instance, a registered provider name, or ``None`` (in
    which case the environment decides - see
    :func:`pyagents.llm.default_provider`).
    """
    from .llm import default_provider, get_provider

    if provider is None:
        return default_provider(model=model)
    if isinstance(provider, BaseProvider):
        return provider
    if isinstance(provider, str):
        return get_provider(provider, model)
    # Duck-typed custom providers are accepted as long as they can answer.
    if hasattr(provider, "complete") or hasattr(provider, "acomplete"):
        return provider
    raise ConfigurationError(
        "provider must be None, a provider name, or an object with complete()/acomplete()",
        provider=type(provider).__name__,
    )
