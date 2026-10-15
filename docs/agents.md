# Agents

An `Agent` is a name, instructions, a provider, and optionally tools, memory and
a run store. Everything else - the reasoning loop, retries, parsing - is
handled for you.

```python
from pyagents import Agent

agent = Agent(
    "analyst",                                    # name (also used in events)
    instructions="You are precise and cite numbers.",
    provider="openai",                            # provider name or instance
    model="gpt-4o-mini",
    tools=[],                                     # Tool objects or plain callables
    memory=None,                                  # see Memory & state
    store=None,                                   # see Memory & state
    max_steps=10,                                 # model calls per run
    temperature=0.2,                              # forwarded to the provider
    output_model=None,                            # dataclass / pydantic model
    stream_model=True,                            # stream token deltas
)
```

## The loop

Each run repeats the same cycle:

1. build the message list (system prompt + memory + task + tool observations)
2. call the provider (streaming or not)
3. if the answer contains tool calls: validate arguments, run the tools -
   concurrently when there is more than one - and append observations
4. otherwise: finish, parse structured output if requested, save the run

`max_steps` caps step 3; when it is hit the run ends with
`RunStatus.MAX_STEPS` and the last assistant text is still returned.
Set `raise_on_max_steps=True` to get a `MaxStepsExceeded` instead.

## Running

```python
result = agent.run("task")              # blocking, returns RunResult
async_result = await agent.arun("task") # native coroutine

for event in agent.stream("task"):      # blocking event iterator
    ...

async for event in agent.astream("task"):
    ...
```

`run()` and `stream()` work even inside a running event loop: PyAgents moves
the coroutine to a worker thread instead of raising "this event loop is already
running".

## Results

A `RunResult` is a complete record of one run:

| attribute | meaning |
|---|---|
| `output` | final assistant text |
| `structured` | parsed `output_model` instance, or `None` |
| `status` | `COMPLETED`, `FAILED`, `MAX_STEPS`, `CANCELLED` |
| `error` | failure message when something went wrong |
| `steps` | list of `Step` (message, tool calls, tool results, timings) |
| `usage` | aggregated token counts |
| `messages` | the full `Message` history |
| `tool_calls` / `tool_results` | flattened views of the steps |
| `handoffs` | agent-to-agent transfers, if any |
| `metadata` | provider, model, duration, structured-output diagnostics |
| `run_id` | unique id; also used as the key in a run store |

Helpers: `result.ok`, `result.duration_ms`, `result.transcript()`,
`result.tool_outputs()`, `result.to_dict()` / `RunResult.from_dict()`.

## Events

`stream()` yields `AgentEvent` objects:

| type | when | payload |
|---|---|---|
| `RUN_START` | run begins | `run` id |
| `STEP_START` | each model call | step index |
| `TEXT` | token delta (streaming) | `event.text` |
| `MESSAGE` | assistant message complete | `event.raw` (LLMResponse) |
| `TOOL_CALL` | model asked for a tool | `event.tool_call` |
| `TOOL_RESULT` | tool finished | `event.tool_result` |
| `CHECKPOINT` | run persisted | run id |
| `HANDOFF` | control moved between agents | `event.text` |
| `AGENT_START` / `AGENT_END` | inside teams/workflows | `event.agent` |
| `ERROR` | run failed | `event.data["error"]` |
| `RUN_END` | run finished | `event.raw` (`RunResult`) |

```python
from pyagents.types import EventType

for event in agent.stream("..."):
    if event.type is EventType.TOOL_RESULT and event.tool_result.is_error:
        print("tool failed:", event.tool_result.content)
```

## Hooks

Hooks let you observe (or veto) the loop without subclassing. A hook receives
the agent as its first argument plus keyword values, and may be a plain
function or a coroutine function:

```python
from pyagents import Agent

def log_call(agent, *, tool=None, call=None, run=None):
    print(f"[{agent.name}] calling {call.name}")

def redact(agent, *, messages=None, run=None):
    return [m for m in messages if "secret" not in m.text]

def must_be_confirmed(agent, *, tool=None, call=None, run=None):
    if tool.name == "delete_file":
        raise PermissionError("delete_file is disabled")
    return call          # return a (possibly modified) ToolCall to continue

agent = Agent("guarded", provider="openai", tools=[...], hooks={
    "before_model": redact,             # messages=, run=      -> list[Message] | None
    "before_tool": must_be_confirmed,   # tool=, call=, run=   -> ToolCall | None
    "after_tool": log_call,
})
```

| hook | keyword arguments | returning a value |
|---|---|---|
| `before_run` | `messages`, `run` | replaces the message list |
| `after_run` | `result` | ignored |
| `before_model` | `messages`, `run` | replaces the message list |
| `after_model` | `response`, `run` | replaces the `LLMResponse` |
| `before_tool` | `tool`, `call`, `run` | replaces the `ToolCall` (raising vetoes it) |
| `after_tool` | `tool`, `call`, `result`, `run` | replaces the `ToolResult` |

Unknown hook names raise `ConfigurationError` at construction time. Returning
`None` always means "keep going unchanged".

## Structured output

Set `output_model` to a dataclass or pydantic model:

```python
import dataclasses

@dataclasses.dataclass
class Answer:
    summary: str
    confidence: float

agent = Agent("extract", provider="openai", output_model=Answer)
result = agent.run("Summarise the article and rate your confidence.")
result.structured          # Answer(summary='...', confidence=0.8)
```

How it works:

* schema-derived instructions are appended to the system prompt
* the final text is parsed leniently: raw JSON, fenced JSON, or the first
  JSON object/array inside prose
* on failure the model gets one repair attempt with the parse error
* if it still fails, the run finishes normally: `output` keeps the raw text,
  `structured` is `None`, and `metadata["structured_error"]` explains why
* control is available through hooks (`parse`) and
  `Agent(output_parser=...)` for fully custom formats

## Error handling

```python
from pyagents.errors import PyAgentsError, ProviderError

try:
    agent.run("...")
except ProviderError as exc:
    partial = getattr(exc, "run_result", None)
    if partial is not None:
        print(partial.steps)      # what happened before the failure
```

Any exception raised during a run is attached to
`exc.run_result` (when the object supports attributes) and the partially
completed run is saved to the store with `RunStatus.FAILED` before the
exception propagates.

## Cancellation

Runs are regular coroutines, so cancellation is ordinary asyncio:

```python
import asyncio

task = asyncio.create_task(agent.arun("long task"))
await asyncio.sleep(2)
task.cancel()

try:
    await task
except asyncio.CancelledError:
    ...
```

A cancelled run is marked `RunStatus.CANCELLED`, emitted as a final `ERROR`
event, and persisted to the store before the `CancelledError` propagates.
Teams and workflows behave the same way.
