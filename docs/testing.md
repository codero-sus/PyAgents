# Testing agents

Agent code is mostly glue: prompts in, tool calls out. That makes it very
testable - as long as you never touch a network. PyAgents ships three fakes so
your test suite can be fast, deterministic and offline.

| tool | replaces | use it when |
|---|---|---|
| `EchoProvider` | a whole provider | you only need *something* that answers |
| `ScriptedProvider` | a provider's reasoning | you want to script a multi-turn tool loop |
| `FakeTransport` | HTTP | you are testing a provider adapter or payload shape |

## EchoProvider

```python
from pyagents import Agent

agent = Agent("demo", provider="echo")
assert "hello" in agent.run("hello").output
```

Deterministic, instant, and registered under the name `echo`, so it also works
from the CLI (`pyagents run "hi" --provider echo`).

## ScriptedProvider

Replays a queue of answers. Every element can be an `LLMResponse`, a plain
string, a callable `(messages) -> LLMResponse | str`, or an exception to raise.

```python
from pyagents import Agent
from pyagents.testing import ScriptedProvider, text_response, tool_response
from pyagents.tools import calculator

provider = ScriptedProvider([
    tool_response("calculator", {"expression": "2 + 2"}),   # turn 1: use a tool
    text_response("2 + 2 is 4."),                           # turn 2: answer
])
agent = Agent("math", provider=provider, tools=[calculator], max_steps=3)

result = agent.run("What is 2 + 2?")

assert result.output == "2 + 2 is 4."
assert result.tool_results[0].content == "2 + 2 = 4"
```

What the provider remembers for you:

| attribute | what it holds |
|---|---|
| `provider.calls` | every `List[Message]` the agent sent, in order |
| `provider.tools_seen` | the tool specs offered on each call |
| `provider.last_messages()` | the most recent message list |
| `provider.last_user_message()` | the most recent user message text |
| `provider.pop(messages)` | consume one script entry manually |

Helper constructors:

```python
from pyagents.testing import response, text_response, tool_response, cycle

response("plain text")                             # -> LLMResponse
text_response("hello", usage=Usage(1, 2, 3))       # -> assistant message
tool_response("calculator", {"expression": "1+1"}) # -> assistant message with a tool call
cycle(lambda i: f"answer {i}")                     # varies by call index
```

Injecting failures is just as easy - queue an exception, or raise it from a
callable:

```python
from pyagents.errors import ProviderError

provider = ScriptedProvider([
    ProviderError("upstream is down", provider="scripted"),
    text_response("recovered"),
])
```

When the script runs out, `ScriptedProvider` keeps answering with
`default_text` (`"Done."`), which keeps tests that assert on step counts from
exploding.

`RecordingProvider` is an alias of `ScriptedProvider` for when you want the
name to say why you are using it.

## FakeTransport

`FakeTransport` implements the same interface as the real HTTP transports
(`post_json`, `post_sse`) and records everything:

```python
from pyagents import OpenAIProvider
from pyagents.testing import FakeTransport
from pyagents.types import Message

transport = FakeTransport([
    {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
])
provider = OpenAIProvider("gpt-4o-mini", api_key="sk-fake", transport=transport)

assert provider.complete([Message.user("hello")]).text == "hi"
assert transport.call_count == 1
assert transport.last_payload["model"] == "gpt-4o-mini"
assert transport.last_request["headers"]["Authorization"].startswith("Bearer ")
```

Queued items can be:

* a `dict` - returned as the parsed JSON body
* a callable `(payload) -> dict` - computed from the request
* an `Exception` instance - raised instead of returning
* a `BaseException` element inside an `sse` bundle - raised mid-stream

Streaming is queued separately and follows the SSE wire format, where each
`data:` line is followed by a blank line:

```python
transport = FakeTransport(
    sse=[[
        '{"choices": [{"delta": {"content": "Hel"}}]}', "",
        '{"choices": [{"delta": {"content": "lo"}}]}', "",
        '{"choices": [{"delta": {}, "finish_reason": "stop"}]}', "",
    ]]
)
chunks = [c for c in OpenAIProvider("m", api_key="k", transport=transport).stream([Message.user("x")])]
assert "".join(c.text for c in chunks if c.kind == "text") == "Hello"
```

Mid-stream failures (`ProviderError` as a *list element* inside an `sse`
bundle) are supported too - that is how the retry semantics ("never retry a
stream that already produced tokens") are tested.

Note that `FakeTransport` receives the raw payload dict, so it is the right
place to assert on authentication headers, control options and message
conversions.

## Recipes

### Assert the agent used a tool

```python
assert [call.name for call in result.tool_calls] == ["calculator"]
```

### Assert the prompt looked right

```python
provider = ScriptedProvider([text_response("ok")])
Agent("t", provider=provider, instructions="Be terse.").run("hello")

system, user = provider.last_messages()[:2]
assert system.role.value == "system" and "terse" in system.text
assert user.text == "hello"
```

### Assert failures stay harmless

```python
provider = ScriptedProvider([
    tool_response("lookup_order", {"order_id": "NOPE"}),
    text_response("I could not find that order."),
])
result = Agent("support", provider=provider, tools=[lookup_order], max_steps=3).run("...")

assert result.status.value == "completed"
assert result.tool_results[0].is_error
```

### Assert partial results survive an exception

```python
from pyagents.errors import ProviderError

agent = Agent("t", provider=ScriptedProvider([ProviderError("boom", provider="x")]))
try:
    agent.run("hi")
except ProviderError as exc:
    assert exc.run_result.status.value == "failed"
```

### Assert memory behaviour

```python
from pyagents import Agent, BufferMemory

memory = BufferMemory(max_messages=4)
provider = ScriptedProvider([text_response("one"), text_response("two")])
agent = Agent("t", provider=provider, memory=memory)

agent.run("first")
agent.run("second")

assert [m.text for m in provider.calls[1] if m.role.value == "user"] == ["first", "second"]
```

### Assert teams and workflows

```python
from pyagents import sequential
from pyagents.testing import ScriptedProvider, text_response

a = Agent("a", provider=ScriptedProvider([text_response("draft")]))
b = Agent("b", provider=ScriptedProvider([text_response("final")]))

result = sequential(a, b).run("task")
assert result.outputs == {"a": "draft", "b": "final"}
assert [(h.from_agent, h.to_agent) for h in result.handoffs] == [("a", "b")]
```

## Async tests

The whole API is async underneath. With `pytest-asyncio`:

```python
import pytest

@pytest.mark.asyncio
async def test_async_run():
    from pyagents import Agent
    from pyagents.testing import ScriptedProvider, text_response

    agent = Agent("t", provider=ScriptedProvider([text_response("hi")]))
    assert (await agent.arun("hello")).output == "hi"
```

Without an async plugin, call the sync wrappers instead - they behave
identically by design.

## Streamed events

```python
from pyagents.types import EventType

events = list(agent.stream("hello"))
assert [e.type for e in events][0] is EventType.RUN_START
assert events[-1].type is EventType.RUN_END
assert events[-1].raw.output == "hi"
```

`ScriptedProvider(stream_delay=0.01)` adds a small delay per chunk when you need
to test cancellation or timeouts.

## Running PyAgents' own tests

```bash
python -m pytest tests -q                       # ~1 second, no network
python -m pytest tests -q --cov=pyagents        # with coverage
python -m pytest tests/test_agent.py -k hook    # a single area
```

The suite imports the package from `src/` via `tests/conftest.py`, so it works
from a checkout even without `pip install -e .`.
