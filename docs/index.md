# PyAgents

**PyAgents is an async-first, zero-dependency framework for building multi-agent
LLM applications in Python.** It gives you one small, well-tested agent loop and
composes it into teams, workflows and pipelines - without hiding the transcript,
the tokens or the tool calls from you.

```python
from pyagents import Agent
from pyagents.tools import calculator

agent = Agent(
    "analyst",
    instructions="You are a careful analyst. Show your reasoning briefly.",
    provider="openai",            # or "anthropic", "ollama", or your own
    model="gpt-4o-mini",
    tools=[calculator],
)

result = agent.run("If I save 250 EUR a month for 3 years, how much do I have?")

print(result.output)          # the answer
print(result.usage.total_tokens)
print(result.tool_results[0].content)   # what the calculator actually returned
```

## Why PyAgents

| | |
|---|---|
| **Zero required dependencies** | The core runs on the standard library alone. `httpx` and `pydantic` are optional extras. |
| **Async-first, sync-friendly** | Everything is implemented as coroutines; `run()`, `stream()`, `Team.run()` and `Workflow.run()` are synchronous wrappers that even work inside a running event loop. |
| **Three providers, one interface** | OpenAI (and every compatible API), Anthropic Messages, and local Ollama. Swap with one argument. |
| **Observable by design** | Every run emits `AgentEvent`s: token deltas, tool calls, tool results, handoffs, checkpoints, errors. |
| **Tools without ceremony** | `@tool` on a typed function is enough - the JSON schema comes from the signature and docstring. |
| **Patterns included** | Sequential, parallel and orchestrator teams; deterministic DAG workflows; agent-as-tool composition. |
| **Testable offline** | `ScriptedProvider`, `FakeTransport` and `EchoProvider` let the whole test suite run without a network. |
| **Typed and checked** | Full type hints, `py.typed`, mypy-clean, 300+ tests, ruff-formatted. |

## Install

```bash
pip install pyagents                 # core, no dependencies
pip install "pyagents[http]"         # + httpx transport
pip install "pyagents[pydantic]"     # + pydantic structured output
pip install "pyagents[dev]"          # everything needed to develop PyAgents
```

## Try it without an API key

```bash
$ pyagents run "hello there" --provider echo
You said: "hello there". (This is the offline echo provider.)
```

Or from Python:

```python
from pyagents.testing import ScriptedProvider, tool_response, text_response
from pyagents.tools import calculator
from pyagents import Agent

provider = ScriptedProvider([
    tool_response("calculator", {"expression": "17 * 23 + 100"}),
    text_response("The answer is 491."),
])

agent = Agent("math", provider=provider, tools=[calculator])
print(agent.run("What is 17 * 23 + 100?").output)
```

## Where to go next

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** - from install to a multi-agent pipeline in ten minutes
- **[Agents](agents.md)** - the reasoning loop, hooks, structured output, events
- **[Tools](tools.md)** - schemas from type hints, validation, built-in tools
- **[Multi-agent patterns](multi-agent.md)** - teams and workflows
- **[Memory & state](memory-and-state.md)** - memory strategies and run stores
- **[Providers](providers.md)** - OpenAI, Anthropic, Ollama, and custom adapters
- **[Command line](cli.md)** - `pyagents run`, `chat`, `runs`, and friends
- **[Testing agents](testing.md)** - fakes, recording providers and assertions

</div>

## Package layout

```
src/pyagents/
├── agent.py        # the single-agent reasoning loop
├── teams.py        # sequential / parallel / orchestrator teams
├── workflows.py    # deterministic DAG pipelines
├── tools/          # @tool, schema inference, built-in tools
├── llm/            # provider adapters + transports
├── memory.py       # conversation memory strategies
├── stores.py       # run persistence (memory + SQLite)
├── parsers.py      # structured output parsing
├── embeddings.py   # hash / tf-idf / OpenAI embeddings
├── testing.py      # ScriptedProvider, FakeTransport, EchoProvider
└── cli.py          # the `pyagents` console script
```

## License

MIT - see [LICENSE](https://github.com/codero-sus/PyAgents/blob/main/LICENSE).
