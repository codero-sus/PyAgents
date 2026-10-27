# PyAgents

**An async-first, zero-dependency framework for building multi-agent LLM applications in Python.**

[![CI](https://github.com/codero-sus/PyAgents/actions/workflows/ci.yml/badge.svg)](https://github.com/codero-sus/PyAgents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typing-mypy%20clean-blue)](src/pyagents/py.typed)

PyAgents gives you one small, well-tested agent loop and composes it into
teams, workflows and pipelines - without hiding the transcript, the tokens or
the tool calls from you.

```python
from pyagents import Agent
from pyagents.tools import calculator

agent = Agent(
    "analyst",
    instructions="You are a careful analyst. Show your reasoning briefly.",
    provider="openai",                 # or "anthropic", "ollama", your own
    model="gpt-4o-mini",
    tools=[calculator],
)

result = agent.run("If I save 250 EUR a month for 3 years, how much do I have?")

print(result.output)                     # the answer
print(result.usage.total_tokens)         # token accounting
print(result.tool_results[0].content)    # what the calculator actually returned
```

## Highlights

| | |
|---|---|
| **Zero required dependencies** | The core runs on the standard library. `httpx` and `pydantic` are optional extras. |
| **Async-first, sync-friendly** | Everything is a coroutine underneath; `run()`, `stream()`, `Team.run()` and `Workflow.run()` are synchronous wrappers that also work inside a running event loop. |
| **Three providers, one interface** | OpenAI (and any compatible endpoint), Anthropic Messages, and local Ollama behind a single `BaseProvider`. Swap with one argument or one env var. |
| **Multi-agent patterns included** | Sequential, parallel and orchestrator teams, deterministic DAG workflows, and agent-as-a-tool composition. |
| **Tools without ceremony** | `@tool` on a typed function is enough - the JSON schema comes from the signature and the docstring, and failures come back as observations instead of exceptions. |
| **Observable by design** | Every run emits `AgentEvent`s: token deltas, tool calls, tool results, handoffs, checkpoints, errors. |
| **Persistent and resumable** | Run stores (`SQLiteRunStore`, `MemoryStore` or your own) checkpoint every step, so crashes and cancellations still leave an auditable record. |
| **Structured output** | Dataclasses or pydantic models, with lenient parsing and one automatic repair attempt. |
| **Testable offline** | `ScriptedProvider`, `FakeTransport` and `EchoProvider` let the entire 300+ test suite run without a network or an API key. |
| **Typed and linted** | Full type hints, PEP 561 `py.typed`, mypy-clean, ruff-formatted, docstrings with runnable examples. |

## Install

```bash
pip install pyagents                 # core, no dependencies
pip install "pyagents[http]"         # + httpx transport
pip install "pyagents[pydantic]"     # + pydantic structured output
pip install "pyagents[docs]"         # + mkdocs toolchain
pip install "pyagents[dev]"          # everything needed to develop PyAgents
```

Requires Python 3.9+.

## Quickstart

### 1. Try it with no configuration

```bash
$ pyagents run "hello there" --provider echo
You said: "hello there". (This is the offline echo provider.)
```

### 2. Point it at a real model

```bash
export OPENAI_API_KEY=sk-...                     # or ANTHROPIC_API_KEY
# or a local model, no key needed:
export PYAGENTS_PROVIDER=ollama PYAGENTS_MODEL=llama3.2

pyagents run "Explain the CAP theorem in two sentences" --verbose
```

From Python:

```python
from pyagents import Agent, get_provider, default_provider

Agent("a", provider="openai", model="gpt-4o-mini")
Agent("b", provider="anthropic", model="claude-3-5-haiku-latest")
Agent("c", provider="ollama", model="llama3.2")
Agent("d", provider=default_provider())        # picks from the environment
```

### 3. Stream events instead of waiting

```python
from pyagents.types import EventType

for event in agent.stream("Write a haiku about asyncio."):
    if event.type is EventType.TEXT:
        print(event.text, end="", flush=True)
    elif event.type is EventType.TOOL_CALL:
        print(f"\n[calling {event.tool_call.name}]")
```

### 4. Compose agents

```python
from pyagents import Agent, Team, Workflow, parallel, sequential

researcher = Agent("researcher", provider="openai", instructions="Find the facts.")
writer = Agent("writer", provider="openai", instructions="Write clearly.")

# pipeline: each agent sees the previous answer
print(sequential(researcher, writer).run("Explain HTTPS simply").output)

# fan-out / fan-in with an aggregator
board = parallel(researcher, writer, aggregator=writer)
board.run("Review our new landing page")

# a manager agent that delegates via handoff tools
manager = Agent("manager", provider="openai", description="Coordinates the team.")
Team([manager, researcher, writer], mode="orchestrator").run("Produce a brief on TLS 1.3")

# or an explicit DAG when you already know the steps
flow = Workflow("report")
flow.add(researcher, name="research")
flow.add(writer, name="write", depends_on=["research"],
         input_template="Task: {task}\nNotes: {research}")
flow.set_output("write")
print(flow.run("Summarise this quarter").output)

# human-in-the-loop review step
from pyagents import Agent, HumanAgent, sequential

editor = HumanAgent("editor", instructions="Approve or rewrite the draft.")
pipeline = sequential(researcher, editor)
pipeline.run("Explain HTTPS simply")
```

### 5. Persist and inspect runs

```python
from pyagents import Agent, SQLiteRunStore

store = SQLiteRunStore("runs.sqlite3")
agent = Agent("worker", provider="openai", store=store)

result = agent.run("Do the thing")
store.load(result.run_id)             # later, even in another process
```

```bash
pyagents runs list --store runs.sqlite3
pyagents runs show run_ab12cd34 --store runs.sqlite3
```

## Command line

```
pyagents run "Summarise this" --tools calculator --json
pyagents chat --provider openai --history 30
pyagents tools list --all --verbose
pyagents providers
pyagents runs list --store runs.sqlite3 --status failed
pyagents info --json
```

`run` supports `--provider`, `--model`, `--base-url`, `--instructions`,
`--tools`, `--max-steps`, `--store`, `--history`, `--no-stream`, `--json`,
`--quiet` and `--verbose`. See [docs/cli.md](docs/cli.md).

## Testing without an API key

```python
from pyagents import Agent
from pyagents.testing import ScriptedProvider, text_response, tool_response
from pyagents.tools import calculator

provider = ScriptedProvider([
    tool_response("calculator", {"expression": "17 * 23 + 100"}),
    text_response("The answer is 491."),
])
agent = Agent("math", provider=provider, tools=[calculator])

result = agent.run("What is 17 * 23 + 100?")
assert result.output == "The answer is 491."
assert provider.last_user_message() == "What is 17 * 23 + 100?"
```

`FakeTransport` does the same at the HTTP layer, which is how PyAgents tests
its own provider adapters. See [docs/testing.md](docs/testing.md).

## Package layout

```
src/pyagents/
├── agent.py        # the single-agent reasoning loop (async, hooks, handoffs)
├── teams.py        # sequential / parallel / orchestrator teams
├── workflows.py    # deterministic DAG pipelines
├── tools/          # @tool, schema inference, validation, built-ins
├── llm/            # provider adapters + pluggable HTTP transports
├── memory.py       # conversation memory strategies
├── stores.py       # run persistence (in-memory + SQLite)
├── parsers.py      # structured-output parsing
├── embeddings.py   # hash / tf-idf / OpenAI embeddings
├── testing.py      # ScriptedProvider, FakeTransport, EchoProvider
└── cli.py          # the `pyagents` console script
```

## Documentation

| page | contents |
|---|---|
| [docs/index.md](docs/index.md) | overview and design goals |
| [docs/quickstart.md](docs/quickstart.md) | ten minutes, no API key |
| [docs/agents.md](docs/agents.md) | the loop, results, events, hooks, cancellation |
| [docs/tools.md](docs/tools.md) | schema inference, validation, built-in tools |
| [docs/multi-agent.md](docs/multi-agent.md) | teams, workflows, choosing a pattern |
| [docs/memory-and-state.md](docs/memory-and-state.md) | memory strategies and run stores |
| [docs/providers.md](docs/providers.md) | OpenAI, Anthropic, Ollama, custom adapters |
| [docs/cli.md](docs/cli.md) | every command and flag |
| [docs/testing.md](docs/testing.md) | fakes and test recipes |
| [docs/api.md](docs/api.md) | API reference |
| [docs/contributing.md](docs/contributing.md) | development workflow |

Runnable, offline examples live in [`examples/`](examples):

| example | shows |
|---|---|
| `01_hello_agent.py` | `run()`, `stream()`, `arun()` |
| `02_tools.py` | custom tools, the tool loop, error handling |
| `03_structured_output.py` | dataclasses, pydantic, automatic repair |
| `04_memory_and_stores.py` | memory strategies, SQLite run stores |
| `05_teams.py` | sequential, parallel and orchestrator teams |
| `06_workflow.py` | DAG workflows with function and agent nodes |
| `07_custom_provider.py` | writing and testing a new provider |
| `08_testing_agents.py` | testing agents with the built-in fakes |
| `09_human_in_the_loop.py` | human-in-the-loop review inside a pipeline |

```bash
python examples/01_hello_agent.py
```

## Development

```bash
git clone https://github.com/codero-sus/PyAgents.git
cd PyAgents
pip install -e ".[dev]"

python -m pytest tests -q          # 300+ tests, no network required
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy                     # strict-ish config in pyproject.toml
python examples/01_hello_agent.py
```

CI runs the same commands on Python 3.9, 3.10, 3.11 and 3.12 for every push and
pull request.

## License

MIT - see [LICENSE](LICENSE).
