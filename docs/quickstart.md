# Quickstart

Ten minutes, no API key required.

## 1. Install

```bash
pip install pyagents
# optional extras
pip install "pyagents[http]"       # faster async HTTP transport
pip install "pyagents[pydantic]"   # pydantic-validated structured output
```

PyAgents requires Python 3.9+ and has **no required dependencies** - the core
uses `urllib` from the standard library.

## 2. Pick a provider

The shortest path is the offline echo provider, which needs no configuration:

```python
from pyagents import Agent

agent = Agent("demo", provider="echo")
print(agent.run("hello").output)
# You said: "hello". (This is the offline echo provider.)
```

For a real model, either pass credentials explicitly or let PyAgents find them
in the environment:

```python
# OpenAI
Agent("demo", provider="openai", model="gpt-4o-mini")        # OPENAI_API_KEY

# Anthropic
Agent("demo", provider="anthropic", model="claude-3-5-haiku-latest")  # ANTHROPIC_API_KEY

# Local Ollama (no key needed)
Agent("demo", provider="ollama", model="llama3.2")           # OLLAMA_HOST, default localhost:11434
```

Any OpenAI-compatible endpoint (vLLM, LM Studio, OpenRouter, ...) works too:

```python
from pyagents import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "my-model", base_url="http://localhost:8000/v1", api_key="not-needed"
)
```

## 3. Run it

```python
result = agent.run("Explain what an agent loop is in two sentences.")

result.output           # the final text
result.status           # RunStatus.COMPLETED
result.iterations       # how many model calls happened
result.usage            # prompt / completion / total tokens
result.steps            # every Step: message, tool calls, tool results
result.metadata         # timing, provider name, structured-output info
result.transcript()     # the full Message list sent to the model
```

Prefer events while the run is still going:

```python
from pyagents.types import EventType

for event in agent.stream("Write a haiku about asyncio."):
    if event.type is EventType.TEXT:
        print(event.text, end="", flush=True)
    elif event.type is EventType.TOOL_CALL:
        print(f"\n[calling {event.tool_call.name}]")
```

And the async original, if you are already in an event loop:

```python
import asyncio

result = asyncio.run(agent.arun("hello"))
```

## 4. Give it tools

Type hints and a docstring are the schema:

```python
from pyagents import Agent
from pyagents.tools import calculator, tool

@tool
def convert(amount: float, currency: str = "USD") -> str:
    """Convert an amount into the given currency.

    Args:
        amount: How much money to convert.
        currency: Target currency code.
    """
    rates = {"USD": 1.0, "EUR": 0.92, "INR": 83.4}
    return f"{amount * rates.get(currency, 1.0):.2f} {currency}"

agent = Agent("finance", provider="openai", tools=[calculator, convert])
print(agent.run("Convert 40 EUR into INR using the rates you have.").output)
```

See [Tools](tools.md) for validation, error handling and the built-in library.

## 5. Get structured data back

Any dataclass or pydantic model works:

```python
import dataclasses
from typing import List, Optional
from pyagents import Agent

@dataclasses.dataclass
class Ticket:
    title: str
    priority: str
    components: List[str]
    assignee: Optional[str] = None

agent = Agent("triage", provider="openai", output_model=Ticket)
ticket = agent.run("Extract a ticket from: 'Checkout page 500s on Safari.'").structured

assert isinstance(ticket, Ticket)
print(ticket.priority, ticket.components)
```

Parsing is tolerant (fenced code blocks, surrounding prose) and repairs itself
once if the model returns malformed JSON - see
[Agents → structured output](agents.md#structured-output).

## 6. Add memory

```python
from pyagents import Agent, BufferMemory

memory = BufferMemory(max_messages=20)
agent = Agent("assistant", provider="openai", memory=memory)

agent.run("My name is Ada.")
agent.run("What is my name?")     # the second call still knows
```

## 7. Compose agents

Teams come in three shapes:

```python
from pyagents import Agent, Team, parallel, sequential

researcher  = Agent("researcher", provider="openai", instructions="Find facts.")
writer      = Agent("writer",     provider="openai", instructions="Write clearly.")
reviewer    = Agent("reviewer",   provider="openai", instructions="Check for errors.")

# 1. pipeline: each agent sees the previous answer
pipeline = sequential(researcher, writer, reviewer)
print(pipeline.run("Explain HTTPS to a non-technical reader").output)

# 2. fan-out / fan-in
board = parallel(researcher, writer, reviewer, aggregator=reviewer)
board.run("Review our new landing page")

# 3. manager with handoff tools
manager = Agent("manager", provider="openai", description="Coordinates the team.")
team = Team([manager, researcher, writer], mode="orchestrator")
team.run("Produce a one-page brief on TLS 1.3")
```

Deterministic pipelines are a `Workflow`:

```python
from pyagents import Workflow

flow = Workflow("report")
flow.add(researcher, name="research")
flow.add(writer, name="write", depends_on=["research"],
         input_template="Task: {task}\nNotes: {research}")
flow.set_output("write")

print(flow.run("Summarise this quarter").output)
```

## 8. Persist runs

```python
from pyagents import Agent, SQLiteRunStore

store = SQLiteRunStore("runs.sqlite3")
agent = Agent("worker", provider="openai", store=store)

result = agent.run("Do the thing")
store.load(result.run_id)        # later, even in another process
```

From the shell:

```bash
pyagents runs list  --store runs.sqlite3
pyagents runs show  run_ab12cd34 --store runs.sqlite3
```

## 9. Next steps

* [Agents](agents.md) - hooks, cancellation, max steps, error handling
* [Tools](tools.md) - the `Tool` class, `Toolkit`, built-ins
* [Multi-agent patterns](multi-agent.md) - when to use which team shape
* [Testing agents](testing.md) - write fast tests with `ScriptedProvider`
* [Command line](cli.md) - everything the CLI can do
