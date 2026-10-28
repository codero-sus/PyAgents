# Multi-agent patterns

PyAgents ships three composition styles for agents, plus a deterministic
workflow engine for when you want to control the graph yourself.

| pattern | good for | control flow |
|---|---|---|
| `sequential` | pipelines: research → write → review | fixed order, previous output available |
| `parallel` | independent opinions, fan-out/fan-in | all at once, then aggregate |
| `orchestrator` | open-ended tasks needing judgement | a manager agent decides via handoff tools |
| `Workflow` | reproducible DAGs with typed steps | declared by `depends_on` |

## Sequential teams

```python
from pyagents import Agent, sequential

researcher = Agent("researcher", provider="openai", instructions="Find the facts.")
writer = Agent("writer", provider="openai", instructions="Write for a general audience.")
reviewer = Agent("reviewer", provider="openai", instructions="Fix errors, keep it short.")

pipeline = sequential(researcher, writer, reviewer)
result = pipeline.run("Explain HTTPS to a non-technical reader")

result.output            # the last agent's answer
result.outputs           # {"researcher": "...", "writer": "...", "reviewer": "..."}
result.agent_results     # the individual RunResult objects
result.handoffs          # [Handoff(researcher -> writer), Handoff(writer -> reviewer)]
result.usage             # token usage summed over every agent
```

By default each agent receives the previous agent's output as context
(`pass_context=True`). With `turns=N` the pipeline loops N times, feeding each
round's result into the next - useful for iterative refinement:

```python
polish = sequential(writer, reviewer, turns=3)
```

## Parallel teams

```python
from pyagents import Agent, parallel

legal = Agent("legal", provider="openai", instructions="Flag legal risk.")
security = Agent("security", provider="openai", instructions="Flag security risk.")
product = Agent("product", provider="openai", instructions="Flag UX risk.")
chair = Agent("chair", provider="openai", instructions="Synthesise one recommendation.")

board = parallel(legal, security, product, aggregator=chair, max_concurrency=8)
result = board.run("Review our new signup flow")

result.output                    # the aggregator's synthesis
[run.agent for run in result.agent_results]   # who contributed
```

Without an `aggregator` the outputs are concatenated with headers. Every member
runs on its own; a member that raises is recorded in
`result.metadata["errors"][agent_name]` instead of aborting the whole batch.

## Orchestrator teams

Give the manager description-bearing agents and it gets a `transfer_to_<name>`
tool for each of them:

```python
from pyagents import Agent, Team

manager = Agent(
    "manager",
    provider="openai",
    description="Coordinates specialists and writes the final answer.",
)
researcher = Agent("researcher", provider="openai", description="Finds facts online.")
analyst = Agent("analyst", provider="openai", description="Does quantitative analysis.")

team = Team([manager, researcher, analyst], mode="orchestrator", name="briefing")
result = team.run("Produce a one-page brief on the state of solid-state batteries")

result.handoffs      # which specialist the manager handed off to, and why
result.agent_results # each specialist's RunResult
```

Handoff tools carry `task` (what to do) and `reason` (why) arguments, and the
manager can chain handoffs. Cycles are rejected at construction time, so a team
cannot bounce forever between two agents.

## Composition: agents as tools

Any agent can be exposed to another agent as a tool - useful when you want
sub-agents without a team object:

```python
expert = Agent("expert", provider="openai", instructions="Answer networking questions.")
generalist = Agent("generalist", provider="openai", tools=[])

generalist.add_tool(expert.as_tool(description="Ask the networking expert."))
generalist.run("How large is an IPv6 address?")
```

The expert runs a full agent loop behind the scenes and its final answer is
returned to the caller as a tool observation.

## Workflows: explicit DAGs

When the order of steps is known up front, a `Workflow` is simpler and cheaper
than an orchestrator: no model calls are spent deciding what to do next.

```python
from pyagents import Agent, Workflow

research = Agent("researcher", provider="openai")
summarise = Agent("summariser", provider="openai")

flow = Workflow("weekly-report", context={"owner": "platform-team"})
flow.add(research, name="research")
flow.add(
    summarise,
    name="summary",
    depends_on=["research"],
    input_template="Task: {task}\nNotes: {research}\nWrite one paragraph.",
)
flow.set_output("summary")

result = flow.run("Summarise this week")
result.outputs["research"]     # every node's raw output
result.output                  # the node marked with set_output()
```

Nodes may be agents or plain callables. Plain callables receive a
`WorkflowContext` (or the raw task string if they only take one argument) and
can read upstream results with `context.get("research")` and share state via
`context.shared`. This is how you mix deterministic data work with LLM steps:

```python
def metrics(context) -> dict:
    numbers = extract_numbers(context.get("research"))
    context.shared["seen_metrics"] = True
    return {"values": numbers, "total": sum(numbers)}

flow.add(metrics, name="metrics", depends_on=["research"])
```

Declarative form:

```python
from pyagents.workflows import node

@node(name="metrics", depends_on=["research"])
def metrics(context):
    ...
```

Errors are configuration errors, not surprises: unknown nodes, duplicate names,
dependency cycles and missing inputs all raise `ConfigurationError` before
anything runs.

## Persistence and observability

Teams and workflows accept a `store=` just like agents - the composite result
is saved under its own run id:

```python
from pyagents import SQLiteRunStore, sequential

store = SQLiteRunStore("runs.sqlite3")
pipeline = sequential(researcher, writer, store=store)
result = pipeline.run("...")
store.load(result.run_id).output
```

Both emit the same events as single agents, so one streaming loop can drive a
progress UI:

```python
from pyagents.types import EventType

for event in pipeline.stream("..."):
    if event.type is EventType.AGENT_START:
        print(f"[{event.agent}] started")
    elif event.type is EventType.TEXT:
        print(event.text, end="")
    elif event.type is EventType.AGENT_END:
        print(f"[{event.agent}] done: {event.text[:60]}")
```

`HANDOFF` events appear when an agent delegates through a handoff tool (the
orchestrator pattern and `Agent.as_tool`); sequential pipelines record the same
information as `Handoff` entries on the result.

## Human-in-the-loop

Sometimes the best "agent" is a person. `HumanAgent` lets a human review,
approve or rewrite an agent's answer inside a team or pipeline, without
requiring an LLM call for that step::

    from pyagents import Agent, HumanAgent, sequential

    researcher = Agent("researcher", instructions="Find the facts.")
    editor = HumanAgent("editor", instructions="Approve or rewrite the draft.")

    pipeline = sequential(researcher, editor)
    result = pipeline.run("Write a brief on HTTPS.")

The human's response becomes `editor`'s output and flows into the next team
member exactly like any other agent result. Use it for compliance review,
fact-checking, or any step where a person needs to be in the loop.

`HumanAgent` supports the same interface as `Agent` (`run`, `arun`, `stream`,
`astream`, `call`, `acall`) and works in sequential, parallel and orchestrator
teams. It reads from `sys.stdin` by default, or you can pass an `input=`
callable for scripted or programmatic responses::

    from pyagents.human import HumanAgent

    reviewer = HumanAgent("reviewer", input=lambda prompt: "Approved.")

From the CLI, pass `--human-review` to `pyagents run` to prompt a human to
approve or edit the agent's answer before the command finishes::

    pyagents run "Write a brief on HTTPS" --human-review

## Which pattern should I use?

- **One agent with tools** - start here. Most "multi-agent" problems are one
  agent with better tools.
- **`sequential`** - the stages are genuinely different jobs and each one needs
  the previous result (translation, summarisation, review).
- **`parallel`** - you want independent attempts or perspectives, then a
  synthesis (risk review, brainstorming, ensembling).
- **`orchestrator`** - the number and order of steps depends on what the model
  finds; you trust a manager to delegate.
- **`Workflow`** - you already know the steps and want reproducibility,
  caching, retries at the node level, or a mix of code and LLM work.
