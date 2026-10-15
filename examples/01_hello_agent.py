"""The smallest useful PyAgents program: one agent, one question.

Run it:

    $ python examples/01_hello_agent.py
    $ python examples/01_hello_agent.py "Explain TCP in one sentence"

Also shows the three ways to invoke an agent:

* ``run()``     - synchronous, returns a ``RunResult``
* ``stream()``  - synchronous iteration of events (token streaming)
* ``arun()``    - the async original that both of the above are built on
"""

from __future__ import annotations

import asyncio

from _bootstrap import demo_provider, section
from pyagents import Agent
from pyagents.types import EventType


def main() -> None:
    task = (
        " ".join(__import__("sys").argv[1:])
        or "Say hello and name three uses for a Python agent framework."
    )

    agent = Agent(
        "greeter",
        instructions="You are concise and friendly. Answer in at most three sentences.",
        provider=demo_provider(),
    )

    section("1. agent.run() — one call, full result")
    result = agent.run(task)
    print(result.output)
    print(
        f"\n[status={result.status} steps={result.iterations} "
        f"tokens={result.usage.total_tokens} duration={result.duration_ms:.0f}ms]"
    )

    section("2. agent.stream() — watch the tokens arrive")
    for event in agent.stream("Write a haiku about async Python."):
        if event.type is EventType.TEXT:
            print(event.text, end="", flush=True)
    print()

    section("3. await agent.arun() — the async API")
    async_result = asyncio.run(agent.arun("Summarise what an agent framework does in one line."))
    print(async_result.output)

    section("The transcript of run #1")
    for message in result.transcript():
        preview = message.text.replace("\n", " ")[:90]
        print(f"  {message.role.value:>9}: {preview}")


if __name__ == "__main__":
    main()
