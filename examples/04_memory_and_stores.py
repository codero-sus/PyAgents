"""Conversation memory and durable run history.

Memory answers "what should the model see next time?" and run stores answer
"what happened last time?". This example shows both, plus how to resume
inspection of a stored run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _bootstrap import section
from pyagents import (
    Agent,
    BufferMemory,
    MemoryStore,
    SQLiteRunStore,
    SummaryMemory,
    TokenWindowMemory,
)
from pyagents.testing import ScriptedProvider, text_response


def main() -> None:
    section("1. Memory gives an agent continuity between runs")
    memory = BufferMemory(max_messages=10)
    provider = ScriptedProvider(
        [text_response("Hi Ada!"), text_response("You asked about Python first.")]
    )
    agent = Agent("assistant", provider=provider, memory=memory)

    agent.run("My name is Ada and I love Python.")
    agent.run("What did I say I love?")

    print("memory contents:")
    for message in memory.messages():
        print(f"  {message.role.value:>9}: {message.text}")

    print("\nsecond call saw:")
    for message in provider.calls[-1]:
        print(f"  {message.role.value:>9}: {message.text[:70]}")

    section("2. Memory strategies are interchangeable")
    strategies = {
        "BufferMemory": BufferMemory(max_messages=2),
        "TokenWindowMemory": TokenWindowMemory(max_tokens=40),
    }
    for name, strategy in strategies.items():
        for index in range(6):
            strategy.add(text_response(f"message {index}").message)
        kept = [message.text.split()[-1] for message in strategy.messages()]
        print(f"  {name:<20} keeps messages: {kept}")

    print("\n  SummaryMemory compresses instead of dropping:")
    summariser = SummaryMemory(
        ScriptedProvider([text_response("The user discussed messages 0-2.")]),
        summary_threshold=2,
        keep_recent=2,
    )
    for index in range(6):
        summariser.add(text_response(f"topic {index}").message)
    for message in summariser.messages():
        print(f"    {message.role.value:>9}: {message.text}")

    section("3. Run stores: inspect what happened")
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteRunStore(Path(tmp) / "runs.sqlite3")
        agent = Agent(
            "worker",
            provider=ScriptedProvider([text_response("finished the job")]),
            store=store,
        )
        result = agent.run("do the job")

        print(f"saved run {result.run_id}")
        for row in store.list_runs():
            print(f"  {row['run_id']}  {row['status']:<10} {row['output'][:40]!r}")

        reloaded = store.load(result.run_id)
        print(
            f"\nreloaded from SQLite: status={reloaded.status} tokens={reloaded.usage.total_tokens}"
        )
        print(f"steps recorded: {len(reloaded.steps)}")
        store.close()

    section("4. In-memory store is the same API")
    quick = MemoryStore()
    Agent("temp", provider=ScriptedProvider([text_response("ephemeral")]), store=quick).run("hi")
    print(f"runs in memory: {len(quick.list_runs())}")

    section("Shell access to a store")
    print("Any SQLite store can be inspected from the CLI:")
    print("  pyagents runs list  --store runs.sqlite3")
    print("  pyagents runs show  <run_id> --store runs.sqlite3")
    print("  pyagents runs delete <run_id> --store runs.sqlite3")


if __name__ == "__main__":
    main()
