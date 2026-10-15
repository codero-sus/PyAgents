"""Giving an agent tools, and watching the tool loop work.

Two things are demonstrated:

1. how little it takes to define a tool - a docstring, type hints, and ``@tool``
2. what happens inside the loop: the model asks for a call, PyAgents runs it,
   feeds the observation back, and the model answers

This example uses a scripted provider so the loop is deterministic and runs
offline. Swap in ``demo_provider()`` (see ``_bootstrap``) to run it against a
real model - the agent code does not change.
"""

from __future__ import annotations

from _bootstrap import section
from pyagents import Agent
from pyagents.testing import ScriptedProvider, text_response, tool_response
from pyagents.tools import Tool, Toolkit, calculator, tool
from pyagents.types import EventType


# --------------------------------------------------------------------------- #
# A custom tool: schema, validation and docs are derived from the signature
# --------------------------------------------------------------------------- #
@tool(tags=["demo"])
def word_stats(text: str) -> dict:
    """Count words and characters in a piece of text.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    return {
        "words": len(words),
        "characters": len(text),
        "longest_word": max(words, key=len) if words else "",
    }


# Tools can also be declared from existing functions or as classes.
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return round(celsius * 9 / 5 + 32, 1)


converter = Tool(
    celsius_to_fahrenheit,
    name="celsius_to_fahrenheit",
    description="Convert Celsius to Fahrenheit.",
)

kit = Toolkit([calculator, word_stats, converter])


def main() -> None:
    section("The generated schemas (this is what the model sees)")
    for entry in (word_stats, calculator):
        print(f"\n{entry.name}: {entry.description}")
        for argument, spec in entry.parameters["properties"].items():
            print(f"  {argument}: {spec.get('type')} ({spec.get('description', 'no description')})")

    section("A scripted tool loop")
    provider = ScriptedProvider(
        [
            # Turn 1: the model decides it needs a tool.
            tool_response("word_stats", {"text": "async python agents are neat"}),
            # Turn 2: it gets the observation back and answers.
            text_response("That sentence has 5 words and 28 characters."),
        ]
    )
    agent = Agent(
        "analyst",
        instructions="Use tools when they help, then answer briefly.",
        provider=provider,
        tools=list(kit),
    )

    result = None
    for event in agent.stream("How long is the sentence 'async python agents are neat'?"):
        if event.type is EventType.TOOL_CALL:
            print(f"  → model calls {event.tool_call}")
        elif event.type is EventType.TOOL_RESULT:
            print(f"  ✓ observation: {event.tool_result.content}")
        elif event.type is EventType.TEXT:
            print(event.text, end="")
        elif event.type is EventType.RUN_END:
            result = event.raw
    print()

    assert result is not None
    section("What the run recorded")
    print(f"steps: {[step.index for step in result.steps]}")
    print(f"tool calls: {result.tool_calls}")
    print(f"tool results: {result.tool_results[0].content}")

    section("Tools fail safely")
    bad = calculator.invoke({"expression": "1 +"})
    print(f"calculator('1 +') -> is_error={bad.is_error} content={bad.content!r}")
    print("The failure is handed back to the model as an observation, not raised.")


if __name__ == "__main__":
    main()
