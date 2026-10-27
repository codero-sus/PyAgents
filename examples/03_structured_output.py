"""Getting typed Python objects out of a model instead of loose text.

Set ``output_model`` on an agent and PyAgents:

* appends format instructions (derived from the schema) to the system prompt
* parses the final answer, tolerating fenced code blocks and prose
* retries once with the error message when parsing fails
* exposes the object as ``result.structured``

Both dataclasses and pydantic models work; pydantic gets full validation.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from _bootstrap import section
from pyagents import Agent
from pyagents.testing import ScriptedProvider, text_response


@dataclasses.dataclass
class Ticket:
    """A support ticket extracted from free-form text."""

    title: str
    priority: str
    components: List[str]
    assignee: Optional[str] = None


def main() -> None:
    section("1. Dataclass output (works with no extra dependencies)")
    provider = ScriptedProvider(
        [
            text_response(
                """
                Sure! Here is the structured ticket:
                ```json
                {"title": "Login fails on Safari", "priority": "high",
                 "components": ["auth", "frontend"], "assignee": "ada"}
                ```
                """
            )
        ]
    )
    agent = Agent("triage", provider=provider, output_model=Ticket)
    result = agent.run("Extract a ticket from: 'Safari users cannot log in since Tuesday.'")

    ticket = result.structured
    assert isinstance(ticket, Ticket)
    print(f"title:      {ticket.title}")
    print(f"priority:   {ticket.priority}")
    print(f"components: {ticket.components}")
    print(f"assignee:   {ticket.assignee}")

    section("2. The instructions the model received")
    print(result.messages[0].text)

    section("3. Automatic repair when the model returns junk")
    retrying = Agent(
        "triage",
        provider=ScriptedProvider(
            [
                text_response("I could not do that."),  # first attempt: not JSON
                text_response(
                    '{"title": "Slow dashboard", "priority": "low", "components": ["ui"]}'
                ),
            ]
        ),
        output_model=Ticket,
    )
    repaired = retrying.run("Extract a ticket.")
    print(f"recovered on attempt 2: {repaired.structured}")
    print(f"steps taken: {len(repaired.steps)}")

    section("4. When parsing really fails, nothing explodes")
    broken = Agent(
        "triage",
        provider=ScriptedProvider([text_response("nope"), text_response("still nope")]),
        output_model=Ticket,
    )
    failed = broken.run("Extract a ticket.")
    print(f"structured: {failed.structured}")
    print(f"output kept as text: {failed.output!r}")
    print(f"recorded error: {failed.metadata['structured_error']}")

    section("5. Pydantic models (optional extra)")
    try:
        import pydantic
    except ImportError:
        print("pydantic is not installed: `pip install 'pyagents[pydantic]'`")
        return

    class Person(pydantic.BaseModel):
        name: str
        age: int

    typed = Agent(
        "extract",
        provider=ScriptedProvider([text_response('{"name": "Ada Lovelace", "age": "36"}')]),
        output_model=Person,
    )
    person = typed.run("Who wrote the first algorithm?").structured
    print(f"validated by pydantic: {person!r} (age coerced to {type(person.age).__name__})")


if __name__ == "__main__":
    main()
