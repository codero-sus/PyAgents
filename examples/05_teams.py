"""Three ways to make several agents work on one problem.

* **sequential** - a pipeline; each agent sees the previous agent's answer
* **parallel** - everyone works on the same task at once, then an aggregator
  (or plain concatenation) combines the results
* **orchestrator** - a manager agent decides who does what through generated
  ``transfer_to_*`` handoff tools

Everything here runs offline with scripted providers; swap in
``demo_provider()`` for the real thing.
"""

from __future__ import annotations

from _bootstrap import section
from pyagents import Agent, Team, parallel, sequential
from pyagents.testing import ScriptedProvider, text_response, tool_response
from pyagents.types import EventType


def make(name: str, *answers: str, **kwargs) -> Agent:
    """Create an agent with scripted answers."""
    provider = ScriptedProvider([text_response(answer) for answer in answers])
    return Agent(name, provider=provider, **kwargs)


def main() -> None:
    section("1. Sequential team: research → write → review")
    researcher = make("researcher", "Facts: HTTPS encrypts traffic with TLS.")
    writer = make("writer", "HTTPS keeps your browsing private by encrypting traffic.")
    reviewer = make("reviewer", "Approved: accurate and readable.")
    pipeline = sequential(researcher, writer, reviewer)

    result = pipeline.run("Explain HTTPS to a non-technical reader")
    print(f"final output: {result.output}")
    print(f"per-agent outputs: {result.outputs}")
    print(f"handoff chain: {[(h.from_agent, h.to_agent) for h in result.handoffs]}")
    print(f"total tokens: {result.usage.total_tokens}")

    print("\nThe writer received the researcher's output:")
    print(f"  {writer.provider.calls[-1][-1].text[:120]}…")

    section("2. Parallel team: three reviewers, one synthesis")
    legal = make("legal", "Legal risk: low, add a privacy note.")
    security = make("security", "Security: enable HSTS.")
    product = make("product", "Product: keep the onboarding copy short.")
    aggregator = make("chair", "Recommendation: enable HSTS, add a privacy note, keep copy short.")

    team = parallel(legal, security, product, aggregator=aggregator)
    verdict = team.run("Review our new login page")
    print(f"aggregated answer: {verdict.output}")
    print(f"contributors: {[member.agent for member in verdict.agent_results]}")

    section("3. Orchestrator team: a manager delegating work")
    manager = Agent(
        "manager",
        description="Coordinates specialists.",
        provider=ScriptedProvider(
            [
                tool_response(
                    "transfer_to_math_specialist",
                    {"task": "Compute 24 * 7", "reason": "arithmetic"},
                ),
                text_response("The answer is 168."),
            ]
        ),
    )
    specialist = make("math_specialist", "24 * 7 = 168")
    orchestra = Team([manager, specialist], mode="orchestrator", name="support")

    print("manager toolbelt:", [spec["name"] for spec in manager.tool_specs()])
    print("\nevents during the orchestrated run:")
    for event in orchestra.stream("What is 24 * 7?"):
        if event.type is EventType.HANDOFF:
            print(f"  handoff: {event.text}")
        elif event.type is EventType.AGENT_START:
            print(f"  agent start: {event.agent}")
        elif event.type is EventType.TOOL_RESULT:
            print(f"  observation: {event.tool_result.content}")

    final = orchestra.run("What is 24 * 7?")
    print(f"\nfinal: {final.output}")
    print(f"members that ran: {[member.agent for member in final.agent_results]}")

    section("4. Agent-as-a-tool (no team required)")
    expert = make("expert", "IPv6 uses 128-bit addresses.")
    caller = Agent(
        "generalist",
        provider=ScriptedProvider(
            [
                tool_response("expert", {"task": "How long are IPv6 addresses?"}),
                text_response("128 bits."),
            ]
        ),
    )
    caller.add_tool(expert.as_tool(description="Answers networking questions."))
    print(caller.run("How long are IPv6 addresses?").output)


if __name__ == "__main__":
    main()
