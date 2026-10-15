"""Deterministic pipelines with ``Workflow``.

Teams are for *collaboration*; workflows are for *control flow*. A workflow is a
DAG whose nodes are agents or plain functions, executed in dependency order,
with every node's output available to the nodes that follow it.

Run it:

    $ python examples/06_workflow.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from _bootstrap import section
from pyagents import Agent, Workflow
from pyagents.testing import ScriptedProvider, text_response
from pyagents.types import EventType
from pyagents.workflows import WorkflowContext


# --------------------------------------------------------------------------- #
# Plain functions are first-class nodes
# --------------------------------------------------------------------------- #
def extract_metrics(context: WorkflowContext) -> dict:
    """Pull numbers out of the research note (no LLM involved)."""
    text = context.get("research", "")
    numbers = [int(value) for value in re.findall(r"\b\d+\b", text)]
    return {"values": numbers, "total": sum(numbers)}


def render_report(context: WorkflowContext) -> str:
    """Format the final markdown from upstream outputs."""
    metrics = context.get("metrics", {})
    return (
        "# Weekly report\n\n"
        f"Summary: {context.get('summary', '')}\n\n"
        f"Numbers found: {metrics.get('values')}\n"
        f"Total: {metrics.get('total')}\n\n"
        f"{context.get('narrative', '')}\n"
    )


@dataclass
class RunStats:
    nodes: int
    events: int


def main() -> None:
    research_agent = Agent(
        "researcher",
        instructions="Extract facts and numbers.",
        provider=ScriptedProvider(
            [text_response("Three teams shipped 12, 7 and 5 features this week.")]
        ),
    )
    narrate_agent = Agent(
        "narrator",
        instructions="Write one clear sentence.",
        provider=ScriptedProvider(
            [text_response("Delivery is trending up across all three teams.")]
        ),
    )

    workflow = Workflow("weekly-report", context={"owner": "platform-team"})
    workflow.add(research_agent, name="research")
    workflow.add(extract_metrics, name="metrics", depends_on=["research"])
    workflow.add(
        narrate_agent,
        name="narrative",
        depends_on=["research"],
        input_template="Task: {task}\nNotes: {research}\nWrite the single-sentence takeaway.",
    )
    workflow.add(
        render_report,
        name="report",
        depends_on=["metrics", "narrative"],
    )
    workflow.set_output("report")

    section("Execution order (topologically sorted)")
    print(" → ".join(workflow.order()))

    section("Streaming the workflow")
    for event in workflow.stream("Summarise this week"):
        node_name = event.data.get("node")
        if event.type is EventType.STEP_START and node_name:
            print(f"  ▸ node {node_name} started")
        elif event.type is EventType.MESSAGE and node_name:
            print(f"  ✓ node {node_name} produced: {event.data['output'][:60]!r}")

    result = workflow.run("Summarise this week")

    section("Node outputs")
    print(json.dumps({key: str(value)[:60] for key, value in result.outputs.items()}, indent=2))

    section("Final document")
    print(result.output)

    section("What the narrator node actually saw")
    print(narrate_agent.provider.calls[-1][-1].text)

    section("Custom node result")
    print(RunStats(nodes=len(workflow.nodes), events=6))

    section("Registering nodes with the decorator")
    print("@node(name=..., depends_on=[...]) attaches a spec to a function:")
    print(
        f"  render_report.workflow_node = {getattr(render_report, 'workflow_node', 'set it with @node')}"
    )


if __name__ == "__main__":
    main()
