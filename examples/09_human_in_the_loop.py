"""Human-in-the-loop: a person reviews an agent's answer inside a pipeline.

Run with piped input (no interactive terminal needed):

    $ echo "Approved." | python examples/09_human_in_the_loop.py

Or pass scripted answers directly in Python (useful for tests)::

    from pyagents.human import HumanAgent
    editor = HumanAgent("editor", input=lambda prompt: "Looks good.")
"""

from __future__ import annotations

import sys

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from _bootstrap import demo_provider
from pyagents import Agent, HumanAgent, sequential


def main() -> None:
    provider = demo_provider()
    researcher = Agent(
        "researcher",
        provider=provider,
        instructions="Find the key facts and present them concisely.",
    )
    editor = HumanAgent(
        "editor",
        instructions=(
            "Review the researcher's draft below. "
            "Press Enter to approve it as-is, or type a revised version."
        ),
    )

    pipeline = sequential(researcher, editor)

    print("Researching...")
    result = pipeline.run("Explain the CAP theorem in two sentences.")

    print(f"\nFinal output (from {result.agent}):")
    print(result.output)


if __name__ == "__main__":
    main()
