"""Shared helpers for the example scripts.

Examples are designed to run with **no configuration at all**: when no API key
is present they fall back to the offline echo provider. Set ``PYAGENTS_PROVIDER``
(or ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``) to run them against a real model.

    $ python examples/01_hello_agent.py                      # offline
    $ export OPENAI_API_KEY=sk-...                           # real OpenAI
    $ export PYAGENTS_PROVIDER=ollama PYAGENTS_MODEL=llama3.2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running the examples straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyagents import BaseProvider, default_provider
from pyagents.llm.echo import EchoProvider
from pyagents.types import Message


def has_credentials() -> bool:
    """Return ``True`` when a real provider can be used."""
    return any(
        os.environ.get(name)
        for name in ("PYAGENTS_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_HOST")
    )


def demo_provider(model: str | None = None) -> BaseProvider:
    """Return a real provider when configured, else the offline echo provider."""
    if has_credentials():
        provider = default_provider(model=model)
        print(f"→ using {provider.name}/{provider.model}\n")
        return provider
    print(
        "→ no API key found: using the offline echo provider.\n"
        "  Export OPENAI_API_KEY (or PYAGENTS_PROVIDER=ollama) to use a real model.\n"
    )
    return EchoProvider()


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def echo(prompt: str, *, model: str | None = None) -> str:
    """Run a single prompt through whichever provider is configured."""
    provider = demo_provider(model)
    return provider.complete([Message.user(prompt)]).text
