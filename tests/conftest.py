"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyagents import Agent
from pyagents.memory import BufferMemory
from pyagents.testing import (
    FakeTransport,
    ScriptedProvider,
    text_response,
    tool_response,
)


@pytest.fixture()
def scripted():
    """Factory fixture: ``scripted(resp1, resp2, ...)`` returns a provider."""

    def make(*responses, **kwargs):
        return ScriptedProvider(list(responses), **kwargs)

    return make


@pytest.fixture()
def echo_agent():
    """An agent backed by the offline echo provider."""

    def make(**kwargs):
        kwargs.setdefault("instructions", "Repeat the user.")
        return Agent("echo", provider="echo", **kwargs)

    return make


@pytest.fixture()
def agent(scripted):
    """An agent that answers "hello" once."""

    def make(*responses, **kwargs):
        responses = responses or (text_response("hello"),)
        kwargs.setdefault("provider", scripted(*responses))
        kwargs.setdefault("memory", BufferMemory(max_messages=20))
        return Agent("test-agent", **kwargs)

    return make


@pytest.fixture()
def transport():
    """A fresh :class:`FakeTransport`."""
    return FakeTransport()


@pytest.fixture()
def tool_call_response():
    """Re-exported helper so tests can build tool-call responses quickly."""
    return tool_response
