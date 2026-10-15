"""PyAgents - an async-first framework for multi-agent LLM applications.

Quick start::

    from pyagents import Agent
    from pyagents.tools import calculator

    agent = Agent(
        "analyst",
        instructions="You are a careful analyst.",
        provider="openai",            # or "anthropic", "ollama", or a provider instance
        model="gpt-4o-mini",
        tools=[calculator],
    )
    result = agent.run("What is 17 * 23 + 100?")
    print(result.output)

Package layout:

* :mod:`pyagents.agent` - the single-agent reasoning loop
* :mod:`pyagents.teams` - sequential, parallel and orchestrator teams
* :mod:`pyagents.workflows` - deterministic multi-step pipelines
* :mod:`pyagents.tools` - tool definitions, schema inference and built-ins
* :mod:`pyagents.llm` - OpenAI, Anthropic and Ollama providers
* :mod:`pyagents.memory` - conversation memory strategies
* :mod:`pyagents.stores` - run persistence and checkpointing
* :mod:`pyagents.parsers` - structured output parsing
* :mod:`pyagents.testing` - fakes for tests (no network required)
"""

from __future__ import annotations

from .agent import Agent
from .errors import (
    AgentError,
    ConfigurationError,
    MaxStepsExceeded,
    OrchestrationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    PyAgentsError,
    RunFailed,
    RunNotFound,
    StructuredOutputError,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
)
from .llm import (
    PROVIDERS,
    AnthropicProvider,
    BaseProvider,
    LLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    default_provider,
    get_provider,
    register_provider,
)
from .memory import (
    BufferMemory,
    CompositeMemory,
    Memory,
    SummaryMemory,
    TokenWindowMemory,
    VectorMemory,
    WindowMemory,
    memory_from_dict,
)
from .parsers import AutoParser, JsonParser, ListParser, Parser, RegexParser, parser_for
from .stores import MemoryStore, RunStore, SQLiteRunStore, store_from_uri
from .teams import Team, TeamResult, parallel, sequential
from .tools import Tool, Toolkit, default_toolkit, default_tools, tool
from .types import (
    AgentEvent,
    EventType,
    Handoff,
    LLMResponse,
    Message,
    Role,
    RunResult,
    RunStatus,
    Step,
    StreamChunk,
    ToolCall,
    ToolResult,
    Usage,
    new_id,
)
from .workflows import Workflow, WorkflowContext, WorkflowResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "Agent",
    "Team",
    "TeamResult",
    "Workflow",
    "WorkflowResult",
    "WorkflowContext",
    "sequential",
    "parallel",
    # tools
    "Tool",
    "Toolkit",
    "tool",
    "default_tools",
    "default_toolkit",
    # providers
    "BaseProvider",
    "LLMProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "PROVIDERS",
    "get_provider",
    "default_provider",
    "register_provider",
    # memory + persistence
    "Memory",
    "BufferMemory",
    "WindowMemory",
    "TokenWindowMemory",
    "SummaryMemory",
    "VectorMemory",
    "CompositeMemory",
    "memory_from_dict",
    "RunStore",
    "MemoryStore",
    "SQLiteRunStore",
    "store_from_uri",
    # parsing
    "Parser",
    "AutoParser",
    "JsonParser",
    "ListParser",
    "RegexParser",
    "parser_for",
    # types
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "Usage",
    "LLMResponse",
    "StreamChunk",
    "Step",
    "Handoff",
    "RunResult",
    "RunStatus",
    "AgentEvent",
    "EventType",
    "new_id",
    # errors
    "PyAgentsError",
    "ConfigurationError",
    "AgentError",
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "MaxStepsExceeded",
    "RunFailed",
    "RunNotFound",
    "StructuredOutputError",
    "OrchestrationError",
]


def __getattr__(name: str):
    """Lazily expose submodules such as ``pyagents.testing``.

    Keeping :mod:`pyagents.testing` out of the import graph until it is asked
    for makes the core package import a little faster.
    """
    if name in {"testing", "embeddings", "utils", "cli"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
