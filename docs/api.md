# API reference

Every public name is importable from the top-level `pyagents` package:

```python
from pyagents import Agent, Team, Workflow, Tool, tool, SQLiteRunStore
```

The reference below is generated from the source, so signatures and docstrings
always match the released code.

## Agents

::: pyagents.agent
    options:
      members:
        - Agent
        - HOOK_NAMES
      show_root_heading: true
      members_order: source

## Teams

::: pyagents.teams
    options:
      members:
        - Team
        - TeamResult
        - sequential
        - parallel
        - MODES

## Workflows

::: pyagents.workflows
    options:
      members:
        - Workflow
        - WorkflowResult
        - WorkflowContext
        - node

## Tools

::: pyagents.tools.base
    options:
      members:
        - Tool
        - Toolkit
        - FunctionTool
        - tool
        - json_schema_from_function
        - validate_arguments
        - coerce_arguments
        - run_tool_call
        - run_tool_call_sync
        - ensure_tool

## Built-in tools

::: pyagents.tools.builtin
    options:
      members:
        - calculator
        - current_time
        - read_file
        - write_file
        - list_directory
        - http_get
        - http_request
        - python_repl
        - default_tools
        - default_toolkit

## Providers

::: pyagents.llm.base
    options:
      members:
        - BaseProvider
        - LLMProvider

::: pyagents.llm.openai
    options:
      members:
        - OpenAIProvider
        - OpenAICompatibleProvider

::: pyagents.llm.anthropic
    options:
      members:
        - AnthropicProvider

::: pyagents.llm.ollama
    options:
      members:
        - OllamaProvider

::: pyagents.llm
    options:
      members:
        - PROVIDERS
        - get_provider
        - default_provider
        - register_provider

## Transports

::: pyagents.llm.transport
    options:
      members:
        - HttpTransport
        - UrllibTransport
        - HttpxTransport
        - get_default_transport

## Memory

::: pyagents.memory
    options:
      members:
        - Memory
        - BufferMemory
        - WindowMemory
        - TokenWindowMemory
        - SummaryMemory
        - VectorMemory
        - CompositeMemory
        - memory_from_dict

## Run stores

::: pyagents.stores
    options:
      members:
        - RunStore
        - MemoryStore
        - SQLiteRunStore
        - store_from_uri

## Parsers

::: pyagents.parsers
    options:
      members:
        - Parser
        - AutoParser
        - JsonParser
        - ListParser
        - RegexParser
        - parser_for
        - parse_json_loose
        - schema_to_instructions

## Embeddings

::: pyagents.embeddings
    options:
      members:
        - Embedder
        - HashEmbedder
        - TfidfEmbedder
        - OpenAIEmbedder
        - cosine_similarity

## Core types

::: pyagents.types
    options:
      members:
        - Message
        - Role
        - ToolCall
        - ToolResult
        - Usage
        - LLMResponse
        - StreamChunk
        - Step
        - Handoff
        - RunResult
        - RunStatus
        - AgentEvent
        - EventType

## Testing helpers

::: pyagents.testing
    options:
      members:
        - ScriptedProvider
        - RecordingProvider
        - FakeTransport
        - response
        - text_response
        - tool_response
        - cycle

## Errors

::: pyagents.errors
    options:
      members:
        - PyAgentsError
        - ConfigurationError
        - AgentError
        - ProviderError
        - ProviderAuthError
        - ProviderRateLimitError
        - ProviderTimeoutError
        - ToolError
        - ToolNotFoundError
        - ToolValidationError
        - StructuredOutputError
        - MaxStepsExceeded
        - RunFailed
        - RunNotFound
        - OrchestrationError

## Utilities

::: pyagents.utils
    options:
      members:
        - parse_docstring
        - truncate
        - slugify
        - camel_to_snake
        - estimate_tokens
        - estimate_message_tokens
        - collect_annotations
        - maybe_await
        - is_async_callable
        - gather_limited
        - chunked
        - safe_json_loads
        - json_dumps
