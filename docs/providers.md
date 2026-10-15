# Providers

A provider is anything that can turn a list of `Message`s into an
`LLMResponse`. PyAgents ships four and lets you register your own.

| provider | registry name(s) | needs | notes |
|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | current Chat Completions API, streaming, tools |
| OpenAI-compatible | `openai-compatible` | usually nothing | vLLM, LM Studio, OpenRouter, Together, ... |
| Anthropic | `anthropic`, `claude` | `ANTHROPIC_API_KEY` | Messages API, tool use, streaming |
| Ollama | `ollama` | a local server | no key needed; uses `OLLAMA_HOST` or `http://localhost:11434` |
| Echo | `echo` | nothing | offline deterministic provider for tests and demos |

## Choosing one

```python
from pyagents import Agent, get_provider, default_provider

Agent("a", provider="openai", model="gpt-4o-mini")
Agent("b", provider="anthropic", model="claude-3-5-haiku-latest")
Agent("c", provider="ollama", model="llama3.2")
Agent("d", provider="echo")

provider = get_provider("openai", model="gpt-4o", temperature=0.7)
provider = default_provider()          # first configured provider
```

`default_provider()` resolves in this order:

1. `PYAGENTS_PROVIDER` (and `PYAGENTS_MODEL`) if set
2. `OPENAI_API_KEY` → OpenAI
3. `ANTHROPIC_API_KEY` → Anthropic
4. otherwise → Ollama

Environment variables:

| variable | used by |
|---|---|
| `PYAGENTS_PROVIDER`, `PYAGENTS_MODEL` | `default_provider()` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI (and the compatible provider) |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` | Anthropic |
| `OLLAMA_HOST` | Ollama (`host:port` or a full URL) |

## OpenAI

```python
from pyagents import OpenAIProvider

provider = OpenAIProvider(
    "gpt-4o-mini",
    api_key="sk-...",          # defaults to OPENAI_API_KEY
    base_url="https://api.openai.com/v1",
    timeout=60.0,
    max_retries=2,
)
```

Supports text, tool calls, streaming (including incremental tool-call
arguments) and usage accounting (`stream_options.include_usage` is set
automatically when streaming).

## OpenAI-compatible endpoints

```python
from pyagents import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",       # sent as a bearer token only if non-empty
)
```

This is the same wire format with `requires_api_key = False`, which keeps the
CLI and `default_provider()` from demanding a key.

## Anthropic

```python
from pyagents import AnthropicProvider

provider = AnthropicProvider(
    "claude-3-5-haiku-latest",
    api_key="sk-ant-...",       # defaults to ANTHROPIC_API_KEY
    max_tokens=4096,            # required by the API
)
```

Details handled for you:

* system messages are hoisted into Anthropic's top-level `system` parameter
* tool results are merged into user turns as `tool_result` blocks
* images passed as `{"type": "image_url", ...}` become base64 image blocks
* the `anthropic-version: 2023-06-01` header is sent

## Ollama

```python
from pyagents import OllamaProvider

provider = OllamaProvider("llama3.2", base_url="http://localhost:11434")
provider.list_models()        # GET /api/tags
provider.has_model("llama3.2")
```

`OLLAMA_HOST` accepts both `localhost:11434` and `http://localhost:11434` (the
scheme is added when missing), and the `base_url=` argument overrides it.
Streaming uses Ollama's newline-delimited JSON format rather than SSE, which
the provider handles by overriding the base class's event iterator.

## Echo, for offline development

```python
from pyagents.testing import EchoProvider

provider = EchoProvider()
provider.complete([Message.user("ping")]).text
# 'You said: "ping". (This is the offline echo provider.)'
```

`echo` is registered like any other provider, so
`pyagents run "hello" --provider echo` works with no setup at all.

## What the base class gives you

All of the built-ins subclass `BaseProvider`; custom providers get these for
free:

* retries with exponential backoff and jitter for 408/409/425/429/5xx (a
  `Retry-After` header is honoured)
* `stream()`/`astream()` implemented on top of `complete()` for providers
  without a streaming API
* the async variants (`acomplete`, `astream`) as thin wrappers
* tool-spec normalisation, control-option stripping and typed errors
  (`ProviderAuthError`, `ProviderRateLimitError`, `ProviderTimeoutError`,
  `ProviderError`)

To integrate your own API, subclass `BaseProvider` and implement the wire
format hooks - see
[the custom provider example](https://github.com/codero-sus/PyAgents/blob/main/examples/07_custom_provider.py):

```python
from pyagents import BaseProvider, register_provider

class AcmeProvider(BaseProvider):
    name = "acme"
    api_key_env = "ACME_API_KEY"
    default_model = "acme-small"

    def _endpoint(self) -> str: ...
    def _auth_headers(self) -> dict: ...
    def _payload(self, messages, *, tools, tool_choice, stream, options) -> dict: ...
    def _parse_response(self, data) -> LLMResponse: ...

register_provider("acme", AcmeProvider)
```

## Transports

Providers speak HTTP through a pluggable transport:

* `UrllibTransport` - the default; standard library only
* `HttpxTransport` - used automatically when `httpx` is installed
  (`pip install "pyagents[http]"`), HTTP/2-capable and connection-pooled

```python
from pyagents.llm import OpenAIProvider
from pyagents.llm.transport import HttpxTransport

provider = OpenAIProvider("gpt-4o-mini", transport=HttpxTransport())
```

Passing `transport=FakeTransport(...)` is how the test suite exercises the real
request/response paths without touching the network - see
[Testing agents](testing.md).

## Errors

```python
from pyagents.errors import (
    ProviderAuthError,        # 401/403 - bad or missing key
    ProviderRateLimitError,   # 429 - has .retry_after
    ProviderTimeoutError,     # 408/504 and socket timeouts
    ProviderError,            # everything else, carries .status_code/.body
)
```

All of them derive from `PyAgentsError`, so one `except PyAgentsError` catches
everything the framework raises. Retries happen for the transient cases; after
`max_retries` the error propagates to the caller, and the partially completed
run is available as `exc.run_result`.
