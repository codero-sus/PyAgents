"""Pluggable LLM providers.

The registry in :data:`PROVIDERS` maps short names to adapter classes, which is
what powers ``pyagents.get_provider("openai", "gpt-4o-mini")`` and the CLI's
``--provider`` flag:

    >>> from pyagents.llm import PROVIDERS
    >>> sorted(PROVIDERS)
    ['anthropic', 'claude', 'echo', 'ollama', 'openai', 'openai-compatible']
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from ..errors import ConfigurationError
from .anthropic import AnthropicProvider
from .base import BaseProvider, LLMProvider, normalize_tool_spec, normalize_tool_specs
from .echo import EchoProvider
from .ollama import OllamaProvider
from .openai import OpenAICompatibleProvider, OpenAIProvider
from .transport import (
    HttpTransport,
    HttpxTransport,
    UrllibTransport,
    get_default_transport,
    iter_sse,
    reset_default_transport,
)

__all__ = [
    "PROVIDERS",
    "AnthropicProvider",
    "BaseProvider",
    "EchoProvider",
    "HttpTransport",
    "HttpxTransport",
    "LLMProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "UrllibTransport",
    "default_provider",
    "get_default_transport",
    "get_provider",
    "iter_sse",
    "normalize_tool_spec",
    "normalize_tool_specs",
    "register_provider",
    "reset_default_transport",
]

#: Registry of built-in providers, keyed by the name users pass around.
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "openai-compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "ollama": OllamaProvider,
    "echo": EchoProvider,
}


#: Environment variable that pins the default model.
MODEL_ENV = "PYAGENTS_MODEL"
#: Environment variable that pins the default provider.
PROVIDER_ENV = "PYAGENTS_PROVIDER"


def default_provider(model: Optional[str] = None, **kwargs: Any) -> BaseProvider:
    """Pick a provider without being told which one to use.

    The decision order is:

    1. ``PYAGENTS_PROVIDER`` (with ``PYAGENTS_MODEL`` or ``model``)
    2. ``OPENAI_API_KEY`` present -> OpenAI
    3. ``ANTHROPIC_API_KEY`` present -> Anthropic
    4. otherwise -> Ollama on localhost (works without credentials)

    Args:
        model: Explicit model override.
        kwargs: Forwarded to the provider constructor.

    Raises:
        ConfigurationError: if ``PYAGENTS_PROVIDER`` names an unknown provider.
    """
    import os

    chosen_model = model or os.environ.get(MODEL_ENV) or None
    pinned = os.environ.get(PROVIDER_ENV)
    if pinned:
        return get_provider(pinned, chosen_model, **kwargs)
    if os.environ.get("OPENAI_API_KEY"):
        return get_provider("openai", chosen_model or "gpt-4o-mini", **kwargs)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return get_provider("anthropic", chosen_model, **kwargs)
    return get_provider("ollama", chosen_model, **kwargs)


def register_provider(name: str, provider: Type[BaseProvider]) -> None:
    """Add or replace a provider class in the registry.

    >>> from pyagents.llm import BaseProvider, register_provider, PROVIDERS
    >>> class MyProvider(BaseProvider):  # doctest: +SKIP
    ...     ...
    >>> register_provider("mine", MyProvider)  # doctest: +SKIP
    """
    if not isinstance(provider, type) or not issubclass(provider, BaseProvider):
        raise ConfigurationError("register_provider expects a BaseProvider subclass", provider=name)
    PROVIDERS[name.lower()] = provider


def get_provider(name: str, model: Optional[str] = None, **kwargs: Any) -> BaseProvider:
    """Instantiate a registered provider by name.

    Args:
        name: Registry key, e.g. ``"openai"``, ``"anthropic"``, ``"ollama"``.
        model: Model identifier. Falls back to the adapter's default model.
        kwargs: Forwarded to the provider constructor.

    Raises:
        ConfigurationError: if the name is unknown.

    >>> get_provider("ollama", "llama3.2").model
    'llama3.2'
    """
    key = (name or "").lower()
    if key not in PROVIDERS:
        known = ", ".join(sorted(set(PROVIDERS)))
        raise ConfigurationError(
            f"Unknown provider '{name}'. Known providers: {known}", provider=name
        )
    provider_cls = PROVIDERS[key]
    try:
        return provider_cls(model, **kwargs) if model is not None else provider_cls(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(
            f"Provider '{name}' needs more arguments: {exc}", provider=name
        ) from exc
