"""Exception hierarchy for PyAgents.

Every error raised by the framework derives from :class:`PyAgentsError`, so a
single ``except PyAgentsError`` clause is enough to catch anything the library
throws on purpose.

    >>> try:
    ...     raise ToolNotFoundError("nope")
    ... except PyAgentsError as exc:
    ...     print(exc)
    nope
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "AgentError",
    "ConfigurationError",
    "MaxStepsExceeded",
    "OrchestrationError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "PyAgentsError",
    "RunFailed",
    "RunNotFound",
    "StructuredOutputError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
]


def attach_run_result(exc: BaseException, result: Any) -> None:
    """Attach a partial run result to an exception.

    When a run fails, the transcript collected so far is still useful. The
    runtime stores it on the exception as ``exc.run_result`` so callers can
    inspect partial output after catching the error::

        try:
            agent.run("...")
        except PyAgentsError as exc:
            partial = getattr(exc, "run_result", None)
    """
    exc.__dict__["run_result"] = result


class PyAgentsError(Exception):
    """Base class for all PyAgents errors."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({details})"


class ConfigurationError(PyAgentsError):
    """Raised when an agent, tool or provider is configured incorrectly."""


# --------------------------------------------------------------------------- #
# Provider errors
# --------------------------------------------------------------------------- #
class ProviderError(PyAgentsError):
    """Raised when an LLM provider call fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        body: Any = None,
        **context: Any,
    ) -> None:
        super().__init__(message, **context)
        self.provider = provider
        self.status_code = status_code
        self.body = body


class ProviderAuthError(ProviderError):
    """Raised on 401/403 responses - the API key is missing or invalid."""


class ProviderRateLimitError(ProviderError):
    """Raised on 429 responses. Retries are attempted before this surfaces."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call exceeds the configured timeout."""


# --------------------------------------------------------------------------- #
# Tool errors
# --------------------------------------------------------------------------- #
class ToolError(PyAgentsError):
    """Raised when tool execution fails.

    Tool failures are normally converted into error observations that are fed
    back to the model instead of aborting the run - see
    :attr:`pyagents.tools.Tool.output_errors`.
    """


class ToolNotFoundError(ToolError):
    """Raised when a model requests a tool that is not registered."""


class ToolValidationError(ToolError):
    """Raised when tool arguments do not match the tool's JSON schema."""


# --------------------------------------------------------------------------- #
# Agent / orchestration errors
# --------------------------------------------------------------------------- #
class AgentError(PyAgentsError):
    """Raised for agent-level failures."""


class MaxStepsExceeded(AgentError):
    """Raised when an agent needs more reasoning steps than ``max_steps``."""


class RunFailed(AgentError):
    """Raised when a run ends in a failed state.

    Carries the :class:`~pyagents.types.RunResult` (if any) in ``context`` so
    callers can inspect partial output.
    """


class RunNotFound(PyAgentsError):
    """Raised when a run id cannot be resolved by a store."""


class StructuredOutputError(AgentError):
    """Raised when an agent's final answer does not match ``output_model``."""

    def __init__(self, message: str, *, raw: str = "", errors: Any = None) -> None:
        super().__init__(message, raw=raw)
        self.raw = raw
        self.errors = errors


class OrchestrationError(PyAgentsError):
    """Raised when a team/workflow cannot be built or executed."""
