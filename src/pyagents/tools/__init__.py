"""Tools: the functions agents can call.

>>> from pyagents.tools import tool
>>> @tool(tags=["demo"])
... def shout(text: str) -> str:
...     '''Uppercase some text.'''
...     return text.upper()
>>> shout.to_openai_spec()["function"]["name"]
'shout'
"""

from __future__ import annotations

from .base import (
    FunctionTool,
    Tool,
    Toolkit,
    coerce_arguments,
    ensure_tool,
    json_schema_from_function,
    run_tool_call,
    run_tool_call_sync,
    tool,
    validate_arguments,
)
from .builtin import (
    calculator,
    current_time,
    default_toolkit,
    default_tools,
    http_get,
    http_request,
    list_directory,
    python_repl,
    read_file,
    write_file,
)

__all__ = [
    "Tool",
    "FunctionTool",
    "Toolkit",
    "tool",
    "ensure_tool",
    "json_schema_from_function",
    "validate_arguments",
    "coerce_arguments",
    "run_tool_call",
    "run_tool_call_sync",
    # built-ins
    "calculator",
    "current_time",
    "read_file",
    "write_file",
    "list_directory",
    "http_get",
    "http_request",
    "python_repl",
    "default_tools",
    "default_toolkit",
]
