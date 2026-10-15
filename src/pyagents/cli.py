"""Command-line interface: ``pyagents <command>``.

Installed as a console script by ``pip install pyagents``. Run ``pyagents
--help`` for the full surface, or read on:

* ``pyagents run "Summarise this file" --tools read_file`` - one-shot run
* ``pyagents chat`` - interactive session with memory
* ``pyagents tools list`` - inspect built-in tools and their schemas
* ``pyagents providers`` - show provider registry and credential status
* ``pyagents runs list|show`` - inspect a SQLite run store
* ``pyagents info`` - environment diagnostics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .agent import Agent
from .errors import PyAgentsError
from .llm import PROVIDERS, BaseProvider, default_provider, get_provider
from .memory import BufferMemory
from .stores import SQLiteRunStore, store_from_uri
from .tools.builtin import default_tools
from .types import EventType, RunResult
from .utils import json_dumps

__all__ = ["build_parser", "main"]

PROG = "pyagents"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_tools(names: Optional[str]) -> List[Any]:
    """Turn a ``--tools`` string into tool objects.

    Accepts comma/space separated built-in names; ``all`` includes every
    built-in, including the powerful ones (``python_repl``, ``write_file``).
    """
    if not names:
        return []
    requested = [item.strip() for item in names.replace(" ", ",").split(",") if item.strip()]
    if not requested:
        return []
    return default_tools(requested)


def build_provider(args: argparse.Namespace) -> BaseProvider:
    """Create the provider described by parsed CLI arguments."""
    if args.provider:
        return get_provider(args.provider, args.model, base_url=args.base_url, timeout=args.timeout)
    provider = default_provider(model=args.model, timeout=args.timeout)
    if args.base_url and hasattr(provider, "base_url"):
        provider.base_url = args.base_url.rstrip("/")
    return provider


def build_agent(args: argparse.Namespace, *, memory: Any = None) -> Agent:
    """Create an agent from parsed CLI arguments."""
    provider = build_provider(args)
    store = store_from_uri(getattr(args, "store", None))
    return Agent(
        name="cli",
        instructions=args.instructions,
        provider=provider,
        tools=resolve_tools(args.tools),
        memory=memory,
        max_steps=args.max_steps,
        store=store,
        verbose=False,
    )


def _print_event(event: Any, *, stream: bool, quiet: bool) -> None:
    """Render a single event to the terminal."""
    if quiet:
        return
    if event.type is EventType.TEXT and stream:
        print(event.text, end="", flush=True)
    elif event.type is EventType.TOOL_CALL:
        print(f"\n  → {event.tool_call}", file=sys.stderr)
    elif event.type is EventType.TOOL_RESULT:
        preview = event.tool_result.text if event.tool_result else ""
        preview = preview if len(preview) < 300 else preview[:300] + "…"
        marker = "✗" if event.tool_result and event.tool_result.is_error else "✓"
        print(f"  {marker} {preview}", file=sys.stderr)
    elif event.type is EventType.HANDOFF:
        print(f"\n  ⇄ {event.text}", file=sys.stderr)
    elif event.type is EventType.AGENT_START and event.agent != "cli":
        print(f"\n  ▶ {event.agent}: {event.data.get('task', '')[:80]}", file=sys.stderr)
    elif event.type is EventType.ERROR:
        print(f"\nerror: {event.data.get('error')}", file=sys.stderr)


def _run_summary(result: RunResult) -> str:
    """Human-readable one-block summary of a run."""
    return (
        f"status: {result.status}\n"
        f"run_id: {result.run_id}\n"
        f"steps: {len(result.steps)}\n"
        f"tool calls: {len(result.tool_calls)}\n"
        f"tokens: {result.usage.total_tokens} "
        f"(prompt {result.usage.prompt_tokens}, completion {result.usage.completion_tokens})\n"
        f"duration: {result.duration_ms:.0f} ms"
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    """Handle ``pyagents run``."""
    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read()
    if not prompt:
        print("error: provide a prompt (or '-' to read from stdin)", file=sys.stderr)
        return 2

    memory = BufferMemory(max_messages=args.history) if args.history else None
    agent = build_agent(args, memory=memory)
    stream = not args.no_stream

    try:
        if args.json:
            result = agent.run(prompt)
            print(json_dumps(result.to_dict(), indent=2))
            return 0 if result.ok else 1

        events = agent.stream(prompt)
        run_result: Optional[RunResult] = None
        for event in events:
            _print_event(event, stream=stream, quiet=args.quiet)
            if event.type is EventType.RUN_END and event.raw is not None:
                run_result = event.raw
        if run_result is None:  # pragma: no cover - defensive
            print("error: run produced no result", file=sys.stderr)
            return 1
        if not args.quiet:
            if stream:
                print()
            if args.verbose:
                print(_run_summary(run_result), file=sys.stderr)
            if not stream:
                print(run_result.output)
        return 0 if run_result.ok else 1
    except PyAgentsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130


def cmd_chat(args: argparse.Namespace) -> int:
    """Handle ``pyagents chat``: an interactive REPL with memory."""
    memory = BufferMemory(max_messages=args.history)
    try:
        agent = build_agent(args, memory=memory)
    except PyAgentsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{PROG} chat — {agent.provider.name}/{agent.provider.model}.")
    print("Commands: /help, /tools, /reset, /instruction <text>, /exit\n")
    interactive = sys.stdin.isatty()
    while True:
        try:
            line = input("you> " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        prompt = line.strip()
        if not prompt:
            if not interactive:
                break
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/help":
            print("Commands: /help, /tools, /reset, /instruction <text>, /exit")
            continue
        if prompt == "/tools":
            for tool in agent.all_tools:
                print(f"- {tool.name}: {tool.description}")
            continue
        if prompt == "/reset":
            memory.clear()
            print("(memory cleared)")
            continue
        if prompt.startswith("/instruction "):
            agent.instructions = prompt[len("/instruction ") :]
            print("(updated system prompt)")
            continue

        print("agent> ", end="", flush=True)
        try:
            for event in agent.stream(prompt):
                if event.type is EventType.TEXT:
                    print(event.text, end="", flush=True)
                elif event.type is EventType.TOOL_CALL:
                    print(f"\n  → {event.tool_call}", file=sys.stderr)
                elif event.type is EventType.TOOL_RESULT:
                    tool_result = event.tool_result
                    preview = tool_result.text if tool_result else ""
                    failed = bool(tool_result and tool_result.is_error)
                    print(f"  {'✗' if failed else '✓'} {preview[:200]}", file=sys.stderr)
            print()
        except PyAgentsError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
        if not interactive:
            break
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """Handle ``pyagents tools``."""
    tools = default_tools(["all"] if args.all else None)
    if args.json:
        print(json_dumps([tool.to_spec() for tool in tools], indent=2))
        return 0
    for tool in tools:
        print(f"{tool.name}: {tool.description}")
        if args.verbose:
            required = ", ".join(tool.required_arguments) or "none"
            print(f"    required: {required}")
            print("    schema: " + json_dumps(tool.parameters))
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    """Handle ``pyagents providers``."""

    def credential_state(cls: Any) -> str:
        env_var = getattr(cls, "api_key_env", None)
        if not getattr(cls, "requires_api_key", True):
            return "not required"
        return "set" if env_var and os.environ.get(env_var) else "missing"

    if args.json:
        print(
            json_dumps(
                [
                    {
                        "name": name,
                        "class": cls.__name__,
                        "credential": credential_state(cls),
                        "api_key_env": getattr(cls, "api_key_env", None),
                    }
                    for name, cls in sorted(PROVIDERS.items())
                ],
                indent=2,
            )
        )
        return 0
    print("Providers (name → adapter, credential)")
    for name, cls in sorted(PROVIDERS.items()):
        print(f"  {name:<18} {cls.__name__:<26} {credential_state(cls)}")
    try:
        provider = default_provider()
        print(f"\nDefault provider: {provider.name} ({provider.model})")
    except PyAgentsError as exc:
        print(f"\nDefault provider unavailable: {exc}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """Handle ``pyagents runs`` (list/show/delete) against a SQLite store."""
    path = Path(args.store or "pyagents_runs.sqlite3")
    if not path.exists() and args.action != "list":
        print(f"error: no run store at {path}", file=sys.stderr)
        return 1
    store = SQLiteRunStore(path)
    try:
        if args.action == "list":
            rows = store.list_runs(limit=args.limit, status=args.status)
            if args.json:
                print(json_dumps(rows, indent=2))
                return 0
            if not rows:
                print(f"(no runs in {path})")
                return 0
            for row in rows:
                print(
                    f"{row['run_id']}  {row['status']:<10} steps={row['steps']:<3} {row['output'][:60]!r}"
                )
            return 0
        if args.action == "show":
            result = store.load(args.run_id)
            print(json_dumps(result.to_dict(), indent=2))
            return 0
        if args.action == "delete":
            store.delete(args.run_id)
            print(f"deleted {args.run_id}")
            return 0
    except PyAgentsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Handle ``pyagents info``: environment diagnostics."""
    import platform

    info: Dict[str, Any] = {
        "pyagents": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "httpx": _optional_version("httpx"),
        "pydantic": _optional_version("pydantic"),
        "cwd": str(Path.cwd()),
        "env": {
            key: ("set" if os.environ.get(key) else "unset")
            for key in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "OLLAMA_HOST",
                "PYAGENTS_PROVIDER",
                "PYAGENTS_MODEL",
            )
        },
    }
    try:
        provider = default_provider()
        info["default_provider"] = f"{provider.name}/{provider.model}"
    except PyAgentsError as exc:
        info["default_provider"] = f"unavailable: {exc}"
    if args.json:
        print(json_dumps(info, indent=2))
    else:
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    return 0


def _optional_version(module: str) -> str:
    """Return an installed module's version, or ``"not installed"``."""
    try:
        import importlib.metadata as metadata

        return metadata.version(module)
    except Exception:
        return "not installed"


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="provider name: " + ", ".join(sorted(set(PROVIDERS))))
    parser.add_argument("--model", help="model identifier (defaults to the provider's)")
    parser.add_argument("--base-url", dest="base_url", help="override the API base URL")
    parser.add_argument("--instructions", help="system prompt for the agent")
    parser.add_argument(
        "--tools",
        help="comma-separated built-in tools (calculator, current_time, read_file, http_get, 'all', ...)",
    )
    parser.add_argument(
        "--max-steps", dest="max_steps", type=int, default=10, help="reasoning step budget"
    )
    parser.add_argument("--store", help="run store URI: a SQLite path or 'memory:'")
    parser.add_argument("--timeout", type=float, default=60.0, help="request timeout in seconds")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``pyagents`` command."""
    parser = argparse.ArgumentParser(prog=PROG, description="Multi-agent LLM framework CLI")
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an agent once")
    run_parser.add_argument("prompt", help="the task; '-' reads from stdin")
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--history", type=int, default=0, help="keep N messages of memory between runs"
    )
    run_parser.add_argument("--json", action="store_true", help="print the full RunResult as JSON")
    run_parser.add_argument(
        "--no-stream", dest="no_stream", action="store_true", help="disable token streaming"
    )
    run_parser.add_argument(
        "--quiet", action="store_true", help="suppress the final answer printing"
    )
    run_parser.add_argument("--verbose", action="store_true", help="print a run summary to stderr")
    run_parser.set_defaults(func=cmd_run)

    chat_parser = subparsers.add_parser("chat", help="interactive chat session")
    _add_common_arguments(chat_parser)
    chat_parser.add_argument("--history", type=int, default=20, help="messages of memory to keep")
    chat_parser.set_defaults(func=cmd_chat)

    tools_parser = subparsers.add_parser("tools", help="inspect built-in tools")
    tools_parser.add_argument("action", nargs="?", choices=["list"], default="list")
    tools_parser.add_argument(
        "--all", action="store_true", help="include powerful tools (python_repl, write_file)"
    )
    tools_parser.add_argument("--verbose", action="store_true", help="show schemas")
    tools_parser.add_argument("--json", action="store_true", help="emit JSON")
    tools_parser.set_defaults(func=cmd_tools)

    providers_parser = subparsers.add_parser("providers", help="list providers and credentials")
    providers_parser.add_argument("--json", action="store_true", help="emit JSON")
    providers_parser.set_defaults(func=cmd_providers)

    runs_parser = subparsers.add_parser("runs", help="inspect a run store")
    runs_parser.add_argument(
        "action", choices=["list", "show", "delete"], nargs="?", default="list"
    )
    runs_parser.add_argument("run_id", nargs="?", help="run id for show/delete")
    runs_parser.add_argument("--store", help="path to the SQLite run store")
    runs_parser.add_argument("--limit", type=int, default=20)
    runs_parser.add_argument("--status", help="filter by status")
    runs_parser.add_argument("--json", action="store_true")
    runs_parser.set_defaults(func=cmd_runs)

    info_parser = subparsers.add_parser("info", help="environment diagnostics")
    info_parser.add_argument("--json", action="store_true")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``pyagents`` console script.

    Returns:
        Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "instructions") and not getattr(args, "instructions", None):
        args.instructions = None
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
    except PyAgentsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
