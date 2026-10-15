"""Writing your own provider (and testing it without the network).

Any object with ``complete``/``acomplete`` can drive an agent, but subclassing
:class:`~pyagents.llm.base.BaseProvider` is worth it: you inherit retries,
streaming aggregation and the async variants, and only implement the four
wire-format hooks.

The provider below talks to a fictional "Acme LLM" HTTP API. It is exercised
with :class:`~pyagents.testing.FakeTransport`, so the example runs offline and
still proves the request/response shapes are right.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional

from _bootstrap import section
from pyagents import Agent, BaseProvider, ToolCall, register_provider
from pyagents.testing import FakeTransport
from pyagents.types import LLMResponse, Message, StreamChunk, Usage


class AcmeProvider(BaseProvider):
    """Provider for the (fictional) Acme LLM API."""

    name = "acme"
    api_key_env = "ACME_API_KEY"
    default_model = "acme-small"

    @classmethod
    def default_base_url(cls) -> str:
        return "https://api.acme.example/v1"

    # ---- where and how to talk to the API -------------------------------- #
    def _endpoint(self) -> str:
        return f"{self.base_url}/generate"

    def _auth_headers(self) -> Dict[str, str]:
        return {"X-Acme-Key": self.api_key}

    def _payload(
        self,
        messages: List[Message],
        *,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any],
        stream: bool,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_name": self.model,
            "prompt": [{"author": str(m.role), "body": m.text} for m in messages],
            "stream": stream,
        }
        if tools:
            payload["functions"] = [
                {"name": t["name"], "about": t["description"], "args": t["parameters"]}
                for t in tools
            ]
        if "temperature" in options:
            payload["temp"] = options["temperature"]
        return payload

    # ---- how to read the answer back ------------------------------------- #
    def _parse_response(self, data: Mapping[str, Any]) -> LLMResponse:
        result = data.get("result") or {}
        calls = [
            ToolCall(
                name=call.get("tool", ""),
                arguments=dict(call.get("params") or {}),
                id=call.get("ref", "call"),
            )
            for call in result.get("tool_invocations") or []
        ]
        usage = data.get("billing") or {}
        return LLMResponse(
            message=Message.assistant(str(result.get("body", "")), tool_calls=calls),
            model=str(data.get("model_name") or self.model),
            usage=Usage(
                prompt_tokens=int(usage.get("in", 0)),
                completion_tokens=int(usage.get("out", 0)),
                total_tokens=int(usage.get("in", 0)) + int(usage.get("out", 0)),
            ),
            finish_reason=result.get("why_stopped"),
            raw=data,
        )

    # ---- streaming: Acme sends one JSON object per delta ----------------- #
    def _new_stream_state(self) -> Dict[str, Any]:
        return {"text": "", "tool_calls": {}, "usage": Usage(), "finish_reason": None}

    def _parse_stream_data(
        self, data: Mapping[str, Any], state: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        delta = data.get("delta") or {}
        if text := delta.get("body"):
            state["text"] += text
            yield StreamChunk(kind="text", text=text)
        if delta.get("done"):
            state["finish_reason"] = "stop"
            usage = data.get("billing") or {}
            state["usage"] = Usage(
                int(usage.get("in", 0)),
                int(usage.get("out", 0)),
                int(usage.get("in", 0)) + int(usage.get("out", 0)),
            )

    def _iter_events(self, payload: Mapping[str, Any]):  # type: ignore[no-untyped-def]
        """Acme streams newline-delimited JSON rather than SSE."""
        for line in self.transport.post_sse(
            self._endpoint(),
            headers=self._headers(),
            payload=payload,
            timeout=self.timeout,
            provider=self.name,
        ):
            if line.strip():
                yield None, line.strip()


def main() -> None:
    register_provider("acme", AcmeProvider)

    section("1. A complete() call, verified with FakeTransport")
    transport = FakeTransport(
        [
            {
                "model_name": "acme-small",
                "result": {
                    "body": "Hello from Acme.",
                    "tool_invocations": [
                        {"tool": "calculator", "params": {"expression": "2+2"}, "ref": "r1"}
                    ],
                    "why_stopped": "tool",
                },
                "billing": {"in": 11, "out": 4},
            }
        ]
    )
    provider = AcmeProvider("acme-small", api_key="acme-key", transport=transport)
    response = provider.complete([Message.user("hi")])

    print(f"text: {response.text}")
    print(f"tool calls: {response.tool_calls}")
    print(f"tokens: {response.usage.total_tokens}")
    print(f"request went to: {transport.requests[0]['url']}")
    print(f"auth header: {transport.requests[0]['headers']['X-Acme-Key']}")
    print(f"payload: {transport.last_payload}")

    section("2. Streaming, aggregated by the base class")
    stream_transport = FakeTransport(
        sse=[
            [
                '{"delta": {"body": "Hel"}}',
                '{"delta": {"body": "lo"}}',
                '{"delta": {"done": true}, "billing": {"in": 3, "out": 2}}',
            ]
        ]
    )
    chunks = list(
        AcmeProvider("acme-small", api_key="k", transport=stream_transport).stream(
            [Message.user("hi")]
        )
    )
    print(f"chunks: {[(chunk.kind, chunk.text) for chunk in chunks if chunk.kind == 'text']}")
    print(
        f"assembled: {chunks[-1].response.text!r} ({chunks[-1].response.usage.total_tokens} tokens)"
    )

    section("3. Retries are inherited for free")
    from pyagents import ProviderRateLimitError

    retry_transport = FakeTransport(
        [
            ProviderRateLimitError("slow down", provider="acme", status_code=429, retry_after=0),
            {"result": {"body": "second try worked"}},
        ]
    )
    provider = AcmeProvider("acme-small", api_key="k", transport=retry_transport, max_retries=1)
    print(f"after a 429: {provider.complete([Message.user('hi')]).text!r}")
    print(f"attempts made: {retry_transport.call_count}")

    section("4. Using the custom provider from an agent")
    agent_transport = FakeTransport([{"result": {"body": "The scripted answer."}}])
    agent = Agent(
        "acme-agent",
        provider=AcmeProvider("acme-small", api_key="k", transport=agent_transport),
        # Acme streams newline-delimited JSON; here we only queued a
        # non-streaming reply, so ask the agent for a single completion call.
        stream_model=False,
    )
    print(f"agent output: {agent.run('hello there').output!r}")

    section("5. Registered in the provider registry")
    from pyagents.llm import PROVIDERS

    print(f"'acme' is now available: {PROVIDERS['acme'].__name__}")
    print("…which means it also works from the CLI:  pyagents run 'hi' --provider acme")


if __name__ == "__main__":
    main()
