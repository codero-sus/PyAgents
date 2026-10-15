"""Tests for the LLM providers, transports and registry."""

from __future__ import annotations

import pytest

from pyagents.errors import (
    ConfigurationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from pyagents.llm import (
    PROVIDERS,
    AnthropicProvider,
    BaseProvider,
    EchoProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    default_provider,
    get_provider,
    register_provider,
)
from pyagents.llm.transport import UrllibTransport, iter_sse
from pyagents.testing import FakeTransport, response
from pyagents.types import Message, ToolCall

OPENAI_OK = {
    "model": "gpt-test",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def sse_lines(*payloads: str, done: bool = False) -> list:
    lines = []
    for payload in payloads:
        lines += [f"data: {payload}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return lines


class TestRegistry:
    def test_builtin_providers(self):
        assert {"openai", "anthropic", "ollama", "echo", "openai-compatible"} <= set(PROVIDERS)

    def test_get_provider_by_name(self):
        provider = get_provider("openai", "gpt-4o-mini", api_key="sk-x")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_unknown_provider(self):
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            get_provider("nope")

    def test_register_and_use_custom_provider(self):
        class Dummy(BaseProvider):
            name = "dummy"
            requires_api_key = False

            @classmethod
            def default_base_url(cls):
                return "http://dummy"

            def _endpoint(self):
                return f"{self.base_url}/x"

            def _auth_headers(self):
                return {}

            def _payload(self, messages, **kwargs):
                return {"model": self.model, "messages": []}

            def _parse_response(self, data):
                return response("ok")

        register_provider("dummy", Dummy)
        assert isinstance(get_provider("dummy", "m"), Dummy)
        with pytest.raises(ConfigurationError):
            register_provider("bad", dict)  # type: ignore[arg-type]

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="API key"):
            OpenAIProvider("gpt-4o-mini")

    def test_default_provider_precedence(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-a")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-b")
        monkeypatch.setenv("PYAGENTS_PROVIDER", "anthropic")
        assert default_provider().name == "anthropic"
        monkeypatch.delenv("PYAGENTS_PROVIDER")
        assert default_provider().name == "openai"
        monkeypatch.delenv("OPENAI_API_KEY")
        assert default_provider().name == "anthropic"

    def test_default_provider_falls_back_to_ollama(self, monkeypatch):
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PYAGENTS_PROVIDER"):
            monkeypatch.delenv(key, raising=False)
        assert default_provider().name == "ollama"

    def test_default_provider_honours_model_env(self, monkeypatch):
        monkeypatch.setenv("PYAGENTS_PROVIDER", "ollama")
        monkeypatch.setenv("PYAGENTS_MODEL", "mistral")
        assert default_provider().model == "mistral"

    def test_provider_needs_more_arguments(self):
        with pytest.raises(ConfigurationError, match="needs more arguments"):
            get_provider("openai-compatible", "m")  # base_url missing


class TestOpenAIProvider:
    def test_request_shape(self):
        transport = FakeTransport([OPENAI_OK])
        provider = OpenAIProvider(
            "gpt-test", api_key="sk-x", transport=transport, temperature=0.3, max_tokens=100
        )
        result = provider.complete(
            [Message.system("sys"), Message.user("hi")],
            tools=[{"name": "calc", "description": "d", "parameters": {"type": "object"}}],
            tool_choice="auto",
        )
        payload = transport.last_payload
        assert payload["model"] == "gpt-test"
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 100
        assert payload["tools"][0]["function"]["name"] == "calc"
        assert payload["tool_choice"] == "auto"
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer sk-x"
        assert result.text == "hello"
        assert result.usage.total_tokens == 7

    def test_tool_choice_variants(self):
        transport = FakeTransport([OPENAI_OK, OPENAI_OK])
        provider = OpenAIProvider("m", api_key="k", transport=transport)
        tool = {"name": "calc", "description": "d", "parameters": {"type": "object"}}
        provider.complete(
            [Message.user("x")],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "calc"}},
        )
        assert transport.last_payload["tool_choice"]["function"]["name"] == "calc"
        provider.complete([Message.user("x")], tools=[tool], tool_choice=ToolCall("calc"))
        assert transport.last_payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "calc"},
        }

    def test_missing_tool_choice_is_omitted(self):
        transport = FakeTransport([OPENAI_OK])
        OpenAIProvider("m", api_key="k", transport=transport).complete(
            [Message.user("x")], tools=[{"name": "t", "description": "", "parameters": {}}]
        )
        assert "tool_choice" not in transport.last_payload

    def test_message_conversion(self):
        transport = FakeTransport([OPENAI_OK])
        provider = OpenAIProvider("m", api_key="k", transport=transport)
        provider.complete(
            [
                Message.user("hi"),
                Message.assistant("thinking", tool_calls=[ToolCall("calc", {"a": 1}, id="call_1")]),
                Message.tool_result("42", tool_call_id="call_1", name="calc"),
            ]
        )
        wire = transport.last_payload["messages"]
        assert wire[0] == {"role": "user", "content": "hi"}
        assert wire[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
        assert wire[2] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}

    def test_parse_tool_calls_and_cache_details(self):
        transport = FakeTransport(
            [
                {
                    "model": "gpt-test",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_9",
                                        "type": "function",
                                        "function": {"name": "calc", "arguments": '{"a": 2}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                }
            ]
        )
        result = OpenAIProvider("m", api_key="k", transport=transport).complete([Message.user("x")])
        assert result.tool_calls[0].id == "call_9"
        assert result.tool_calls[0].arguments == {"a": 2}
        assert result.usage.cached_tokens == 4
        assert result.usage.total_tokens == 11
        assert result.finish_reason == "tool_calls"

    def test_content_list_is_flattened(self):
        transport = FakeTransport(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "a"},
                                    {"type": "text", "text": "b"},
                                ]
                            }
                        }
                    ]
                }
            ]
        )
        assert (
            OpenAIProvider("m", api_key="k", transport=transport).complete([Message.user("x")]).text
            == "a\nb"
        )

    def test_streaming(self):
        payloads = [
            '{"choices":[{"index":0,"delta":{"content":"Hel"}}]}',
            '{"choices":[{"index":0,"delta":{"content":"lo"}}]}',
            '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"calc","arguments":"{\\"a\\""}}]}}]}',
            '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": 1}"}}]}}]}',
            '{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}',
        ]
        transport = FakeTransport(sse=[sse_lines(*payloads, done=True)])
        chunks = list(
            OpenAIProvider("m", api_key="k", transport=transport).stream([Message.user("hi")])
        )
        assert [c.kind for c in chunks if c.kind != "response"] == [
            "text",
            "text",
            "tool_call",
            "tool_call",
        ]
        final = chunks[-1].response
        assert final.text == "Hello"
        assert final.tool_calls[0].arguments == {"a": 1}
        assert final.usage.total_tokens == 3
        assert transport.last_payload["stream"] is True
        assert transport.last_payload["stream_options"] == {"include_usage": True}

    def test_async_paths(self):
        import asyncio

        transport = FakeTransport([OPENAI_OK])
        provider = OpenAIProvider("m", api_key="k", transport=transport)
        assert asyncio.run(provider.acomplete([Message.user("x")])).text == "hello"


class TestErrorHandling:
    def test_auth_error_is_not_retried(self):
        transport = FakeTransport(
            [ProviderAuthError("bad key", provider="openai", status_code=401)]
        )
        provider = OpenAIProvider(
            "m", api_key="k", transport=transport, max_retries=3, retry_base_delay=0
        )
        with pytest.raises(ProviderAuthError):
            provider.complete([Message.user("x")])
        assert transport.call_count == 1

    def test_rate_limit_is_retried(self):
        transport = FakeTransport(
            [ProviderRateLimitError("slow down", provider="openai", status_code=429), OPENAI_OK]
        )
        provider = OpenAIProvider(
            "m", api_key="k", transport=transport, max_retries=1, retry_base_delay=0
        )
        assert provider.complete([Message.user("x")]).text == "hello"
        assert transport.call_count == 2

    def test_retry_gives_up(self):
        transport = FakeTransport(
            [
                ProviderError("boom", provider="openai", status_code=500),
                ProviderError("boom", provider="openai", status_code=500),
            ]
        )
        provider = OpenAIProvider(
            "m", api_key="k", transport=transport, max_retries=1, retry_base_delay=0
        )
        with pytest.raises(ProviderError):
            provider.complete([Message.user("x")])
        assert transport.call_count == 2

    def test_timeout_is_retried_then_raised(self):
        transport = FakeTransport([ProviderTimeoutError("timeout", provider="openai")])
        provider = OpenAIProvider("m", api_key="k", transport=transport, max_retries=0)
        with pytest.raises(ProviderTimeoutError):
            provider.complete([Message.user("x")])

    def test_stream_does_not_retry_after_emitting(self):
        payloads = ['{"choices":[{"index":0,"delta":{"content":"partial"}}]}']
        stream = [
            *sse_lines(*payloads),
            ProviderError("connection reset", provider="openai", status_code=503),
        ]
        transport = FakeTransport(sse=[stream])
        provider = OpenAIProvider(
            "m", api_key="k", transport=transport, max_retries=3, retry_base_delay=0
        )
        chunks = []
        with pytest.raises(ProviderError):
            for chunk in provider.stream([Message.user("x")]):
                chunks.append(chunk)
        assert chunks  # the partial text was delivered before the failure
        assert transport.call_count == 1  # no retry: output was already emitted

    def test_malformed_json_body(self, monkeypatch):
        import urllib.request

        class FakeResponse:
            status = 200
            headers: dict = {}

            def read(self):
                return b"not json at all"

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def close(self):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
        provider = OpenAIProvider("m", api_key="k", max_retries=0, transport=UrllibTransport())
        with pytest.raises(ProviderError, match="invalid JSON"):
            provider.complete([Message.user("x")])


class TestAnthropicProvider:
    def test_request_shape(self):
        transport = FakeTransport(
            [
                {
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 1},
                }
            ]
        )
        provider = AnthropicProvider("claude-test", api_key="sk-ant", transport=transport)
        result = provider.complete([Message.system("sys"), Message.user("hi")])
        payload = transport.last_payload
        assert payload["max_tokens"] == 4096
        assert payload["system"] == [{"type": "text", "text": "sys"}]
        assert all(message["role"] != "system" for message in payload["messages"])
        assert transport.requests[0]["headers"]["x-api-key"] == "sk-ant"
        assert transport.requests[0]["headers"]["anthropic-version"] == "2023-06-01"
        assert result.usage == result.usage.__class__(3, 2, 5, 1)

    def test_tool_results_merge_into_one_user_turn(self):
        transport = FakeTransport([{"content": [{"type": "text", "text": "ok"}], "usage": {}}])
        provider = AnthropicProvider("m", api_key="k", transport=transport)
        provider.complete(
            [
                Message.user("hi"),
                Message.assistant(
                    "", tool_calls=[ToolCall("a", {}, id="t1"), ToolCall("b", {}, id="t2")]
                ),
                Message.tool_result("one", tool_call_id="t1", name="a"),
                Message.tool_result("two", tool_call_id="t2", name="b"),
            ]
        )
        wire = transport.last_payload["messages"]
        assert [message["role"] for message in wire] == ["user", "assistant", "user"]
        blocks = wire[2]["content"]
        assert [block["type"] for block in blocks] == ["tool_result", "tool_result"]
        assert blocks[0]["tool_use_id"] == "t1"

    def test_error_tool_results_are_flagged(self):
        transport = FakeTransport([{"content": [], "usage": {}}])
        provider = AnthropicProvider("m", api_key="k", transport=transport)
        provider.complete([Message.tool_result("bad", tool_call_id="t1", is_error=True)])
        assert transport.last_payload["messages"][0]["content"][0]["is_error"] is True

    def test_tools_use_input_schema(self):
        transport = FakeTransport([{"content": [], "usage": {}}])
        AnthropicProvider("m", api_key="k", transport=transport).complete(
            [Message.user("x")],
            tools=[{"name": "calc", "description": "d", "parameters": {"type": "object"}}],
        )
        assert transport.last_payload["tools"][0]["input_schema"] == {"type": "object"}

    def test_tool_choice_mapping(self):
        transport = FakeTransport([{"content": [], "usage": {}}, {"content": [], "usage": {}}])
        provider = AnthropicProvider("m", api_key="k", transport=transport)
        tool = {"name": "calc", "description": "d", "parameters": {"type": "object"}}
        provider.complete([Message.user("x")], tools=[tool], tool_choice="calc")
        assert transport.last_payload["tool_choice"] == {"type": "tool", "name": "calc"}
        provider.complete([Message.user("x")], tools=[tool], tool_choice={"type": "any"})
        assert transport.last_payload["tool_choice"] == {"type": "any"}

    def test_temperature_is_clamped(self):
        transport = FakeTransport([{"content": [], "usage": {}}])
        AnthropicProvider("m", api_key="k", transport=transport).complete(
            [Message.user("x")], temperature=1.7
        )
        assert transport.last_payload["temperature"] == 1.0

    def test_image_blocks(self):
        transport = FakeTransport([{"content": [], "usage": {}}])
        provider = AnthropicProvider("m", api_key="k", transport=transport)
        provider.complete(
            [
                Message.user(
                    "look", images=["https://example.com/a.png", "data:image/png;base64,QQ=="]
                )
            ]
        )
        blocks = transport.last_payload["messages"][0]["content"]
        assert blocks[1]["source"]["type"] == "url"
        assert blocks[2]["source"]["type"] == "base64"

    def test_streaming_assembles_tool_use(self):
        payloads = [
            '{"type":"message_start","message":{"usage":{"input_tokens":4}}}',
            '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            '{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"calc"}}',
            '{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\": 1}"}}',
            '{"type":"content_block_stop","index":1}',
            '{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}',
        ]
        transport = FakeTransport(sse=[sse_lines(*payloads)])
        chunks = list(
            AnthropicProvider("m", api_key="k", transport=transport).stream([Message.user("x")])
        )
        final = chunks[-1].response
        assert final.text == "Hi"
        assert final.tool_calls[0].arguments == {"a": 1}
        assert final.usage.prompt_tokens == 4
        assert final.usage.completion_tokens == 6
        assert final.finish_reason == "tool_use"

    def test_stream_error_frame(self):
        payloads = ['{"type":"error","error":{"message":"overloaded"}}']
        transport = FakeTransport(sse=[sse_lines(*payloads)])
        with pytest.raises(ProviderError, match="overloaded"):
            list(
                AnthropicProvider("m", api_key="k", transport=transport, max_retries=0).stream(
                    [Message.user("x")]
                )
            )

    def test_thinking_blocks_are_preserved(self):
        transport = FakeTransport(
            [
                {
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                        {"type": "text", "text": "answer"},
                    ],
                    "usage": {},
                }
            ]
        )
        result = AnthropicProvider("m", api_key="k", transport=transport).complete(
            [Message.user("x")]
        )
        assert result.message.metadata["thinking_block"]["signature"] == "sig"


class TestOllamaProvider:
    def test_no_api_key_needed(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        provider = OllamaProvider("llama3.2")
        assert provider.base_url == "http://localhost:11434"
        assert not provider.api_key

    def test_host_env_is_normalised(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "192.168.1.5:11434")
        assert OllamaProvider("m").base_url == "http://192.168.1.5:11434"
        monkeypatch.setenv("OLLAMA_HOST", "https://ollama.internal")
        assert OllamaProvider("m").base_url == "https://ollama.internal"

    def test_request_and_usage(self):
        transport = FakeTransport(
            [
                {
                    "model": "llama3.2",
                    "message": {
                        "role": "assistant",
                        "content": "local",
                        "tool_calls": [{"function": {"name": "calc", "arguments": {"a": 1}}}],
                    },
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 7,
                    "eval_count": 3,
                }
            ]
        )
        provider = OllamaProvider("llama3.2", transport=transport)
        result = provider.complete([Message.user("hi")], temperature=0.2, max_tokens=64)
        assert result.text == "local"
        assert result.tool_calls[0].arguments == {"a": 1}
        assert result.usage.total_tokens == 10
        assert transport.last_payload["options"]["temperature"] == 0.2
        assert transport.last_payload["options"]["num_predict"] == 64
        assert transport.last_payload["stream"] is False

    def test_ndjson_streaming(self):
        lines = [
            '{"model":"llama3.2","message":{"content":"Hel"},"done":false}',
            '{"model":"llama3.2","message":{"content":"lo"},"done":false}',
            '{"model":"llama3.2","message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":2}',
        ]
        transport = FakeTransport(sse=[lines])
        chunks = list(OllamaProvider("llama3.2", transport=transport).stream([Message.user("hi")]))
        assert [chunk.text for chunk in chunks if chunk.kind == "text"] == ["Hel", "lo"]
        assert chunks[-1].response.usage.total_tokens == 4


class TestEchoProvider:
    def test_echoes_last_user_message(self):
        result = EchoProvider().complete([Message.user("one"), Message.user("two")])
        assert 'You said: "two"' in result.text
        assert result.usage.total_tokens > 0

    def test_streaming(self):
        chunks = list(EchoProvider().stream([Message.user("hi there")]))
        assert chunks[-1].kind == "response"
        assert "".join(chunk.text for chunk in chunks if chunk.kind == "text").startswith(
            'You said: "hi there"'
        )


class TestTransports:
    def test_iter_sse(self):
        lines = [
            ": comment",
            "event: message",
            "data: {",
            'data: "a": 1}',
            "",
            "data: [DONE]",
            "",
        ]
        events = list(iter_sse(iter(lines)))
        assert events[0] == ("message", '{\n"a": 1}')
        assert events[1] == (None, "[DONE]")

    def test_iter_sse_without_trailing_blank_line(self):
        assert list(iter_sse(iter(["data: x"]))) == [(None, "x")]

    def test_status_mapping(self):
        from pyagents.llm.transport import _status_to_error

        assert isinstance(_status_to_error("p", 401, ""), ProviderAuthError)
        assert isinstance(_status_to_error("p", 429, ""), ProviderRateLimitError)
        assert isinstance(_status_to_error("p", 504, ""), ProviderTimeoutError)
        assert isinstance(_status_to_error("p", 500, ""), ProviderError)
        error = _status_to_error("p", 400, {"error": {"message": "bad request"}})
        assert "bad request" in str(error)

    def test_urllib_transport_maps_errors(self, monkeypatch):
        import urllib.error

        def fake_urlopen(*args, **kwargs):
            raise urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        with pytest.raises(ProviderAuthError):
            UrllibTransport().post_json("http://x", headers={}, payload={}, timeout=1, provider="p")

    def test_urllib_transport_network_error(self, monkeypatch):
        import urllib.error

        def fake_urlopen(*args, **kwargs):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        with pytest.raises(ProviderError, match="connection refused"):
            UrllibTransport().post_json("http://x", headers={}, payload={}, timeout=1, provider="p")

    def test_httpx_transport_uses_client(self):
        httpx = pytest.importorskip("httpx")
        from pyagents.llm.transport import HttpxTransport

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        transport = HttpxTransport(client=client)
        assert transport.post_json("http://x", headers={}, payload={}, timeout=1) == {"ok": True}
        transport.close()

    def test_get_default_transport_prefers_httpx(self):
        from pyagents.llm.transport import (
            HttpxTransport,
            get_default_transport,
            reset_default_transport,
        )

        reset_default_transport()
        try:
            assert isinstance(get_default_transport(), HttpxTransport)
        finally:
            reset_default_transport()

    def test_openai_compatible_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        transport = FakeTransport([OPENAI_OK])
        provider = OpenAICompatibleProvider(
            "local-model", base_url="http://localhost:8000/v1", transport=transport
        )
        assert provider.complete([Message.user("x")]).text == "hello"
        assert transport.requests[0]["url"] == "http://localhost:8000/v1/chat/completions"

    def test_azure_header_option(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "azure-secret")
        transport = FakeTransport([OPENAI_OK])
        provider = OpenAIProvider(
            "deploy", api_key="azure-secret", transport=transport, azure_api_key_header=True
        )
        provider.complete([Message.user("x")])
        headers = transport.requests[0]["headers"]
        assert headers["api-key"] == "azure-secret"
        assert "api-key" not in transport.last_payload  # control options never reach the wire
