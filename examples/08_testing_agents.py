"""Testing agents without spending tokens.

PyAgents ships fakes for exactly this: ``ScriptedProvider`` replays a fixed
script of answers, ``FakeTransport`` stands in for HTTP, and ``EchoProvider``
gives you a working provider offline. This file is written in pytest style -
copy it into your own test suite as a starting point.

Run it as a script to see the assertions pass, or with pytest:

    $ python examples/08_testing_agents.py
    $ pytest examples/08_testing_agents.py
"""

from __future__ import annotations

from _bootstrap import section
from pyagents import Agent
from pyagents.errors import ProviderError
from pyagents.testing import FakeTransport, ScriptedProvider, text_response, tool_response
from pyagents.tools import calculator, tool


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order in the (fake) database.

    Args:
        order_id: Order identifier such as ``"A-100"``.
    """
    orders = {"A-100": {"status": "shipped", "eta_days": 2}}
    return orders.get(order_id, {"error": "not found"})


def test_agent_calls_the_right_tool() -> None:
    provider = ScriptedProvider(
        [tool_response("lookup_order", {"order_id": "A-100"}), text_response("It ships in 2 days.")]
    )
    agent = Agent("support", provider=provider, tools=[lookup_order])

    result = agent.run("Where is order A-100?")

    assert result.output == "It ships in 2 days."
    assert result.steps[0].message.tool_calls[0].name == "lookup_order"
    assert result.tool_results[0].output["status"] == "shipped"


def test_agent_survives_a_failing_tool() -> None:
    provider = ScriptedProvider(
        [
            tool_response("lookup_order", {"order_id": "NOPE"}),
            text_response("I could not find that order."),
        ]
    )
    agent = Agent("support", provider=provider, tools=[lookup_order], max_steps=3)

    result = agent.run("Where is order NOPE?")

    assert result.status.value == "completed"
    assert "error" in result.tool_results[0].content


def test_prompt_contains_the_system_message_and_task() -> None:
    provider = ScriptedProvider([text_response("ok")])
    agent = Agent("tester", instructions="You are terse.", provider=provider)
    agent.run("hello")

    messages = provider.last_messages()
    assert messages[0].role.value == "system"
    assert "terse" in messages[0].text
    assert provider.last_user_message() == "hello"


def test_tools_offered_to_the_model() -> None:
    provider = ScriptedProvider([text_response("ok")])
    Agent("tester", provider=provider, tools=[calculator, lookup_order]).run("hi")
    assert {spec["name"] for spec in provider.tools_seen[-1]} == {"calculator", "lookup_order"}


def test_provider_errors_propagate_with_partial_results() -> None:
    def explode(_messages):
        raise ProviderError("upstream is down", provider="scripted")

    agent = Agent("tester", provider=ScriptedProvider([explode], default_text=""))
    try:
        agent.run("hi")
    except ProviderError as exc:
        partial = getattr(exc, "run_result", None)
        assert partial is not None
        assert partial.status.value == "failed"
    else:  # pragma: no cover - the call must raise
        raise AssertionError("ProviderError was not raised")


def test_http_level_provider_with_fake_transport() -> None:
    from pyagents import OpenAIProvider
    from pyagents.types import Message

    transport = FakeTransport(
        [
            {
                "choices": [{"message": {"content": "from openai"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ]
    )
    provider = OpenAIProvider("gpt-4o-mini", api_key="sk-fake", transport=transport)
    assert provider.complete([Message.user("hi")]).text == "from openai"
    assert transport.call_count == 1


def main() -> None:
    tests = [
        test_agent_calls_the_right_tool,
        test_agent_survives_a_failing_tool,
        test_prompt_contains_the_system_message_and_task,
        test_tools_offered_to_the_model,
        test_provider_errors_propagate_with_partial_results,
        test_http_level_provider_with_fake_transport,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"pass {test.__name__}")
    section(f"{len(tests) - failures}/{len(tests)} example tests passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
