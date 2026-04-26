"""
Agent tool-call correctness tests.

For agentic workflows where the model decides which tools to call (vs.
classification where it picks one of N labels), correctness has two
dimensions:
  1. The right tools are called (no hallucinated tools, no skipped tools)
  2. Each tool is called with valid arguments matching the tool's schema

These tests use Anthropic's tool-use API directly. We give the model a
small set of declared tools, send it a scenario, and assert on the resulting
tool_use blocks in the response.
"""
from __future__ import annotations

import pytest

from evals.fixtures import tool_call_dataset

SCENARIOS = tool_call_dataset()


# Three example tools the agent can call. Real-world agent workflows
# might have 10-50 tools; the testing pattern is the same — just larger.
AVAILABLE_TOOLS = [
    {
        "name": "search_listings",
        "description": "Search real-estate listings by criteria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
                "bedrooms": {"type": "integer", "minimum": 0},
            },
            "required": ["city"],
        },
    },
    {
        "name": "schedule_viewing",
        "description": "Schedule a property viewing for a lead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lead_email": {"type": "string", "format": "email"},
                "listing_id": {"type": "string"},
                "preferred_time": {"type": "string"},
            },
            "required": ["lead_email", "listing_id", "preferred_time"],
        },
    },
    {
        "name": "send_email",
        "description": "Send a follow-up email to a lead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "format": "email"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


@pytest.mark.agent
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["id"])
def test_agent_calls_expected_tools(anthropic_client, model_name, scenario):
    """
    Send a scenario to the agent and verify that the tools it chose to call
    match the expected_tools list.

    The order of calls is not asserted (some scenarios have flexible ordering).
    What is asserted: the SET of tool names matches expected.
    """
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=1024,
        tools=AVAILABLE_TOOLS,
        messages=[{"role": "user", "content": scenario["user_message"]}],
    )

    called_tools = [
        block.name for block in response.content if block.type == "tool_use"
    ]

    expected_tools = set(scenario["expected_tools"])
    actual_tools = set(called_tools)

    missing = expected_tools - actual_tools
    extra = actual_tools - expected_tools

    error_parts: list[str] = []
    if missing:
        error_parts.append(f"missing required tools: {sorted(missing)}")
    if extra:
        error_parts.append(f"called unexpected tools: {sorted(extra)}")

    assert not error_parts, (
        f"Scenario '{scenario['id']}': tool-call mismatch. "
        + "; ".join(error_parts)
        + f"\nExpected: {sorted(expected_tools)}\nActual: {called_tools}"
    )


@pytest.mark.agent
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["id"])
def test_agent_does_not_hallucinate_tools(anthropic_client, model_name, scenario):
    """
    Verify the agent only calls tools from the AVAILABLE_TOOLS list.

    A model that calls an undeclared tool name (e.g., 'send_text_message'
    when only 'send_email' is declared) is a serious bug — it means the
    Pydantic / tool-router layer in production will silently drop or
    misinterpret the call.
    """
    available_names = {tool["name"] for tool in AVAILABLE_TOOLS}

    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=1024,
        tools=AVAILABLE_TOOLS,
        messages=[{"role": "user", "content": scenario["user_message"]}],
    )

    called = [block.name for block in response.content if block.type == "tool_use"]
    hallucinated = [name for name in called if name not in available_names]

    assert not hallucinated, (
        f"Scenario '{scenario['id']}': agent hallucinated tool names "
        f"not in the declared toolset: {hallucinated}. "
        f"Available tools: {sorted(available_names)}"
    )
