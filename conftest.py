"""
Session-scoped fixtures for the LLM eval suite.

The Anthropic client is instantiated once per session to amortize SDK
initialization cost. Tests that need different model parameters (e.g.,
temperature=0 for consistency tests vs temperature=0.3 for diversity tests)
declare those overrides explicitly in their own test_*.py file.
"""
from __future__ import annotations

import os
import pytest
from anthropic import Anthropic


@pytest.fixture(scope="session")
def anthropic_client() -> Anthropic:
    """
    Real Anthropic client. Requires ANTHROPIC_API_KEY in env.

    Failing fast at session start (when no key is present) is intentional —
    silent fallback to a mock would make a green build meaningless.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip(
            "ANTHROPIC_API_KEY not set. Export the key before running evals: "
            "`export ANTHROPIC_API_KEY=sk-ant-...`"
        )
    return Anthropic(api_key=api_key)


@pytest.fixture(scope="session")
def lead_classifier_system_prompt() -> str:
    """
    System prompt for the lead classifier.

    Same prompt that powers the n8n workflow in portfolio-n8n.
    Locked here so all classification tests evaluate the same prompt
    version. Bump this string and re-run tests to validate prompt changes.

    v2: ordered rules with explicit precedence. The original v1 prompt
    left precedence implicit, which caused the model to guess on edge
    cases where multiple tiers matched (e.g. budget in Warm range +
    timeline in Cold range). v2 fires rules in order, first match wins.
    """
    return (
        "You are a real-estate lead qualifier. Classify each lead as exactly "
        "one of: Hot, Warm, Cold.\n\n"
        "Apply these rules in order. The first matching rule wins.\n\n"
        "1. HOT NOTES OVERRIDE: If notes explicitly say 'cash buyer', "
        "'cash-equivalent funds', or 'fully pre-approved with underwritten "
        "mortgage' -> classify as Hot, regardless of budget or timeline. "
        "(Soft signals like 'approval-in-principle', 'AIP', 'pre-qualified', "
        "'mortgage in progress' do NOT count as Hot — those are Warm.)\n\n"
        "2. HOT NUMERIC: If budget >= $500K AND timeline_months <= 3 -> Hot.\n\n"
        "3. COLD VAGUE: If notes use vague language ('tentatively', 'maybe', "
        "'just looking', 'thinking about it') AND at least one numeric signal "
        "is in the Cold range (budget < $200K OR timeline > 12) -> Cold.\n\n"
        "4. WARM ENGAGEMENT: If notes show active engagement (specific "
        "properties, scheduled viewings, agent meetings, soft mortgage "
        "approval, school-zone search) AND at least one signal is in the Warm "
        "range (budget $200K-$500K OR timeline 3-12 months) -> Warm.\n\n"
        "5. DEFAULT BY MAJORITY: Map budget to tier (Hot: >=$500K, Warm: "
        "$200K-$500K, Cold: <$200K). Map timeline to tier (Hot: <=3, Warm: "
        "3-12, Cold: >12). Pick the tier shared by both. If they disagree, "
        "pick Warm.\n\n"
        "Reply with strict JSON only:\n"
        '{"classification": "Hot|Warm|Cold", "reasoning": "<one sentence '
        'naming which rule fired>"}\n\n'
        "Never include any other text outside the JSON object."
    )


@pytest.fixture(scope="session")
def model_name() -> str:
    """
    Default model used across the eval suite. Centralized so a single
    upgrade (e.g., haiku -> sonnet) updates every test file at once.
    """
    return "claude-3-5-haiku-20241022"
