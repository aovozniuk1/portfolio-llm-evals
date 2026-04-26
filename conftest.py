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
    """
    return (
        "You are a real-estate lead qualifier. Classify each lead as exactly "
        "one of: Hot, Warm, Cold.\n\n"
        "Hot = budget >= $500K AND timeline_months <= 3 OR notes mention "
        "'cash buyer', 'pre-approved', 'cash-equivalent funds'.\n"
        "Warm = budget $200K-$500K OR timeline 3-12 months OR active "
        "interest expressed.\n"
        "Cold = budget < $200K OR timeline > 12 months OR vague interest.\n\n"
        "Reply with strict JSON only:\n"
        '{"classification": "Hot|Warm|Cold", "reasoning": "<one sentence>"}\n\n'
        "Never include any other text outside the JSON object."
    )


@pytest.fixture(scope="session")
def model_name() -> str:
    """
    Default model used across the eval suite. Centralized so a single
    upgrade (e.g., haiku -> sonnet) updates every test file at once.
    """
    return "claude-3-5-haiku-20241022"
