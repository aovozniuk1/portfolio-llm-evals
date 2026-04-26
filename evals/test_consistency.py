"""
Determinism / consistency tests.

At temperature=0, identical inputs should produce identical outputs.
In practice, LLM determinism isn't perfect even at T=0 (tokenization order,
batched inference) — but classification labels should be stable.

These tests catch:
- Prompt / model changes that introduce label flapping
- Hidden non-determinism (e.g., system_fingerprint changes between calls)
- Race conditions in the call layer
"""
from __future__ import annotations

import pytest

from evals.fixtures import (
    call_classifier,
    classification_dataset,
    parse_classifier_response,
)

# Smoke subset — 5 cases is enough to detect flapping; full dataset is overkill
SMOKE_CASES = classification_dataset()[:5]

REPEAT_COUNT = 3


@pytest.mark.consistency
@pytest.mark.slow
@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda c: c["id"])
def test_classification_is_deterministic_at_temperature_zero(
    anthropic_client,
    lead_classifier_system_prompt,
    model_name,
    case,
):
    """
    Run the same input N times at temperature=0 and verify all responses
    produce the same classification label.

    Reasoning text may vary (LLM determinism is not absolute), but the
    classification label should be 100% stable.
    """
    classifications: list[str] = []

    for _ in range(REPEAT_COUNT):
        raw = call_classifier(
            anthropic_client,
            lead_classifier_system_prompt,
            case["input"],
            model_name,
            temperature=0.0,
        )
        parsed = parse_classifier_response(raw)
        classifications.append(parsed["classification"])

    unique = set(classifications)
    assert len(unique) == 1, (
        f"Case '{case['id']}': classification flapped at temperature=0. "
        f"Got {classifications} across {REPEAT_COUNT} runs. "
        "Either the prompt has nondeterministic phrasing, or the model is "
        "near a decision boundary on this input."
    )
