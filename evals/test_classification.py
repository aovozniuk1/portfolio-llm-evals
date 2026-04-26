"""
Ground-truth label match tests for the lead classifier.

Each parametrized case verifies the LLM's classification matches the expected
label. The aggregate test enforces a >=90% accuracy threshold across the full
dataset — that's the gate that fails CI on prompt regressions.
"""
from __future__ import annotations

import pytest

from evals.fixtures import (
    call_classifier,
    classification_dataset,
    parse_classifier_response,
)

CASES = classification_dataset()


@pytest.mark.classification
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_lead_classification_matches_expected(
    anthropic_client,
    lead_classifier_system_prompt,
    model_name,
    case,
):
    """
    Per-case test: classifier output must match the hand-labeled ground truth.

    Failure mode: an individual case mismatches. Useful for diagnosing which
    edge cases regress when the prompt changes. Doesn't fail the build by
    itself — see test_classifier_accuracy_threshold for the build gate.
    """
    raw = call_classifier(
        anthropic_client,
        lead_classifier_system_prompt,
        case["input"],
        model_name,
    )
    parsed = parse_classifier_response(raw)

    assert parsed["classification"] == case["expected"], (
        f"Misclassification on case '{case['id']}'. "
        f"Expected {case['expected']!r}, got {parsed['classification']!r}. "
        f"LLM reasoning: {parsed['reasoning']!r}"
    )


@pytest.mark.classification
@pytest.mark.slow
def test_classifier_accuracy_threshold(
    anthropic_client,
    lead_classifier_system_prompt,
    model_name,
):
    """
    Aggregate test: across the full dataset, accuracy must be >=90%.

    This is the build-gate test. Individual case failures in
    test_lead_classification_matches_expected are informative but not blocking;
    this threshold is what CI checks.

    If accuracy drops below 90%, the failure message lists the specific
    misclassified cases so a prompt engineer can iterate.
    """
    misclassified: list[tuple[str, str, str]] = []

    for case in CASES:
        raw = call_classifier(
            anthropic_client,
            lead_classifier_system_prompt,
            case["input"],
            model_name,
        )
        try:
            parsed = parse_classifier_response(raw)
        except ValueError:
            misclassified.append((case["id"], case["expected"], "PARSE_ERROR"))
            continue

        if parsed["classification"] != case["expected"]:
            misclassified.append(
                (case["id"], case["expected"], parsed["classification"])
            )

    accuracy = (len(CASES) - len(misclassified)) / len(CASES)

    if accuracy < 0.90:
        diagnostic = "\n".join(
            f"  - {cid}: expected {exp}, got {got}"
            for cid, exp, got in misclassified
        )
        pytest.fail(
            f"Classifier accuracy {accuracy:.2%} below 90% threshold "
            f"({len(CASES) - len(misclassified)}/{len(CASES)} correct).\n"
            f"Misclassified cases:\n{diagnostic}"
        )
