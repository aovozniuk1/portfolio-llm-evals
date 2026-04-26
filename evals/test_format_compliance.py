"""
Format-compliance tests for the lead classifier.

Independent of classification accuracy: even if the classifier picks the
wrong label, the response must be parseable JSON with the expected schema.
A regression here breaks downstream consumers (e.g., the n8n workflow's
Switch node) regardless of label correctness.
"""
from __future__ import annotations

import json
import pytest
from jsonschema import Draft7Validator, ValidationError

from evals.fixtures import call_classifier, classification_dataset

CASES = classification_dataset()

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["Hot", "Warm", "Cold"],
        },
        "reasoning": {
            "type": "string",
            "minLength": 10,
            "maxLength": 500,
        },
    },
    "required": ["classification", "reasoning"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def schema_validator() -> Draft7Validator:
    return Draft7Validator(CLASSIFIER_SCHEMA)


@pytest.mark.format
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_response_is_valid_json(
    anthropic_client,
    lead_classifier_system_prompt,
    model_name,
    case,
):
    """Response must be parseable JSON. No prose, no preamble, no trailing text."""
    raw = call_classifier(
        anthropic_client,
        lead_classifier_system_prompt,
        case["input"],
        model_name,
    )
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Case '{case['id']}': response is not valid JSON.\n"
            f"Error: {e}\n"
            f"Raw response: {raw!r}"
        )


@pytest.mark.format
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_response_matches_schema(
    anthropic_client,
    lead_classifier_system_prompt,
    model_name,
    schema_validator,
    case,
):
    """
    Response must match the strict JSON schema:
    - classification ∈ {Hot, Warm, Cold}
    - reasoning is a 10-500 char string
    - no extra fields

    Catches: missing fields, invalid enum values, prose where JSON is expected,
    extra fields that downstream parsers may choke on.
    """
    raw = call_classifier(
        anthropic_client,
        lead_classifier_system_prompt,
        case["input"],
        model_name,
    )

    parsed = json.loads(raw)
    errors = sorted(schema_validator.iter_errors(parsed), key=lambda e: e.path)

    if errors:
        diagnostic = "\n".join(
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}"
            for err in errors
        )
        pytest.fail(
            f"Case '{case['id']}': response failed schema validation.\n"
            f"Schema errors:\n{diagnostic}\n"
            f"Raw response: {raw!r}"
        )


@pytest.mark.format
def test_reasoning_field_is_not_empty(
    anthropic_client,
    lead_classifier_system_prompt,
    model_name,
):
    """
    Sanity check: across the dataset, no response should have empty reasoning.
    A model that returns {"classification": "Hot", "reasoning": ""} satisfies
    the schema technically but is useless to humans reviewing the audit trail.
    """
    empty_reasonings: list[str] = []

    for case in CASES[:5]:  # smoke check, not full sweep
        raw = call_classifier(
            anthropic_client,
            lead_classifier_system_prompt,
            case["input"],
            model_name,
        )
        parsed = json.loads(raw)
        if not parsed.get("reasoning", "").strip():
            empty_reasonings.append(case["id"])

    assert not empty_reasonings, (
        f"Cases with empty reasoning field: {empty_reasonings}. "
        "Reasoning must be non-empty for human review of the audit trail."
    )
