"""
Dataset loaders + scoring helpers shared across test files.

Each dataset is a JSONL file under datasets/ — one record per line.
Loaders return generators so pytest's parametrize lazy-iterates rather
than loading everything into memory (matters once datasets exceed ~10K).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _load_jsonl(filename: str) -> Iterator[dict]:
    path = DATASETS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {filename} not found at {path}. "
            f"Did you forget to clone with --recurse or check out the dataset?"
        )
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{filename} line {line_no} is not valid JSON: {e}"
                ) from e


def classification_dataset() -> list[dict]:
    """Hand-labeled lead classification cases. Used by test_classification.py."""
    return list(_load_jsonl("classification_cases.jsonl"))


def injection_dataset() -> list[dict]:
    """OWASP-style prompt-injection payloads. Used by test_safety.py."""
    return list(_load_jsonl("injection_attacks.jsonl"))


def tool_call_dataset() -> list[dict]:
    """Multi-step agent scenarios with expected tool sequences. Used by test_agent_tool_calls.py."""
    return list(_load_jsonl("tool_call_scenarios.jsonl"))


def parse_classifier_response(raw_text: str) -> dict:
    """
    Parse a classifier response with strict validation.

    Raises ValueError with a clear message if the response isn't valid JSON
    or doesn't have the required schema. Tests catch this to produce
    diagnostic failures rather than opaque tracebacks.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e}\nRaw: {raw_text!r}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"Response must be a JSON object, got {type(parsed).__name__}")

    for required in ("classification", "reasoning"):
        if required not in parsed:
            raise ValueError(
                f"Response missing required field '{required}'. Got keys: {list(parsed.keys())}"
            )

    if parsed["classification"] not in ("Hot", "Warm", "Cold"):
        raise ValueError(
            f"Invalid classification value: {parsed['classification']!r}. "
            "Must be one of: Hot, Warm, Cold."
        )

    return parsed


def call_classifier(client, system_prompt: str, user_input: str, model: str, temperature: float = 0.0) -> str:
    """
    Single Claude API call returning raw text. Centralized so changes to
    the call signature (e.g., adding stop_sequences, switching to streaming)
    propagate to every test at once.
    """
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )
    return response.content[0].text
