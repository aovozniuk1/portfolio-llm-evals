# LLM Evals & Agent Testing — pytest-native harness

A senior QA-engineer's take on LLM testing. Treats LLM outputs like any other test target: define expected behavior → run → assert → report. No dashboards. No SaaS lock-in. Just pytest.

Tests an example real-estate lead classifier (the same prompt that powers the [n8n lead qualifier](https://github.com/aovozniuk1/portfolio-n8n) workflow) across five dimensions:

| Dimension | What it catches |
|---|---|
| **Classification accuracy** | Prompt regressions that drop ground-truth label match below 90% |
| **Format compliance** | Invalid JSON, missing required fields, out-of-spec enum values |
| **Consistency** | Same input producing different outputs at temperature=0 |
| **Safety / prompt injection** | OWASP LLM Top 10 attacks that override the system prompt or leak attack markers |
| **Agent tool-call correctness** | Tool-calling sequences that hallucinate tools, skip required tools, or pass malformed arguments |

## Why pytest-native

Most LLM eval frameworks (DeepEval, LangSmith, Promptfoo, Braintrust) are excellent at what they do, but they're SaaS dashboards or DSL-driven. For a QA team that already runs pytest in CI, the cleanest integration is:

```python
@pytest.mark.parametrize("case", classification_dataset())
def test_lead_classification_accuracy(client, system_prompt, case):
    response = client.classify(case["input"])
    assert response["classification"] == case["expected"]
```

Each LLM eval becomes a test case. The team's existing CI matrix, retry policies, flake quarantine, Allure reporting, and engineer muscle memory all carry over.

This repo is the reference implementation.

## Stack

- **Python 3.11+**
- **pytest** + `pytest-html` + `pytest-xdist` for reporting and parallelism
- **Anthropic SDK** for Claude API calls (Haiku for cheap classification, Sonnet for harder evals)
- **JSONL datasets** (one fixture file per dimension)

## Quick start

```bash
git clone https://github.com/aovozniuk1/portfolio-llm-evals.git
cd portfolio-llm-evals
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
pytest evals/ -v                    # full suite
pytest evals/test_classification.py # one dimension
pytest evals/ -n auto               # parallel via pytest-xdist
pytest evals/ --html=report.html    # HTML report
```

## Repo layout

```
portfolio-llm-evals/
├── README.md
├── LICENSE
├── requirements.txt
├── pytest.ini
├── conftest.py                          # session-scoped LLM client fixture
├── evals/
│   ├── __init__.py
│   ├── fixtures.py                      # dataset loaders, scoring helpers
│   ├── test_classification.py           # ground-truth label match + accuracy threshold
│   ├── test_format_compliance.py        # JSON validity + required fields + enum values
│   ├── test_consistency.py              # determinism at temperature=0
│   ├── test_safety.py                   # prompt-injection resistance (OWASP LLM Top 10)
│   └── test_agent_tool_calls.py         # tool-calling correctness for agentic flows
├── datasets/
│   ├── classification_cases.jsonl       # 20 hand-labeled lead cases
│   ├── injection_attacks.jsonl          # 10 OWASP-style injection payloads
│   └── tool_call_scenarios.jsonl        # 8 multi-step agent scenarios
├── ci/
│   └── github-actions-eval.yml          # run on every PR, fail if accuracy <90%
└── docs/
    └── design-decisions.md              # why deterministic-first, when LLM-judge is needed
```

## Patterns demonstrated

### 1. Deterministic-first, LLM-judge as opt-in

Most assertions in this repo are deterministic: exact label match, JSON schema validation, enum membership. These are cheap, fast, and don't drift over time. LLM-as-judge (using Claude Sonnet to grade Claude Haiku's responses) is reserved for genuinely subjective cases — e.g., "is this generated email natural-sounding?" — where ground-truth labels don't exist.

This matches how mature QA teams treat AI features: lock down what can be locked down, accept ambiguity only where it's irreducible.

### 2. Threshold gates, not pass/fail per case

`test_classifier_accuracy_threshold` runs the entire dataset and fails CI if aggregate accuracy drops below 90%. Individual case failures are diagnostic — the gate is the threshold. This avoids prompt-engineering whack-a-mole where one fix breaks two other cases.

### 3. Same prompt, multiple test angles

The lead-classifier prompt is tested for accuracy, format compliance, consistency, and injection resistance — independently. A regression in one dimension doesn't mask regressions in others. Each test file isolates its concern.

### 4. CI-friendly cost control

The `pytest.ini` `addopts` setting includes `-x` (stop on first failure) by default. The full eval suite costs ~$0.10 per run on Claude Haiku. For a team running on every PR, that's ~$30/mo at typical PR volume — affordable, and the `-x` flag prevents runaway costs on a fundamentally broken prompt.

## Cost estimate

| Test file | API calls per run | Cost (Haiku) |
|---|---|---|
| test_classification.py | ~22 (20 cases + 2 aggregates) | ~$0.007 |
| test_format_compliance.py | ~20 | ~$0.006 |
| test_consistency.py | ~10 (5 cases × 2 runs each) | ~$0.003 |
| test_safety.py | ~10 | ~$0.003 |
| test_agent_tool_calls.py | ~16 (8 scenarios × 2 calls avg) | ~$0.005 |
| **Total per full run** | **~78 calls** | **~$0.024** |

At 1 PR per day = ~$0.72/mo. Even at 100 PRs per day = ~$72/mo.

## Limitations and where this approach breaks

- **Token-level metrics (BLEU, ROUGE, semantic similarity):** outside the scope of this repo. For tasks where exact-match isn't enough (e.g., generated documentation testing), use DeepEval or Promptfoo as a complement, not a replacement.
- **Very large datasets (>10K cases):** pytest's parametrize gets slow. Consider running these via DeepEval's bulk API or a custom runner.
- **Cross-model comparison:** this repo tests one model per run. For "Haiku vs Sonnet vs GPT-4o" benchmarking, use Promptfoo (it's purpose-built for that).
- **LLM-as-judge for subjective tasks:** `test_safety.py` shows a basic pattern, but for production "is this response polite/empathetic/on-brand" testing, you'll need a more sophisticated judge prompt and inter-rater agreement validation.

## License

MIT — use, modify, ship in production. Attribution appreciated, not required.

## About

I'm Andrii Vozniuk, a Senior QA Automation Engineer with 9 years of production experience (currently at N-iX, Kyiv). Previously: Kyivstar (Ukraine's largest telecom), Form.com (low-code SaaS), Genesis (ad-tech). Bridging traditional test infrastructure into LLM/agent testing.

- LinkedIn: https://www.linkedin.com/in/[handle]/
- GitHub: https://github.com/aovozniuk1
- Other portfolios: [QA frameworks](https://github.com/aovozniuk1/portfolio-qa) · [Python dev / FastAPI / RAG](https://github.com/aovozniuk1/portfolio-dev) · [n8n automation](https://github.com/aovozniuk1/portfolio-n8n)
