# Design Decisions

## Why pytest, not DeepEval / LangSmith / Promptfoo?

DeepEval, LangSmith, Promptfoo, and Braintrust are excellent products. They're not the right fit for every team.

**When the eval-platform vendors win:**
- Team needs cross-model benchmarking (Claude vs GPT-4 vs Llama) → Promptfoo
- Team wants a hosted dashboard with traces, replay, and shared datasets → LangSmith / Braintrust
- Team has dedicated ML engineers running offline evaluation campaigns → DeepEval
- Team wants out-of-box metrics like BLEU / ROUGE / GEval → DeepEval / LangSmith

**When pytest-native wins:**
- Team already runs pytest in CI (the most common case for B2B SaaS)
- Team wants tests-as-code, reviewed in PRs, committed to the same repo as the prompt
- Team wants the same retry / quarantine / Allure-reporting muscle memory engineers already have
- Team has small dataset sizes (under 1000 cases per dimension)
- Team wants zero new vendors / dashboards to learn

For most production AI features at most companies, pytest-native is the lower-friction path. This repo is the reference implementation.

## Why deterministic-first, LLM-judge as opt-in?

Every assertion in this repo is deterministic where possible: exact label match, JSON schema validation, enum membership, forbidden-marker absence.

LLM-as-judge is reserved for genuinely subjective cases (e.g., "is this generated email natural-sounding?") where ground-truth labels don't exist.

**Why this order matters:**

| Property | Deterministic assertion | LLM-judge assertion |
|---|---|---|
| Cost per check | $0 (local Python) | ~$0.001-0.005 (one Claude/GPT call) |
| Speed | <1ms | 200-2000ms |
| Drift over time | None | Yes (judge model evolves) |
| Reproducibility | Perfect | Approximate (even at temperature=0) |
| Audit defensibility | Trivial | Requires judge prompt + version pinning |

In a CI gate run hundreds of times a week, deterministic checks accumulate runtime savings, are reproducible across teams, and don't drift when the judge model is updated by its vendor.

Use LLM-judge when you need to, not when you can.

## Why threshold gates instead of per-case pass/fail?

`test_classifier_accuracy_threshold` runs the entire dataset and fails CI if aggregate accuracy drops below 90%. Individual case failures in the parametrized `test_lead_classification_matches_expected` are diagnostic, not gating.

Why this design:
- Prompt engineering is a moving target. A change that fixes case A might break case B. If every case is a gate, the team plays whack-a-mole.
- The threshold (90%) is the team's contract with the rest of the system. Hitting it means the prompt is doing its job. Individual case mismatches are normal as long as the rate stays above the gate.
- The diagnostic test still fires per-case, so when the threshold drops below 90%, the engineer sees exactly which cases regressed.

This pattern transfers from traditional QA: SLOs are aggregate, not per-request. Test the SLO, surface per-case data for triage.

## Why Claude Haiku for the example?

Three reasons:

1. **Cost.** Haiku at ~$0.0003/lead means even the full eval suite (~78 calls) costs ~$0.024 per run. CI affordability matters when you run on every PR.

2. **The job is classification, not reasoning.** Haiku is sufficient for "given these lead signals, output Hot/Warm/Cold." Sonnet would be over-engineered. Opus would be wasteful.

3. **Realistic for production AI features.** Most companies use Haiku-tier models for high-volume, low-complexity tasks (classification, routing, extraction). Testing with the model you'd actually deploy is the only meaningful test.

For dimensions that genuinely need a stronger model — e.g., a future `test_factual_accuracy.py` with hallucination detection over open-domain questions — that test would explicitly request Sonnet or Opus in its own fixture.

## Why pytest-xdist parallelism by default?

Each Claude API call takes ~500-2000ms. The full eval suite has ~78 calls. Sequentially that's 1-3 minutes. With `-n auto` (typically 4-8 workers), it's 15-30 seconds.

CI engineers care about pipeline duration. Adding 90 seconds of LLM eval time to every PR is acceptable; adding 3 minutes is not.

**Caveat:** Anthropic's tier-based rate limits can throttle high parallelism on lower tiers. The default is conservative; teams on Tier 1+ can crank `-n` higher.

## Why fail fast (`-x` in pytest.ini)?

A fundamentally broken prompt will fail every case. Without `-x`, the suite runs 78 useless API calls before reporting. With `-x`, it stops on the first failure and saves both runtime and money.

For production diagnostic runs (e.g., "show me all failures across the dataset for triage"), the engineer can override with `pytest --no-cov -x` or simply `pytest`.

## Open questions / TODO

- **Token-level eval metrics (BLEU, ROUGE, semantic similarity).** Out of scope for v1. If a team needs them, they should add a complementary DeepEval suite — pytest-native isn't the right tool for these.
- **LLM-as-judge for soft criteria.** This repo's `test_safety.py` does basic marker-detection. A future version could add a Sonnet-judged "is this response on-brand and polite?" test. The judge prompt and inter-rater agreement validation would need their own design doc.
- **Cross-run accuracy tracking.** Currently each CI run reports its accuracy in isolation. A persistent "accuracy over time" graph (showing prompt changes vs accuracy regressions) would close the loop on prompt engineering. Needs a metrics backend (DataDog / Honeycomb / etc.) — out of scope here.

## Compatibility

Tested against:
- Python 3.11, 3.12
- anthropic SDK >= 0.40
- pytest >= 8.0
- jsonschema >= 4.21
- claude-3-5-haiku-20241022 (default)
- claude-3-5-sonnet-20241022 (drop-in via `model_name` fixture override)

Not tested but should work:
- OpenAI / GPT-4 (would need an OpenAI client fixture instead of the Anthropic one in `conftest.py`)
- Local LLMs via Ollama (would need a different client wrapper, but the test patterns are model-agnostic)
