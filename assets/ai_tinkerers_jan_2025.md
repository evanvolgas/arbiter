# Arbiter Presentation - AI Tinkerers
**Date:** January 20, 2025

## Setup Instructions

**Clone and Install:**
```bash
# Clone the repository
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

**Required API Key:**
```bash
export OPENAI_API_KEY=your_key_here
```

**Optional (for provider comparison demos):**
```bash
export ANTHROPIC_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
```

All examples work with just OpenAI. Additional providers only needed for `provider_switching.py` demo.

---

## Basic Evaluation
### `examples/basic_evaluation.py` lines 47-84

```python
from arbiter import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Cost: ${await result.total_llm_cost():.6f}")
print(f"Time: {result.processing_time:.2f}s")
print(f"LLM Calls: {len(result.interactions)}")
```

**Talking points:**
- Automatic cost tracking with real pricing data
- Complete observability of all LLM interactions
- No manual instrumentation required

---

## Debugging Multi-Call Systems
### `examples/debugging_multi_call.py`

```python
# Run a 15-call customer support agent
data = await run_multi_stage_agent(
    "I can't log in to my account, getting authentication error"
)

results = data["results"]  # 15 EvaluationResults

# Show what was tracked
print_interaction_breakdown(results)    # All 15 calls with tokens/latency
await print_cost_analysis(results)      # Cost by stage
print_performance_analysis(results)     # Slowest calls
print_debug_example(results)            # Full prompt/response inspection
```

**Output shows:**
- Call breakdown: All 15 calls across 5 stages with token counts and latency
- Cost analysis: Total cost ($0.002652), cost by stage with percentages
- Performance: Slowest calls identified (routing: 12.34s, context_assembly: 9.20s)
- Debug inspection: Full prompt and response for any call

**Code structure:**
```python
# Each evaluate() returns an EvaluationResult
result = await evaluate(...)

# Access tracked data
result.interactions       # List of all LLM calls
result.total_tokens       # Sum of tokens
result.processing_time    # Total latency
await result.total_llm_cost()  # Calculated cost

# Each interaction contains
interaction.prompt        # Full prompt sent
interaction.response      # Full response
interaction.tokens_used   # Token count
interaction.latency       # Call duration
interaction.model         # Which model
```

**Run it:**
```bash
python examples/debugging_multi_call.py
```

---

## Provider flexibility
### `examples/provider_switching.py` lines 54-93

```python

# OpenAI
result1 = await evaluate(
    output=output,
    reference=reference,
    evaluators=["semantic"],
    provider=Provider.OPENAI,
    model="gpt-4o-mini"
)

# Anthropic - SAME CODE
result2 = await evaluate(
    output=output,
    reference=reference,
    evaluators=["semantic"],
    provider=Provider.ANTHROPIC,
    model="claude-sonnet-4-5-20250929"
)

# Google - SAME CODE
result3 = await evaluate(
    output=output,
    reference=reference,
    evaluators=["semantic"],
    provider=Provider.GOOGLE,
    model="gemini-1.5-pro"
)

# Same code across providers
```

**Supported Providers:**

Any model from these providers (via PydanticAI):
- OpenAI (GPT-4o, o1, o3-mini, etc.)
- Anthropic (Claude Sonnet 4.5, Claude 3.5/3 family)
- Google (Gemini 2.0 Flash, 1.5 Pro/Flash)
- Groq (Llama, Mixtral, Gemma)
- Mistral AI (all models)
- Cohere (all models)

---

## Production-Grade Middleware
### `examples/circuit_breaker_example.py`

```python
from arbiter import evaluate
from arbiter.core.middleware import CircuitBreakerMiddleware, MiddlewarePipeline

# Protect against cascading failures
circuit_breaker = CircuitBreakerMiddleware(
    failure_threshold=3,    # Open after 3 failures
    recovery_timeout=60.0,  # Try again after 60s
    half_open_max_calls=2   # Test with 2 calls
)

pipeline = MiddlewarePipeline()
pipeline.add(circuit_breaker)

# If LLM calls start failing, circuit opens automatically
# Prevents costly retry storms
result = await evaluate(
    output=output,
    reference=reference,
    evaluators=["semantic"],
    middleware=pipeline,
    model="gpt-4o-mini"
)

# Circuit breaker state: closed (healthy) | open (failing) | half_open (testing)
print(f"Circuit state: {circuit_breaker.state}")
```

**Other Middleware Available:**
- **Rate Limiting**: Control evaluation throughput
- **Caching**: Avoid redundant LLM calls
- **Logging**: Structured observability
- **Metrics**: Performance monitoring

**Talking points:**
- DeepEval, TruLens, Phoenix - none have this
- They assume evaluations always work
- Arbiter treats evaluation as production infrastructure

---

## Custom Evaluations & Evaluators
### Part A: `examples/custom_criteria_example.py` lines 43-70

**Example: Domain-specific evaluation**
```python
# Medical domain evaluation
result = await evaluate(
    output="""Diabetes management requires careful attention to diet, exercise,
    and medication. Patients should monitor their blood glucose levels regularly...""",
    criteria="Medical accuracy, HIPAA compliance, appropriate tone, clarity",
    evaluators=["custom_criteria"],
    model="gpt-4o-mini"
)

# Shows which criteria were met/not met
for score in result.scores:
    if score.metadata.get("criteria_met"):
        print(f"Criteria Met: {score.metadata['criteria_met']}")
    if score.metadata.get("criteria_not_met"):
        print(f"Criteria Not Met: {score.metadata['criteria_not_met']}")
    print(f"Explanation: {score.explanation}")
```

### Part B: `arbiter/arbiter/evaluators/custom_criteria.py`, lines 104-418

**Template method pattern - 4 required methods:**

```python
from arbiter.evaluators import BasePydanticEvaluator
from pydantic import BaseModel, Field
from typing import Optional, Type

class CustomCriteriaEvaluator(BasePydanticEvaluator):
    # Method 1: Name your evaluator
    @property
    def name(self) -> str:
        return "custom_criteria"

    # Method 2: Define system prompt (evaluator's role)
    def _get_system_prompt(self) -> str:
        return """You are an expert evaluator who assesses outputs against
        specific criteria. Provide scores and identify which criteria are met."""

    # Method 3: Define user prompt (what to evaluate)
    def _get_user_prompt(self, output: str, reference: Optional[str],
                        criteria: Optional[str]) -> str:
        prompt = f"Evaluate this output:\n\n{output}\n\n"
        if criteria:
            prompt += f"Criteria: {criteria}\n\n"
        if reference:
            prompt += f"Reference: {reference}\n\n"
        return prompt

    # Method 4: Define response structure (Pydantic model)
    def _get_response_type(self) -> Type[BaseModel]:
        class CustomCriteriaResponse(BaseModel):
            score: float = Field(..., ge=0.0, le=1.0)
            confidence: float = Field(..., ge=0.0, le=1.0)
            criteria_met: list[str] = Field(default_factory=list)
            criteria_not_met: list[str] = Field(default_factory=list)
            explanation: str

        return CustomCriteriaResponse

    # Method 5: Extract score from LLM response
    async def _compute_score(self, response: BaseModel) -> Score:
        return Score(
            name=self.name,
            value=response.score,
            confidence=response.confidence,
            explanation=response.explanation,
            metadata={
                "criteria_met": response.criteria_met,
                "criteria_not_met": response.criteria_not_met
            }
        )

# 4 methods. Inherits automatic tracking from base class.
```

---

## Built-in Evaluators & Registry

**How string-based evaluators work:**

```python
# String lookup
evaluators=["custom_criteria"]

# Registry maps strings to classes (arbiter/core/registry.py)
AVAILABLE_EVALUATORS = {
    "semantic": SemanticEvaluator,           # Meaning comparison
    "custom_criteria": CustomCriteriaEvaluator,  # Domain-specific
    "factuality": FactualityEvaluator,       # Truth checking
    "groundedness": GroundednessEvaluator,   # Source citation
    "relevance": RelevanceEvaluator,         # Query answering
}

# String → Class → Instance → Evaluation
```

**Quick Comparison:**

| Evaluator | Question | Example |
|-----------|----------|---------|
| **semantic** | "Same meaning?" | "Paris is capital" ≈ "Capital is Paris" |
| **custom_criteria** | "Meets criteria?" | Medical accuracy, HIPAA compliance |
| **factuality** | "Is it TRUE?" | "Paris founded 1985" FALSE (false fact) |
| **groundedness** | "In the SOURCE?" | "Paris 2.2M people" FALSE (source says 2.16M) |
| **relevance** | "Answers query?" | "Eiffel Tower height" for "What's capital?" FALSE |



---

## Why Arbiter  Matters

**Before Arbiter (Manual Tracking):**
```python
# WITHOUT Arbiter: Manual tracking nightmare
import time
from openai import AsyncOpenAI

client = AsyncOpenAI()

# YOU have to track EVERYTHING manually:
start = time.time()
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Evaluate this..."}]
)
latency = time.time() - start

# Extract tokens manually
tokens = response.usage.total_tokens
input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens

# Calculate cost manually (you have to build this!)
cost = calculate_cost_somehow(tokens, "gpt-4o-mini")

# Log manually (you have to remember!)
logger.info(f"Call took {latency}s, used {tokens} tokens, cost ${cost}")

# ↑ Repeat for EVERY LLM call
# ↑ Forget once? Missing data.
# ↑ 15 calls = 15 copies of this code
```

**With Arbiter (Automatic Tracking):**
```python
# WITH Arbiter: Inheritance handles everything
from arbiter import evaluate

result = await evaluate(
    output="Evaluate this...",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

# BasePydanticEvaluator.evaluate() tracked EVERYTHING automatically:
# - Lines 293-372 in base.py handle ALL tracking
# - Every evaluator inherits this behavior
# - ZERO code in user's application

print(f"Calls: {len(result.interactions)}")
print(f"Cost: ${await result.total_llm_cost():.4f}")

for interaction in result.interactions:
    print(f"{interaction.purpose}: {interaction.latency:.2f}s, ${interaction.cost:.6f}")
    print(f"  Prompt: {interaction.prompt[:100]}...")
    print(f"  Response: {interaction.response[:100]}...")
```
---

## Why Arbiter vs Competitors?

### DeepEval (2.8k stars)
```
DeepEval: Pytest integration, 50+ metrics, Confident AI platform (SaaS)
Arbiter: PydanticAI-native, cost-first, pure library (no platform)

Use DeepEval if: You need pytest CI/CD integration and comprehensive metrics
Use Arbiter if: You use PydanticAI and want cost transparency without platform lock-in
```

### TruLens / Phoenix
```
TruLens/Phoenix: Enterprise observability platforms with dashboards
Arbiter: Lightweight library, automatic tracking, no server required

Use TruLens/Phoenix if: You need enterprise features and visualization dashboards
Use Arbiter if: You want a simple library that shows costs and tracks interactions
```

### Key Differentiators

**1. Cost Transparency** (Unique)
- Arbiter: Automatic, prominent, real pricing data
- Others: Hidden, require platform, or not tracked at all

**2. PydanticAI Native**
- Arbiter: Built on PydanticAI, same patterns
- Others: Generic (support everything = great at nothing)

**3. Pure Library**
- Arbiter: `pip install`, no signup, no server
- Others: Open-core + SaaS platforms

---

## GitHub Issues Needed!

**What would make Arbiter more useful for you?**

**Evaluator Ideas:**
- Legal compliance checker? Code quality evaluator?
- Customer service tone analyzer? Medical accuracy validator?

**Storage Backends** (currently in-memory only):
- PostgreSQL for persistent history? Redis for distributed caching? S3 for archival?

**Real-time Monitoring:**
- Datadog dashboards? Prometheus metrics? Grafana? Custom webhooks?

**Streaming Support:**
- Partial evaluation as tokens arrive? Progressive scoring? Real-time quality gates?

**Your Pain Points:**
- What evaluation problem keeps you up at night?

https://github.com/evanvolgas/arbiter/issues

---

## Closing

**Key Points:**
1. **PydanticAI Native** - Same patterns, same type safety
2. **Cost Tracking** - Real-time with live pricing data
3. **Library, Not Platform** - No server, no signup
4. **Automatic Tracking** - Every LLM call tracked automatically

**How You Can Help:**
- REVIEW THIS PR: https://github.com/evanvolgas/arbiter/pull/7
- Open issues with evaluator ideas (see section above)
- Contribute domain-specific evaluators
- Star the repo: github.com/evanvolgas/arbiter

v0.1.0-alpha - looking for early adopters and feedback!


## Q&A

### Expected Questions

**Q: "How is this different from LangSmith/Langfuse?"**
```
A: "They require explicit instrumentation - decorators like @traceable on every
function. Arbiter uses inheritance - if you use our evaluators, tracking is
built-in. No decorators, no manual spans."

[Show code comparison: LangSmith @traceable vs Arbiter's base.py]
```

**Q: "Can I use my own evaluators?"**
```
A: "Yes - template method pattern. 4 methods:
1. name property
2. _get_system_prompt()
3. _get_user_prompt()
4. _compute_score()

Inherit from BasePydanticEvaluator, you get automatic tracking for free."

[Show custom_criteria.py as example]
```

**Q: "What about cost accuracy?"**
```
A: "We use LiteLLM's bundled pricing database - same source as Conduit.
Updated pricing via 'uv update litellm', not hardcoded estimates. Works across all providers."

[Show cost_calculator.py if needed]
```

**Q: "Does this work with streaming?"**
```
A: "Not yet - that's issue #1 we need from you! How should we handle partial
evaluations? Progressive scoring? This is where community input helps."
```

**Q: "How do you handle rate limits?"**
```
A: "Circuit breaker pattern in llm_client.py. Automatic backoff and retry.
Also middleware for custom rate limiting - see middleware.py:462"

[Show if needed: RateLimitingMiddleware]
```

**Q: "Can I evaluate without reference text?"**
```
A: "Yes - CustomCriteriaEvaluator does this. Reference-free evaluation against
criteria strings. Useful for brand voice, tone, compliance checks."

[Show custom_criteria_example.py]
```

**Q: "What's the performance overhead?"**
```
A: "Minimal - tracking is ~5-10ms per call. The LLM call itself is 500ms-2s.
For batch operations, we parallelize automatically (max_concurrency parameter)."
```

**Q: "Can I use this in production?"**
```
A: "Maybe - 95% test coverage, strict mypy typing, production-grade error handling. But it's v0.1.0-alpha, so report issues! I want your feedback!"
```




