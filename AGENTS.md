# AGENTS.md - Project Context (Layer 2)

**Purpose:** How AI agents should work with the Arbiter repository
**Type:** Project Context (Layer 2 of 4-layer context framework)
**Last Updated:** 2025-11-12

---

## Four-Layer Context Framework

This repository uses a four-layer context system for AI agent interactions:

### Layer 1: Global Context (~/.claude/CLAUDE.md)
**What:** Universal rules applying across ALL projects
**Contains:**
- Communication preferences
- General expertise areas
- Things AI should never do
- Known gotchas

**Update Frequency:** Rarely (monthly or less)

### Layer 2: Project Context (THIS FILE - AGENTS.md)
**What:** Repository-specific rules and architecture
**Contains:**
- Tech stack and architecture
- Development workflow
- Critical constraints
- Code ownership
- Repository structure

**Update Frequency:** When architecture changes (monthly)

### Layer 3: Running Context (PROJECT_TODO.md)
**What:** Current session state and active tasks
**Contains:**
- Current milestone tasks
- In-progress items with checkboxes
- Decisions made during work
- Blockers and notes

**Update Frequency:** Daily/weekly during active development

### Layer 4: Prompt Context
**What:** The immediate, specific request
**Contains:**
- Single-use instruction
- Builds on layers 1-3

**Update Frequency:** Every interaction

### How They Work Together

Each layer constrains the next:
```
Global → Project → Running → Prompt
  ↓        ↓         ↓         ↓
Rules → Architecture → Status → Request
```

**Example:**
- **Global:** "Never use placeholders or TODO comments"
- **Project:** "Use template method pattern for evaluators"
- **Running:** "Phase 2.5: 60% complete - Registry system done, documentation next"
- **Prompt:** "Create batch evaluation example"

---

## Repository Architecture

### Directory Structure

```
arbiter/
├── arbiter/                    # Main package
│   ├── __init__.py             # Public API exports
│   ├── api.py                  # High-level evaluate() function
│   ├── core/                   # Core infrastructure
│   │   ├── __init__.py
│   │   ├── exceptions.py       # 8 custom exception types
│   │   ├── interfaces.py       # BaseEvaluator, StorageBackend protocols
│   │   ├── llm_client.py       # Provider-agnostic LLM client
│   │   ├── llm_client_pool.py  # Connection pooling
│   │   ├── middleware.py       # Logging, metrics, caching, rate limiting
│   │   ├── models.py           # Pydantic data models
│   │   ├── monitoring.py       # Performance metrics
│   │   ├── retry.py            # Exponential backoff retry
│   │   ├── registry.py         # Evaluator registry system
│   │   ├── type_defs.py        # TypedDict definitions
│   │   └── types.py            # Provider, MetricType, StorageType, EvaluatorName enums
│   ├── evaluators/             # Evaluation implementations
│   │   ├── __init__.py
│   │   ├── base.py             # BasePydanticEvaluator template
│   │   ├── semantic.py         # SemanticEvaluator (implemented)
│   │   ├── custom_criteria.py  # CustomCriteriaEvaluator (implemented)
│   │   └── pairwise.py         # PairwiseComparisonEvaluator (implemented)
│   ├── storage/                # Storage backends (Phase 4)
│   │   └── __init__.py
│   └── tools/                  # Utilities
│       └── __init__.py
├── examples/
│   ├── basic_evaluation.py           # Basic semantic evaluation
│   ├── custom_criteria_example.py   # Domain-specific criteria
│   ├── pairwise_comparison_example.py # A/B testing
│   ├── multiple_evaluators.py       # Combining evaluators
│   ├── middleware_usage.py          # Logging, metrics, caching
│   ├── error_handling_example.py    # Partial results & errors
│   ├── provider_switching.py        # Multi-provider support
│   └── evaluator_registry_example.py # Custom evaluator registration
├── tests/
│   ├── unit/
│   └── integration/
├── docs/                       # Documentation
├── benchmarks/                 # Performance testing
├── pyproject.toml              # Project configuration
├── README.md                   # Project overview
├── DESIGN_SPEC.md              # What we're building (Layer 1 reference)
├── AGENTS.md                   # THIS FILE (Layer 2)
├── PROJECT_PLAN.md             # Multi-milestone roadmap (Layer 2.5)
├── PROJECT_TODO.md             # Current milestone (Layer 3)
├── PHASE2_REVIEW.md            # Phase 2 assessment
├── EVALUATOR_RECOMMENDATIONS.md # Evaluator priorities
└── CONTRIBUTING.md             # Contribution guidelines
```

### Module Responsibilities

#### `api.py` - Public API
**Purpose:** Simple entry point for users
**Key Functions:**
- `evaluate()` - Main evaluation function (supports multiple evaluators)
- `compare()` - Pairwise comparison function
- Auto-manages LLM clients
- Integrates middleware pipeline
- Uses evaluator registry for validation

#### `core/` - Infrastructure
**Purpose:** Production-grade foundation
**Components:**
- **llm_client.py:** Provider-agnostic LLM interface (OpenAI, Anthropic, Google, Groq)
- **llm_client_pool.py:** Connection pooling with health checks
- **middleware.py:** Logging, metrics, caching, rate limiting
- **models.py:** EvaluationResult, ComparisonResult, Score, Metric, LLMInteraction
- **registry.py:** Evaluator registry system (AVAILABLE_EVALUATORS, register_evaluator)
- **exceptions.py:** ArbiterError hierarchy (8 types)
- **retry.py:** Exponential backoff with 3 presets
- **monitoring.py:** PerformanceMetrics, PerformanceMonitor

#### `evaluators/` - Evaluation Logic
**Purpose:** Evaluator implementations
**Pattern:** Template Method (BasePydanticEvaluator)
**Current:**
- SemanticEvaluator ✅ - Semantic similarity evaluation
- CustomCriteriaEvaluator ✅ - Domain-specific criteria (single & multi-criteria)
- PairwiseComparisonEvaluator ✅ - A/B testing and model comparison
**Planned:**
- FactualityEvaluator (Phase 5)
- RelevanceEvaluator (Phase 5)
- ToxicityEvaluator (Phase 5)
- GroundednessEvaluator (Phase 5)
- ConsistencyEvaluator (Phase 5)

---

## Tech Stack

### Core Technologies
- **Python:** 3.10+ (required for modern type hints)
- **Pydantic:** 2.12+ (data validation and serialization)
- **PydanticAI:** 1.14+ (structured LLM outputs)
- **Pymilvus:** 2.6+ (vector database for Phase 3)
- **HTTPX:** 0.28+ (async HTTP client)

### LLM Provider SDKs
- **OpenAI:** 2.0+ (GPT-4o, GPT-4, GPT-3.5-turbo)
- **Anthropic:** 0.72+ (Claude 3.5 Sonnet, Claude 3)
- **Google:** 0.8.5+ (Gemini 1.5 Pro/Flash)
- **Groq:** Latest (Llama 3.1, Mixtral)
- **Mistral:** 1.0+
- **Cohere:** 5.0+

### Development Tools
- **pytest:** 9.0+ (testing framework)
- **pytest-asyncio:** 1.0+ (async test support)
- **black:** 25.0+ (code formatting)
- **ruff:** 0.14+ (linting)
- **mypy:** 1.18+ (type checking - strict mode)

### Future Dependencies
- **ByteWax:** 0.20+ (streaming support - Phase 7)
- **Redis:** 5.0+ (storage backend - Phase 4)

---

## Development Workflow

### 1. Feature Development

#### Branch Strategy
```bash
# Always work on feature branches
git checkout -b feature/custom-criteria-evaluator

# Never commit to main directly
# Use PRs for all changes
```

#### Implementation Flow
1. **Read Context:**
   - DESIGN_SPEC.md (understand vision)
   - PROJECT_TODO.md (check current milestone)
   - Relevant evaluator code (understand patterns)

2. **Plan:**
   - What are we building?
   - What's the template method structure?
   - What tests are needed?

3. **Implement:**
   - Follow template method pattern
   - Add type hints (strict mypy)
   - Write tests (>80% coverage)
   - Update __init__.py exports

4. **Validate:**
   - Run tests: `make test`
   - Check types: `make type-check`
   - Lint code: `make lint`
   - Format: `make format`

5. **Document:**
   - Add docstrings
   - Update examples/
   - Update PROJECT_TODO.md

### 2. Testing Requirements

**Minimum Coverage:** 80% (strict)

**Test Types:**
1. **Unit Tests** (tests/unit/)
   - Test individual functions and classes
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (tests/integration/)
   - Test end-to-end flows
   - Use real LLM calls (with mocking option)
   - Slower but comprehensive

3. **Performance Tests** (benchmarks/)
   - Latency benchmarks
   - Memory usage
   - Concurrent processing

**Running Tests:**
```bash
# All tests with coverage
make test

# Specific test file
pytest tests/unit/test_semantic.py

# With coverage report
make test-cov

# Fast (unit only)
pytest tests/unit/
```

### 3. Code Quality Standards

#### Type Safety (CRITICAL)
```toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

**Rules:**
- ✅ All functions must have type hints
- ✅ All parameters must be typed
- ✅ Return types must be explicit
- ❌ No `Any` without explicit justification

**Example:**
```python
# ✅ GOOD
async def evaluate(
    output: str,
    reference: Optional[str] = None,
    evaluators: List[str] = ["semantic"]
) -> EvaluationResult:
    ...

# ❌ BAD
async def evaluate(output, reference=None):  # Missing types
    ...
```

#### Code Formatting
- **black:** Line length 88 (default)
- **ruff:** Follow configuration in pyproject.toml
- Run `make format` before committing

#### Docstrings
```python
async def evaluate(
    output: str,
    reference: Optional[str] = None
) -> EvaluationResult:
    """Evaluate LLM output against reference or criteria.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison

    Returns:
        EvaluationResult with scores, metrics, and interactions

    Raises:
        ValidationError: If output is empty
        EvaluatorError: If evaluation fails

    Example:
        >>> result = await evaluate(
        ...     output="Paris is the capital",
        ...     reference="Paris is the capital of France"
        ... )
        >>> print(result.overall_score)
        0.92
    """
```

---

## Critical Constraints

### 1. Provider-Agnostic Design (NON-NEGOTIABLE)

**Rule:** Must work with ANY LLM provider

**Implementation:**
- Use `LLMClient` abstraction
- Test with multiple providers
- No OpenAI-specific assumptions

**Example:**
```python
# ✅ GOOD
client = await LLMManager.get_client(
    provider="anthropic",  # or "openai", "google", etc.
    model="claude-3-5-sonnet"
)

# ❌ BAD
from openai import OpenAI
client = OpenAI()  # Hardcoded to OpenAI
```

### 2. PydanticAI for Structured Outputs (REQUIRED)

**Rule:** All evaluators must use PydanticAI for structured responses

**Rationale:**
- Type safety
- Automatic validation
- Provider abstraction

**Pattern:**
```python
class MyResponse(BaseModel):
    score: float
    explanation: str

class MyEvaluator(BasePydanticEvaluator):
    def _get_response_type(self):
        return MyResponse  # PydanticAI uses this
```

### 3. Automatic Interaction Tracking (CORE FEATURE)

**Rule:** Every LLM call must be tracked automatically

**Implementation:** Done in `BasePydanticEvaluator.evaluate()`

**Never:** Bypass interaction tracking
**Always:** Use `self.interactions` list

### 4. Template Method Pattern for Evaluators (ENFORCED)

**Rule:** All evaluators extend `BasePydanticEvaluator` and implement 4 methods

**Required Methods:**
1. `_get_system_prompt() -> str`
2. `_get_user_prompt(output, reference, criteria) -> str`
3. `_get_response_type() -> Type[BaseModel]`
4. `async _compute_score(response) -> Score`

**Example:**
```python
class CustomCriteriaEvaluator(BasePydanticEvaluator):
    @property
    def name(self) -> str:
        return "custom_criteria"

    def _get_system_prompt(self) -> str:
        return "You are an expert evaluator..."

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        return f"Evaluate '{output}' against: {criteria}"

    def _get_response_type(self) -> Type[BaseModel]:
        return CustomCriteriaResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(CustomCriteriaResponse, response)
        return Score(
            name=self.name,
            value=resp.score,
            confidence=resp.confidence,
            explanation=resp.explanation
        )
```

### 5. No Placeholders or TODOs (STRICT)

**Rule:** Never leave `TODO`, `FIXME`, or placeholder implementations

**Why:** Production-grade code, not scaffolding

**Example:**
```python
# ❌ NEVER DO THIS
def calculate_cost(tokens: int) -> float:
    # TODO: implement cost calculation
    raise NotImplementedError("Coming soon")

# ✅ DO THIS
def calculate_cost(tokens: int, cost_per_1k: float = 0.01) -> float:
    """Calculate cost based on token usage."""
    return (tokens / 1000) * cost_per_1k
```

### 6. Complete Features Only (NO PARTIAL WORK)

**Rule:** If you start a feature, complete it fully

**Complete means:**
- ✅ Implementation finished
- ✅ Tests written (>80% coverage)
- ✅ Docstrings added
- ✅ Example code provided
- ✅ Exported in __init__.py

---

## Code Ownership

### Core Infrastructure (arbiter/core/)
**Owner:** Architecture decisions
**Review:** Required for changes
**Critical:** Affects all evaluators

### Evaluators (arbiter/evaluators/)
**Owner:** Evaluator-specific logic
**Review:** Recommended
**Extensible:** Easy to add new evaluators

### API (arbiter/api.py)
**Owner:** User-facing interface
**Review:** Required for changes
**Stability:** High - breaking changes avoided

### Documentation
**Owner:** Everyone contributes
**Review:** Recommended
**Quality:** Clear, concise, with examples

---

## Integration with Loom

### Relationship to Loom

**Arbiter** is a direct dependency of **Loom** (the AI pipeline orchestrator):
- **Loom**: Orchestrates AI(E)TL pipelines (Extract → Transform → Evaluate → Load)
- **Arbiter**: Provides evaluation engine used in Loom's Evaluate stage
- **Relationship**: Loom imports and uses Arbiter evaluators in quality gates

**Directory Locations:**
- Arbiter: `/Users/evan/Documents/gh/arbiter`
- Loom: `/Users/evan/Documents/gh/loom`

### How Loom Uses Arbiter

Loom pipelines define evaluation stages that use Arbiter evaluators:

```yaml
# Loom pipeline example
name: customer_qa_generation
version: 1.0.0

extract:
  source: postgres://customers/questions

transform:
  type: ai
  prompt: prompts/answer_question.txt
  model: gpt-4o-mini

evaluate:
  evaluators:
    # Uses Arbiter's SemanticEvaluator
    - name: semantic_check
      type: semantic
      threshold: 0.8

    # Uses Arbiter's FactualityEvaluator (enhanced with plugins!)
    - name: factuality_check
      type: factuality
      config:
        plugins:
          - name: tavily
            max_results: 5
        use_cache: true
        threshold: 0.85

    # Uses Arbiter's CustomCriteriaEvaluator
    - name: quality_check
      type: custom_criteria
      criteria: "Helpful, accurate, no hallucination"
      threshold: 0.75

  quality_gate: all_pass  # Defined in Loom's QUALITY_GATES.md

load:
  destination: postgres://customer_service/qa_responses
  on_failure: quarantine  # Quarantine hallucinated responses
```

### Impact of Arbiter Enhancements on Loom

**FactualityEvaluator Plugin Enhancement:**
- **Before**: Loom quality gates use pure LLM-based factuality checking (70-80% accuracy)
- **After**: Loom quality gates use external verification with citations (90-98% accuracy)
- **Benefit**: Loom pipelines can prevent hallucinated responses from reaching production

**Example Loom Evaluation Result:**
```python
{
    "record_id": "cust_q_12345",
    "question": "How tall is the Eiffel Tower?",
    "answer": "The Eiffel Tower is 300 meters tall (324 meters including antennas).",

    "evaluation_results": {
        "semantic_check": {"passed": True, "score": 0.95},
        "factuality_check": {
            "passed": True,
            "score": 1.0,
            "verified_claims": [
                "The Eiffel Tower is 300 meters tall",
                "324 meters including antennas"
            ],
            "sources": [  # ← Citations from Tavily plugin!
                {
                    "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                    "title": "Eiffel Tower - Wikipedia",
                    "snippet": "The tower is 300 metres (984 ft) tall..."
                }
            ]
        },
        "quality_check": {"passed": True, "score": 0.88}
    },

    "quality_gate": "PASSED",
    "loaded_to": "postgres://customer_service/qa_responses"
}
```

### Development Coordination

When developing Arbiter evaluators, consider Loom integration:

1. **Evaluator Names**: Must match Loom pipeline YAML evaluator types
2. **Configuration Schema**: Loom passes config dict to evaluators, ensure validation
3. **Performance**: Loom processes batches, optimize for throughput
4. **Error Handling**: Loom expects graceful degradation, not hard failures
5. **Metadata**: Include rich metadata in Score objects for Loom audit trails

### Testing with Loom

**Unit Tests**: Test evaluators independently (in Arbiter)
**Integration Tests**: Test Loom pipelines using Arbiter evaluators (in Loom)

**Example Test Flow:**
1. Arbiter: Test FactualityEvaluator with plugins → arbiter/tests/
2. Loom: Test quality gates using FactualityEvaluator → loom/tests/
3. E2E: Test complete Loom pipeline with real Arbiter evaluations

---

## Agent Guidelines

### When Starting Work

1. **Read DESIGN_SPEC.md** - Understand vision and architecture
2. **Read PROJECT_TODO.md** - Check current milestone and tasks
3. **Check git status** - Ensure clean working directory
4. **Create feature branch** - `git checkout -b feature/your-feature`

### During Development

1. **Follow template method pattern** for evaluators
2. **Write tests as you code** (not after)
3. **Run tests frequently** (`make test`)
4. **Update PROJECT_TODO.md** with progress
5. **Document decisions** in commit messages

### Before Committing

1. **Run full test suite:** `make test`
2. **Check type safety:** `make type-check`
3. **Lint code:** `make lint`
4. **Format code:** `make format`
5. **Update __init__.py** exports if needed
6. **Update PROJECT_TODO.md** checkbox

### After Completing Feature

1. **Mark task complete** in PROJECT_TODO.md
2. **Add example** to examples/ if user-facing
3. **Update README.md** if API changed
4. **Create PR** with clear description
5. **Link to issue/task** in PR description

---

## Common Patterns

### Pattern 1: Adding a New Evaluator

```python
# 1. Create evaluator file
# arbiter/evaluators/my_evaluator.py

from typing import Type
from pydantic import BaseModel, Field
from .base import BasePydanticEvaluator
from arbiter.core.models import Score

class MyEvaluatorResponse(BaseModel):
    """Structured response for MyEvaluator."""
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    explanation: str

class MyEvaluator(BasePydanticEvaluator):
    """Evaluates outputs based on custom criteria."""

    @property
    def name(self) -> str:
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        return "You are an expert evaluator..."

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        return f"Evaluate: {output}"

    def _get_response_type(self) -> Type[BaseModel]:
        return MyEvaluatorResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(MyEvaluatorResponse, response)
        return Score(
            name=self.name,
            value=resp.score,
            confidence=resp.confidence,
            explanation=resp.explanation
        )

# 2. Export in __init__.py
# arbiter/evaluators/__init__.py
from .my_evaluator import MyEvaluator
__all__ = [..., "MyEvaluator"]

# 3. Update main __init__.py
# arbiter/__init__.py
from .evaluators import MyEvaluator
__all__ = [..., "MyEvaluator"]

# 4. Write tests
# tests/unit/test_my_evaluator.py
import pytest
from arbiter.evaluators import MyEvaluator

@pytest.mark.asyncio
async def test_my_evaluator():
    evaluator = MyEvaluator(model="gpt-4o-mini")
    score = await evaluator.evaluate(
        output="test output",
        reference="test reference"
    )
    assert 0.0 <= score.value <= 1.0

# 5. Add example
# examples/my_evaluator_example.py
```

### Pattern 2: Adding Middleware

```python
# arbiter/core/middleware.py

class MyMiddleware(Middleware):
    """Custom middleware for [purpose]."""

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable,
        context: Dict[str, Any]
    ) -> EvaluationResult:
        # Pre-processing
        start = time.time()

        # Call next in chain
        result = await next_handler(output, reference)

        # Post-processing
        elapsed = time.time() - start
        logger.info(f"Evaluation took {elapsed:.2f}s")

        return result

# Usage
pipeline = MiddlewarePipeline([
    LoggingMiddleware(),
    MyMiddleware(),
    MetricsMiddleware()
])
```

---

## Troubleshooting

### Issue: Type Errors

**Problem:** `mypy` reports type errors

**Solution:**
1. Run `mypy arbiter/` to see errors
2. Add missing type hints
3. Use `cast()` for runtime type narrowing
4. Check pyproject.toml mypy config

### Issue: Import Errors

**Problem:** `ImportError` or circular imports

**Solution:**
1. Check __init__.py exports
2. Use `if TYPE_CHECKING:` for type-only imports
3. Avoid circular dependencies

### Issue: Tests Failing

**Problem:** Tests fail unexpectedly

**Solution:**
1. Run `pytest -v` for verbose output
2. Check if you're using async/await correctly
3. Verify mocks are set up properly
4. Ensure test isolation (no shared state)

### Issue: Middleware Not Working

**Problem:** Middleware not being called

**Solution:**
1. Check if middleware is in MiddlewarePipeline
2. Verify `await next_handler()` is called
3. Ensure pipeline.execute() is used in api.py

---

## Versioning & Releases

### Current Version: 0.1.0 (Alpha)

**Semantic Versioning:** MAJOR.MINOR.PATCH

- **MAJOR:** Breaking API changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

### Release Checklist

Before releasing:
- [ ] All tests pass (`make test`)
- [ ] Type checking clean (`make type-check`)
- [ ] Linting clean (`make lint`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Git tag created

---

## Related Documents

### Essential Reading (Priority Order)
1. **DESIGN_SPEC.md** - What we're building and why
2. **PROJECT_TODO.md** - Current milestone tasks
3. **AGENTS.md** - THIS FILE (how to work here)
4. **CONTRIBUTING.md** - Contribution workflow

### Reference Documents
- **PROJECT_PLAN.md** - Multi-milestone roadmap
- **PHASE2_REVIEW.md** - Phase 2 deep analysis
- **EVALUATOR_RECOMMENDATIONS.md** - Evaluator priorities
- **README.md** - User-facing overview

### External References
- [PydanticAI Docs](https://ai.pydantic.dev/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [Milvus Docs](https://milvus.io/docs)

---

## Questions?

If you're unsure about:
- **Architecture:** Read DESIGN_SPEC.md
- **Current work:** Check PROJECT_TODO.md
- **Evaluator pattern:** Look at evaluators/semantic.py
- **Testing:** See tests/unit/ for examples

**Still unclear?** Open an issue or ask in the PR.

---

**Last Updated:** 2025-11-12 | **Next Review:** 2025-12-12
