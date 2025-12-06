---
name: arbiter-agent
description: Production-grade LLM evaluation framework developer
last_updated: 2025-11-29
---

# Arbiter Agent Guide

**Purpose**: Quick reference for working on Arbiter

**Arbiter**: Production-grade LLM evaluation framework (v0.1.2)
**Stack**: Python 3.11+, PydanticAI, provider-agnostic (OpenAI/Anthropic/Google/Groq)
**Coverage**: 96% test coverage, strict mypy, comprehensive examples
**Pricing**: LiteLLM bundled database (consistent with Conduit)

**Design Philosophy**: Simplicity wins, use good defaults, YAML config where needed, no hardcoded assumptions.

---

## Quick Start (First Session Commands)

**New to this repo? Run these 5 commands first:**

```bash
# 1. Verify you're on a feature branch (NEVER work on main)
git status && git branch

# 2. Run all quality checks
make all

# 3. Run specific evaluator test to verify environment
pytest tests/unit/test_semantic.py -v

# 4. Check for any TODOs or placeholders (should be NONE)
grep -r "TODO\|FIXME\|NotImplementedError" arbiter/ || echo "✅ No placeholders found"

# 5. Verify coverage is >80%
make test-cov | tail -1
```

---

## Boundaries

### Always Do (No Permission Needed)

- Run tests: `make test`, `pytest tests/`, `pytest -v`
- Format code: `make format` (runs black)
- Lint code: `make lint` (runs ruff)
- Type check: `make type-check` (runs mypy in strict mode)
- Add unit tests for new evaluators in `tests/unit/`
- Update docstrings when changing function signatures
- Add examples to `examples/` for new user-facing features
- Export new evaluators in `__init__.py` files
- Run `make all` before committing (format + lint + type-check + test)

### Ask First

**Core Architecture** (Why: Breaks all evaluators):
- Add new evaluators to `arbiter/evaluators/` - Must follow template pattern
- Modify core API in `arbiter/api.py` (evaluate, compare functions) - Breaking change for all users
- Change template method pattern in `BasePydanticEvaluator` - All evaluators inherit from this
- Modify middleware pipeline in `arbiter/core/middleware.py` - Affects all evaluations
- Change LLM client abstraction in `arbiter/core/llm_client.py` - Provider-agnostic guarantee at risk

**Dependencies & Config** (Why: Security and maintenance burden):
- Add/update dependencies in `pyproject.toml` - Increases attack surface
- Change public API examples in `README.md` - User-facing documentation
- Add new storage backends in `arbiter/storage/` - Data persistence implications

**Monitoring & Observability** (Why: Production debugging):
- Modify interaction tracking in `arbiter/core/monitoring.py` - Breaks observability

### Never Touch

**Security (CRITICAL)**:
- NEVER EVER COMMIT CREDENTIALS TO GITHUB
- No API keys, tokens, passwords, secrets in ANY file
- No credentials in code, documentation, examples, tests, or configuration files
- Use environment variables (.env files in .gitignore) ONLY
- This is non-negotiable with serious security consequences

**Other Prohibitions**:
- `.env` files or API keys (use environment variables)
- Production deployment configurations
- Git history manipulation (no force push, interactive rebase on shared branches)
- User's `~/.claude/` configuration files
- Any files outside the `arbiter/` repository
- Test files to make them pass (fix the code, not the tests)
- Type checking configuration to reduce strictness
- Coverage thresholds (must maintain >80%)

**Detection Commands** (Run before committing):
```bash
# Check for security violations
grep -r "API_KEY\|SECRET\|PASSWORD" arbiter/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" || echo "✅ No credentials"

# Check for code quality violations
grep -r "TODO\|FIXME" arbiter/ && echo "🚨 TODO comments found" || echo "✅ No TODOs"

# Check for incomplete features
grep -r "NotImplementedError\|pass  # TODO" arbiter/ && echo "🚨 Placeholder code found" || echo "✅ No placeholders"

# Verify on feature branch
git branch --show-current | grep -E "^(main|master)$" && echo "🚨 ON MAIN BRANCH - CREATE FEATURE BRANCH" || echo "✅ On feature branch"

# Verify coverage >80%
make test 2>&1 | grep "TOTAL" | awk '{if ($NF+0 < 80) print "🚨 COVERAGE " $NF " < 80%"; else print "✅ Coverage " $NF}'
```

---

## Communication Preferences

Don't flatter me. I know what [AI sycophancy](https://www.seangoedecke.com/ai-sycophancy/) is and I don't want your praise. Be concise and direct. Don't use emdashes ever.

---

## Session Analysis & Continuous Improvement

**When to Analyze** (Multiple Triggers):
- During active sessions: After completing major tasks or every 30-60 minutes
- When failures occur: Immediately analyze and update rules
- Session end: Review entire session for patterns before closing
- User corrections: Any time user points out a mistake

**Identify Failures**:
- Framework violations (boundaries crossed, rules ignored)
- Repeated patterns (same mistake multiple times)
- Rules that didn't prevent failures
- User corrections (what needed fixing)

**Analyze Each Failure**:
- What rule should have prevented this?
- Why didn't it work? (too vague, wrong priority, missing detection pattern)
- What would have caught this earlier?

**Update AGENTS.md** (In Real-Time):
- Add new rules or strengthen existing rules immediately
- Add detection patterns (git commands, test patterns, code patterns)
- Include examples of violations and corrections
- Update priority if rule was underweighted
- Propose updates to user during session (don't wait until end)

**Priority Levels**:
- 🔴 **CRITICAL**: Security, credentials, production breaks → Update immediately, stop work
- 🟡 **IMPORTANT**: Framework violations, repeated patterns → Update with detection patterns, continue work
- 🟢 **RECOMMENDED**: Code quality, style issues → Update with examples, lowest priority

**Example Pattern**:
```
Failure: Committed TODO comments in production code (violated "No Partial Features" rule)
Detection: `grep -r "TODO" src/` before commit
Rule Update: Add pre-commit check pattern to Boundaries section
Priority: 🟡 IMPORTANT
Action Taken: Proposed rule update to user mid-session, updated AGENTS.md
```

**Proactive Analysis**:
- Before risky operations: Check if existing rules cover this scenario
- After 3+ similar operations: Look for pattern that should be codified
- When uncertainty arises: Document the decision-making gap

---

## Directory Structure

```
arbiter/
├── arbiter_ai/
│   ├── api.py              # Public API (evaluate, compare)
│   ├── core/               # Infrastructure (llm_client, middleware, monitoring, registry, cost_calculator)
│   ├── evaluators/         # Semantic, CustomCriteria, Pairwise, Factuality, Groundedness, Relevance
│   ├── storage/            # Storage backends (PostgreSQL, Redis)
│   └── verifiers/          # Claim verification (Search, Citation, KnowledgeBase)
├── examples/               # 25+ comprehensive examples
├── tests/                  # Unit + integration tests (583 tests, 96% coverage)
└── pyproject.toml          # Dependencies and config
```

### Cost Calculator

The cost calculator uses LiteLLM's bundled pricing database (same source as Conduit):

```python
from arbiter_ai import get_cost_calculator

calc = get_cost_calculator()
cost = calc.calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
print(f"Cost: ${cost:.6f}")  # Cost: $0.000450
```

To update pricing: `uv update litellm`

---

## Critical Rules

### 1. Template Method Pattern

All evaluators extend `BasePydanticEvaluator` and implement 4 methods:

```python
class MyEvaluator(BasePydanticEvaluator):
    @property
    def name(self) -> str:
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        return "You are an expert evaluator..."

    def _get_user_prompt(self, output: str, reference: Optional[str], criteria: Optional[str]) -> str:
        return f"Evaluate '{output}' against: {criteria}"

    def _get_response_type(self) -> Type[BaseModel]:
        return MyEvaluatorResponse  # Pydantic model

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(MyEvaluatorResponse, response)
        return Score(name=self.name, value=resp.score, confidence=resp.confidence, explanation=resp.explanation)
```

### 2. Provider-Agnostic Design

Must work with ANY LLM provider (OpenAI, Anthropic, Google, Groq, Mistral, Cohere).

```python
# GOOD
client = await LLMManager.get_client(provider="anthropic", model="claude-3-5-sonnet")

# BAD
from openai import OpenAI
client = OpenAI()  # Hardcoded to OpenAI
```

### 3. Type Safety (Strict Mypy)

All functions require type hints, no `Any` without justification.

### 4. No Placeholders/TODOs

Production-grade code only. Complete implementations or nothing.

### 5. Complete Features Only

If you start, you finish:
- Implementation complete
- Tests (>80% coverage)
- Docstrings
- Example code
- Exported in `__init__.py`

### 6. PydanticAI for Structured Outputs

All evaluators use PydanticAI for type-safe LLM responses.

### 7. Pydantic v2 Best Practices

All Pydantic models follow strict validation patterns:

**ConfigDict (Required for ALL models)**:
```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class MyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",              # Reject unknown fields (no backward compatibility)
        str_strip_whitespace=True,   # Auto-strip whitespace from strings
        validate_assignment=True,    # Validate on attribute changes
    )
```

**Computed Fields** (for serializable properties):
```python
from pydantic import computed_field

@computed_field  # type: ignore[prop-decorator]
@property
def total_tokens(self) -> int:
    """Total tokens across input and output."""
    return self.input_tokens + self.output_tokens
```

**Field Validators** (validate individual fields):
```python
@field_validator("explanation")
@classmethod
def validate_explanation_quality(cls, v: str) -> str:
    """Ensure explanation is meaningful and not empty."""
    if len(v.strip()) < 1:
        raise ValueError("Explanation cannot be empty")
    return v.strip()
```

**Model Validators** (cross-field validation):
```python
@model_validator(mode="after")
def validate_high_confidence_scores(self) -> "MyModel":
    """Ensure high-confidence scores have supporting details."""
    if self.confidence > 0.9:
        if not self.supporting_details:
            raise ValueError(
                "High confidence scores (>0.9) require supporting details"
            )
    return self
```

**Testing with Validators**:
- Update test data to pass validators, never relax validators to pass tests
- Use realistic test data (non-empty explanations, supporting details for confident scores)
- When validators fail, fix the test data not the validator

---

## Common Mistakes & How to Avoid Them

### Mistake 1: Breaking Template Method Pattern
**Detection**: New evaluator doesn't implement all 4 required methods
**Prevention**: Copy existing evaluator (semantic.py) as template
**Fix**: Implement `name`, `_get_system_prompt`, `_get_user_prompt`, `_get_response_type`, `_compute_score`
**Why It Matters**: Template pattern ensures all evaluators work consistently

### Mistake 2: Hardcoding LLM Provider
**Detection**: `from openai import OpenAI` in evaluator code
**Prevention**: Use `LLMManager.get_client()` for provider abstraction
**Fix**: Replace direct provider imports with LLMManager
**Why It Matters**: Provider-agnostic design is core feature

### Mistake 3: Not Exporting New Evaluators
**Detection**: New evaluator not importable from `arbiter`
**Prevention**: Add to `__init__.py` exports in both `evaluators/` and root
**Fix**: Add `from .my_evaluator import MyEvaluator` and update `__all__`
**Why It Matters**: Users can't use evaluator if not exported

### Mistake 4: Missing Docstrings
**Detection**: Functions without Args/Returns/Example sections
**Prevention**: Write docstring before implementation
**Fix**: Add complete docstring with all sections
**Why It Matters**: Docstrings are user documentation

### Mistake 5: Using `Any` Type
**Detection**: `grep -r "from typing import Any" arbiter/`
**Prevention**: Use specific Pydantic models for type safety
**Fix**: Create Pydantic model for response structure
**Why It Matters**: Type safety prevents bugs

### Mistake 6: Skipping Examples
**Detection**: New evaluator without example in `examples/`
**Prevention**: Create example file showing usage
**Fix**: Add `examples/my_evaluator_example.py`
**Why It Matters**: Examples are how users learn API

### Mistake 7: Low Test Coverage
**Detection**: `make test` shows coverage <80%
**Prevention**: Write tests as you code
**Fix**: Add unit tests until coverage >80%
**Why It Matters**: Untested evaluators will break

### Mistake 8: Relaxing Validators to Pass Tests
**Detection**: ValidationError in tests, changing `extra="forbid"` to `extra="ignore"` or removing validators
**Prevention**: Write realistic test data that would pass production validation
**Fix**: Update test fixtures with valid data (non-empty explanations, supporting details for high confidence)
**Why It Matters**: Validators enforce data quality - weakening them hides bugs
**Rule**: Fix test data, never fix validators

---

## Testing Decision Matrix

**When to Mock:**
- LLM API calls (OpenAI, Anthropic) - Use mocked responses to avoid costs
- Network requests - Use mocked HTTP responses
- File I/O for storage backends - Use temporary directories
- Time-dependent code - Mock `datetime.now()`

**When to Use Real Dependencies:**
- Pydantic validation - Real validation catches schema bugs
- Template method pattern - Real base class inheritance
- Middleware pipeline - Real middleware execution
- Score computation - Real math operations

**Example:**
```python
# ✅ GOOD - Mock LLM call
@pytest.mark.asyncio
async def test_semantic_evaluator_mocked(mocker):
    mocker.patch("arbiter.core.llm_client.LLMManager.get_client")
    evaluator = SemanticEvaluator()
    # Test logic without hitting real API

# ✅ GOOD - Real Pydantic validation
def test_score_validation():
    score = Score(name="test", value=0.95, confidence=0.9)
    assert score.value == 0.95  # Real validation

# ❌ BAD - Using real API in tests
async def test_evaluator():
    result = await evaluate("test", model="gpt-4")  # Costs money!
```

---

## Development Workflow

### Before Starting

1. Check `git status` and `git branch`
2. Create feature branch: `git checkout -b feature/my-feature`

### During Development

1. Follow template method pattern
2. Write tests as you code (not after)
3. Run `make test` frequently

### Before Committing

**Pre-Commit Validation (Run ALL these checks):**
```bash
# 1. Format code
make format

# 2. Lint code
make lint
if [ $? -ne 0 ]; then echo "🚨 LINT ERRORS - FIX BEFORE COMMIT"; exit 1; fi

# 3. Type check
make type-check
if [ $? -ne 0 ]; then echo "🚨 TYPE ERRORS - FIX BEFORE COMMIT"; exit 1; fi

# 4. Run tests with coverage
make test
if [ $? -ne 0 ]; then echo "🚨 TESTS FAILED OR COVERAGE <80%"; exit 1; fi

# 5. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" arbiter/ && echo "🚨 REMOVE TODOs" && exit 1

# 6. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" arbiter/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" && exit 1

# 7. Verify exports
python -c "from arbiter import *; print('✅ All exports work')" || echo "🚨 EXPORT ERROR"

# All checks passed
echo "✅ All checks passed - ready to commit"
git add <files>
git commit -m "Clear message"
```

### After Completing

1. Add example to `examples/` if user-facing
2. Update README.md if API changed

---

## Common Tasks

### Add New Evaluator

```bash
# 1. Create evaluator file
touch arbiter/evaluators/my_evaluator.py

# 2. Implement template methods (see Critical Rules #1)

# 3. Export in arbiter/evaluators/__init__.py
from .my_evaluator import MyEvaluator
__all__ = [..., "MyEvaluator"]

# 4. Export in arbiter/__init__.py
from .evaluators import MyEvaluator
__all__ = [..., "MyEvaluator"]

# 5. Write tests
touch tests/unit/test_my_evaluator.py

# 6. Add example
touch examples/my_evaluator_example.py
```

### Run Tests

```bash
make test              # Run all tests with coverage (requires >80% coverage to pass)
pytest tests/unit/     # Run unit tests only (fast, mocked dependencies)
pytest -v              # Run all tests with verbose output (shows test names and results)
make test-cov          # Generate detailed coverage report with missing lines
pytest tests/unit/test_semantic.py -v  # Run specific test file with verbose output
pytest -k "test_evaluate" -v  # Run tests matching pattern "test_evaluate"
```

---

## Code Quality Standards

### Docstrings

```python
async def evaluate(output: str, reference: Optional[str] = None) -> EvaluationResult:
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
        >>> result = await evaluate(output="Paris", reference="Paris is the capital of France")
        >>> print(result.overall_score)
        0.92
    """
```

### Formatting

- **black**: Line length 88
- **ruff**: Follow pyproject.toml config
- **mypy**: Strict mode (all functions typed)

---

## Quick Reference

### Key Files

- **evaluators/semantic.py**: Reference evaluator implementation
- **pyproject.toml**: Dependencies and config
- **README.md**: User documentation with examples
- **examples/**: 15+ comprehensive examples

### Key Patterns

- **Evaluators**: Template method pattern (4 required methods)
- **Middleware**: Pre/post processing pipeline
- **LLM Client**: Provider-agnostic abstraction
- **Interaction Tracking**: Automatic LLM call logging
- **Cost Calculator**: LiteLLM bundled pricing (consistent with Conduit)

### Make Targets

```bash
make test          # Run pytest with coverage (requires >80%, shows missing lines)
make type-check    # Run mypy in strict mode (all functions must have type hints)
make lint          # Run ruff linter (checks code style and potential bugs)
make format        # Run black formatter (line length 88, modifies files in place)
make all           # Run all checks in order: format → lint → type-check → test
```

---

## Questions?

Check evaluators/semantic.py (reference implementation) or README.md (user docs)
