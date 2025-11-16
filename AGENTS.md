# CLAUDE.md - AI Agent Guide

**Purpose**: Quick reference for working on Arbiter
**Last Updated**: 2025-11-15

---

## Quick Orientation

**Arbiter**: Production-grade LLM evaluation framework (v0.1.0-alpha)
**Stack**: Python 3.10+, PydanticAI, provider-agnostic (OpenAI/Anthropic/Google/Groq)
**Coverage**: 95% test coverage, strict mypy, comprehensive examples
**Related**: Used by Loom (AI pipeline orchestrator) for quality gates

### Directory Structure

```
arbiter/
├── arbiter/
│   ├── api.py              # Public API (evaluate, compare)
│   ├── core/               # Infrastructure (llm_client, middleware, monitoring, registry)
│   ├── evaluators/         # Semantic, CustomCriteria, Pairwise, Factuality
│   ├── storage/            # Storage backends (Phase 4)
│   └── tools/              # Utilities
├── examples/               # 15+ comprehensive examples
├── tests/                  # Unit + integration tests
├── DESIGN_SPEC.md          # Vision and architecture
├── PROJECT_TODO.md         # Current milestone tasks
└── pyproject.toml          # Dependencies and config
```

**Key docs**: DESIGN_SPEC.md (vision) → PROJECT_TODO.md (current work) → CLAUDE.md (this file)

---

## Critical Rules (Non-Negotiable)

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
# ✅ GOOD
client = await LLMManager.get_client(provider="anthropic", model="claude-3-5-sonnet")

# ❌ BAD
from openai import OpenAI
client = OpenAI()  # Hardcoded to OpenAI
```

### 3. Type Safety (Strict Mypy)
All functions require type hints, no `Any` without justification.

```python
# ✅ GOOD
async def evaluate(output: str, reference: Optional[str] = None) -> EvaluationResult:
    ...

# ❌ BAD
async def evaluate(output, reference=None):  # Missing types
    ...
```

### 4. No Placeholders/TODOs
Production-grade code only. Complete implementations or nothing.

```python
# ❌ NEVER
def calculate_cost(tokens: int) -> float:
    # TODO: implement
    raise NotImplementedError()

# ✅ ALWAYS
def calculate_cost(tokens: int, cost_per_1k: float = 0.01) -> float:
    return (tokens / 1000) * cost_per_1k
```

### 5. Complete Features Only
If you start, you finish:
- ✅ Implementation complete
- ✅ Tests (>80% coverage)
- ✅ Docstrings
- ✅ Example code
- ✅ Exported in `__init__.py`

### 6. PydanticAI for Structured Outputs
All evaluators use PydanticAI for type-safe LLM responses.

```python
class MyResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    explanation: str

# PydanticAI handles validation automatically
```

---

## Development Workflow

### Before Starting
1. Check `git status` and `git branch`
2. Read PROJECT_TODO.md for current milestone
3. Create feature branch: `git checkout -b feature/my-feature`

### During Development
1. Follow template method pattern
2. Write tests as you code (not after)
3. Run `make test` frequently
4. Update PROJECT_TODO.md checkboxes

### Before Committing
```bash
make test        # Tests pass
make type-check  # Mypy clean
make lint        # Ruff clean
make format      # Black formatted
```

### After Completing
1. Mark task complete in PROJECT_TODO.md
2. Add example to `examples/` if user-facing
3. Update README.md if API changed

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
make test              # All tests with coverage
pytest tests/unit/     # Unit tests only
pytest -v              # Verbose output
make test-cov          # Coverage report
```

### Add Middleware
```python
class MyMiddleware(Middleware):
    async def process(self, output: str, reference: Optional[str], next_handler: Callable, context: Dict[str, Any]) -> EvaluationResult:
        # Pre-processing
        start = time.time()

        # Call next in chain
        result = await next_handler(output, reference)

        # Post-processing
        elapsed = time.time() - start
        logger.info(f"Evaluation took {elapsed:.2f}s")

        return result
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

### Essential Files
- **DESIGN_SPEC.md** - Vision and architecture
- **DESIGN_DECISIONS.md** - Architectural choices and rationale
- **ROADMAP.md** - Development timeline and phases
- **PROJECT_TODO.md** - Current milestone tasks
- **evaluators/semantic.py** - Reference evaluator implementation
- **pyproject.toml** - Dependencies and config

### Key Patterns
- **Evaluators**: Template method pattern (4 required methods)
- **Middleware**: Pre/post processing pipeline
- **LLM Client**: Provider-agnostic abstraction
- **Interaction Tracking**: Automatic LLM call logging

### Make Targets
```bash
make test          # Run tests with coverage
make type-check    # Run mypy
make lint          # Run ruff
make format        # Run black
make all           # Format + lint + type-check + test
```

---

**Questions?** Check DESIGN_SPEC.md (vision), PROJECT_TODO.md (current work), or evaluators/semantic.py (reference implementation)

**Last Updated**: 2025-11-15 | **Next Review**: 2025-12-15
