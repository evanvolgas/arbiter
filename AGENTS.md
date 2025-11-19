# CLAUDE.md - AI Agent Guide

**Purpose**: Quick reference for working on Arbiter
**Last Updated**: 2025-11-16

---

## Quick Orientation

**Arbiter**: Production-grade LLM evaluation framework (v0.1.0-alpha)
**Stack**: Python 3.11+, PydanticAI, provider-agnostic
**Supported Providers**: OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude 3.5), Google (Gemini 2.0), Groq, Mistral, Cohere
**Coverage**: 95% test coverage, strict mypy, comprehensive examples

### Directory Structure

```
arbiter/
├── arbiter/
│   ├── api.py              # Public API (evaluate, compare)
│   ├── core/               # Infrastructure (llm_client, middleware, monitoring, registry)
│   ├── evaluators/         # Semantic, CustomCriteria, Pairwise, Factuality, Groundedness, Relevance
│   ├── storage/            # Storage backends (Phase 4)
│   └── tools/              # Utilities
├── examples/               # 15+ comprehensive examples
├── tests/                  # Unit + integration tests
└── pyproject.toml          # Dependencies and config
```

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
# ✅ GOOD
client = await LLMManager.get_client(provider="anthropic", model="claude-3-5-sonnet")

# ❌ BAD
from openai import OpenAI
client = OpenAI()  # Hardcoded to OpenAI
```

### 3. Type Safety (Strict Mypy)
All functions require type hints, no `Any` without justification.

### 4. No Placeholders/TODOs
Production-grade code only. Complete implementations or nothing.

### 5. Complete Features Only
If you start, you finish:
- ✅ Implementation complete
- ✅ Tests (>80% coverage)
- ✅ Docstrings
- ✅ Example code
- ✅ Exported in `__init__.py`

### 6. PydanticAI for Structured Outputs
All evaluators use PydanticAI for type-safe LLM responses.

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
```bash
make test        # Tests pass
make type-check  # Mypy clean
make lint        # Ruff clean
make format      # Black formatted
```

### After Completing
1. Add example to `examples/` if user-facing
2. Update README.md if API changed

---

## Common Tasks

### Run Examples
```bash
# All examples work with uv run
uv run python examples/basic_evaluation.py
uv run python examples/cost_comparison.py
uv run python examples/custom_criteria_example.py

# Or use make
make run-example EXAMPLE=basic_evaluation.py
```

**Important**: Always use `uv run` to execute examples. The virtual environment created by `uv sync` requires `uv run` for proper package discovery.

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
- **evaluators/semantic.py** - Reference evaluator implementation
- **pyproject.toml** - Dependencies and config
- **README.md** - User documentation with examples
- **examples/** - 15+ comprehensive examples

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

## Working with AI Agents

### Task Management
**Use TodoWrite for multi-step tasks**: When a task involves 3+ distinct steps, track progress with TodoWrite. This ensures nothing gets forgotten and provides visibility into progress.

**Plan before executing**: For complex tasks, create a plan first. Understand requirements, identify dependencies, then execute systematically.

### Audience & Context Recognition
**Recognize audience immediately**: Engineers get technical tone, no marketing language. Business audiences get value/ROI focus. Academic audiences get methodology and rigor. Adapt tone and content immediately based on context.

**Separate material types**: Code examples stay clean (no narratives or marketing). Presentation materials (openers, talking points) live in separate files. Documentation explains architecture and usage patterns.

### Quality & Testing
**Test output quality, not just functionality**: Run code AND verify the output is actually useful. Truncated or abstracted output defeats the purpose of examples. Show real data structures, not summaries.

**Verify before committing**: Run tests and verify examples work before showing output. Test both functionality and usefulness.

**Connect work to strategy**: Explicitly reference project milestones, coverage targets, and strategic priorities when completing work. Celebrate milestones when achieved.

### Workflow Patterns
**Iterate fast**: Ship → test → get feedback → fix → commit. Don't perfect upfront. Progressive refinement beats upfront perfection.

**Proactive problem solving**: Use tools like Glob to check file existence before execution. Anticipate common issues and handle them gracefully.

**Parallel execution**: Batch independent operations (multiple reads, parallel test execution) to improve efficiency.

### Communication & Feedback
**Direct feedback enables fast iteration**: Clear, immediate feedback on what's wrong enables rapid course correction. Specific, actionable requests work better than vague suggestions.

**Match user communication style**: Some users prefer speed over process formality, results over explanations. Adapt communication style accordingly while maintaining quality standards.

### Git & Commit Hygiene
**Commit hygiene**: Each meaningful change gets its own commit with clear message (what + why). This makes progress tracking and rollback easier.

**Clean git workflow**: Always check `git status` and `git branch` before operations. Use feature branches for all changes.

---

**Questions?** Check evaluators/semantic.py (reference implementation) or README.md (user docs)

**Last Updated**: 2025-01-18
