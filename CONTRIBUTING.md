# Contributing to Arbiter

Thank you for your interest in contributing to Arbiter! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

Be respectful and professional. We're building production-grade software together.

## Getting Started

### Prerequisites

- Python 3.11+
- At least one LLM provider API key (OpenAI, Anthropic, Google, or Groq)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ashita-ai/arbiter.git
cd arbiter

# Install with development dependencies
uv sync --all-extras

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Verify installation
uv run pytest tests/unit/test_semantic.py -v
```

## Making Changes

### Before You Start

1. **Check existing issues** - Look for related issues or discussions
2. **Create an issue first** - For significant changes, open an issue to discuss your approach
3. **Create a feature branch** - Never work directly on `main`

```bash
git checkout -b feature/your-feature-name
```

### Code Standards

- **Type hints required** - All functions must have complete type annotations
- **Docstrings required** - Use Google-style docstrings with Args, Returns, Raises, Example
- **No placeholders** - No TODO comments or NotImplementedError in production code
- **Complete features** - Finish what you start (implementation + tests + docs + examples)

### Template Method Pattern

All evaluators must follow the template method pattern:

```python
class MyEvaluator(BasePydanticEvaluator):
    @property
    def name(self) -> str:
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        return "System prompt..."

    def _get_user_prompt(self, output: str, reference: Optional[str], criteria: Optional[str]) -> str:
        return f"Evaluate: {output}"

    def _get_response_type(self) -> Type[BaseModel]:
        return MyEvaluatorResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(MyEvaluatorResponse, response)
        return Score(name=self.name, value=resp.score, ...)
```

## Testing

### Running Tests

```bash
# All tests with coverage (requires >80%)
uv run pytest

# Specific test file
uv run pytest tests/unit/test_semantic.py -v

# With coverage report
uv run pytest --cov=arbiter --cov-report=term-missing
```

### Test Requirements

- **Coverage** - Maintain >80% coverage for all new code
- **Unit tests** - Test individual functions with mocked dependencies
- **Integration tests** - Test end-to-end evaluation flows
- **Mock external APIs** - Don't hit real LLM APIs in unit tests

## Code Quality

### Pre-Commit Checklist

Run these commands before every commit:

```bash
# 1. Format code
make format

# 2. Lint code (must pass)
make lint

# 3. Type check (must pass)
make type-check

# 4. Run tests (must pass with >80% coverage)
make test

# Or run all checks at once
make all
```

### Required Tools

- **black** - Code formatting (line length 88)
- **ruff** - Fast linting
- **mypy** - Strict type checking
- **pytest** - Testing framework

## Pull Request Process

### Before Submitting

1. **Update tests** - Add tests for new features or bug fixes
2. **Update documentation** - Update README.md if API changed
3. **Add examples** - Create example file in `examples/` for new evaluators
4. **Update exports** - Add new classes to `__init__.py` files
5. **Run all checks** - Ensure `make all` passes

### PR Requirements

- **Clear description** - Explain what and why (not just what)
- **Reference issues** - Link to related issues
- **One feature per PR** - Keep PRs focused and reviewable
- **Passing CI** - All checks must pass
- **No merge conflicts** - Rebase on main if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New evaluator
- [ ] Enhancement
- [ ] Documentation

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] All tests pass
- [ ] Coverage >80%

## Checklist
- [ ] Code follows style guidelines (black, ruff, mypy pass)
- [ ] Added/updated docstrings
- [ ] Updated README.md if needed
- [ ] Added example in examples/ if needed
- [ ] Updated __init__.py exports
```

## Issue Guidelines

### Bug Reports

Use the bug report template. Include:
- **Model and evaluator used**
- **Expected vs actual behavior**
- **Minimal reproduction code**
- **Error messages and stack traces**
- **Cost observed** (if relevant)

### Feature Requests

Use the feature request template. Include:
- **Use case** - Why is this needed?
- **Proposed solution** - What should it do?
- **Alternatives considered** - What else did you consider?

### Questions

For questions about usage:
- Check README.md and examples/ first
- Search existing issues
- Provide context about what you're trying to achieve

## What We Won't Build

To set clear expectations:

- **Hosted platform or UI** - Arbiter is a pure Python library
- **Support for Python <3.10** - Modern type hints required
- **Built-in dashboard** - Use external tools for visualization
- **Non-LLM evaluators** - Focus is LLM-as-judge evaluation

## Additional Resources

- **README.md** - User documentation and examples
- **AGENTS.md** - Detailed development guide (AI-focused)
- **examples/** - Comprehensive usage examples
- **docs/** - API documentation

## Questions?

- Open an issue for bugs or features
- Check existing issues and examples first
- Be specific and provide context

Thank you for contributing to Arbiter!
