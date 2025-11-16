# Arbiter - Project Index & Knowledge Base

**Version:** 0.1.0-alpha
**Last Updated:** 2025-11-14
**Status:** Active Development (Alpha)

---

## 📋 Quick Navigation

| Section | Description | Link |
|---------|-------------|------|
| **Getting Started** | Installation, setup, quick start | [README.md](README.md) |
| **API Reference** | Complete API documentation | [API_REFERENCE.md](API_REFERENCE.md) |
| **Architecture** | Design decisions and patterns | [DESIGN_SPEC.md](DESIGN_SPEC.md) |
| **Development** | Contributing, testing, workflows | [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Project Planning** | Roadmap and milestones | [ROADMAP.md](ROADMAP.md), [PROJECT_TODO.md](PROJECT_TODO.md) |
| **Examples** | Usage examples and patterns | [examples/](examples/) |
| **Analysis** | Code improvements and reviews | [claudedocs/](claudedocs/) |

---

## 🏗️ Project Structure

```
arbiter/
├── arbiter/                    # Main package
│   ├── __init__.py             # Public API exports
│   ├── api.py                  # High-level evaluate() and compare() functions
│   ├── core/                   # Core infrastructure
│   │   ├── exceptions.py       # Custom exception hierarchy (8 types)
│   │   ├── interfaces.py       # BaseEvaluator, StorageBackend protocols
│   │   ├── llm_client.py       # Provider-agnostic LLM client
│   │   ├── llm_client_pool.py  # Connection pooling with health checks
│   │   ├── middleware.py       # Logging, metrics, caching, rate limiting
│   │   ├── models.py           # Pydantic data models (EvaluationResult, etc.)
│   │   ├── monitoring.py       # Performance metrics and observability
│   │   ├── registry.py         # Evaluator registry system
│   │   ├── retry.py            # Exponential backoff retry logic
│   │   ├── type_defs.py        # TypedDict definitions
│   │   └── types.py            # Enums (Provider, MetricType, etc.)
│   ├── evaluators/             # Evaluation implementations
│   │   ├── base.py             # BasePydanticEvaluator template
│   │   ├── semantic.py         # SemanticEvaluator (LLM-based similarity)
│   │   ├── custom_criteria.py  # CustomCriteriaEvaluator (domain-specific)
│   │   └── pairwise.py         # PairwiseComparisonEvaluator (A/B testing)
│   ├── storage/                # Storage backends (Phase 4)
│   └── tools/                  # External tools integration
├── tests/
│   ├── unit/                   # Unit tests (>80% coverage target)
│   ├── integration/            # Integration tests
│   └── conftest.py             # Pytest fixtures and configuration
├── examples/                   # Usage examples
├── docs/                       # Documentation (mkdocs)
├── claudedocs/                 # AI-generated analysis and reports
└── benchmarks/                 # Performance testing (planned)
```

---

## 🎯 Core Concepts

### 1. Evaluation Pipeline

**Flow:** Input → Evaluator(s) → LLM Call(s) → Score(s) → Result

```python
# Single evaluator
result = await evaluate(
    output="LLM output to evaluate",
    reference="Ground truth or reference",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

# Multiple evaluators
result = await evaluate(
    output="LLM output",
    evaluators=["semantic", "custom_criteria"],
    criteria="accuracy, clarity"
)
```

**Key Components:**
- **Evaluators:** Pluggable evaluation strategies (semantic, criteria-based, pairwise)
- **LLM Client:** Provider-agnostic interface (OpenAI, Anthropic, Google, Groq)
- **Middleware:** Cross-cutting concerns (logging, metrics, caching)
- **Results:** Structured outputs with full observability

### 2. Automatic Interaction Tracking

**Unique Differentiator:** Every LLM call is automatically tracked

```python
result = await evaluate(output, reference)

# Inspect all LLM calls
for interaction in result.interactions:
    print(f"Model: {interaction.model}")
    print(f"Tokens: {interaction.tokens_used}")
    print(f"Cost: ${result.total_llm_cost():.4f}")
    print(f"Latency: {interaction.latency:.2f}s")
```

**Benefits:**
- **Cost Visibility:** Track exact costs for each evaluation
- **Debugging:** See prompts and responses for transparency
- **Audit Trail:** Complete history of all LLM interactions
- **Performance:** Identify bottlenecks and optimization opportunities

### 3. Provider-Agnostic Design

**Principle:** Work with any LLM provider without code changes

```python
# OpenAI
result = await evaluate(output, model="gpt-4o")

# Anthropic
result = await evaluate(output, model="claude-3-5-sonnet")

# Google Gemini
result = await evaluate(output, model="gemini-1.5-pro")

# Groq (fast inference)
result = await evaluate(output, model="mixtral-8x7b")
```

**Supported Providers:**
- OpenAI (GPT-3.5, GPT-4, GPT-4 Turbo)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini Pro, Gemini Flash)
- Groq (Llama, Mixtral)
- Mistral, Cohere

### 4. Middleware Pipeline

**Pattern:** Cross-cutting concerns applied consistently

```python
from arbiter.core.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    MiddlewarePipeline,
)

pipeline = MiddlewarePipeline([
    LoggingMiddleware(log_level="INFO"),
    MetricsMiddleware(),
    CachingMiddleware(ttl=3600),
])

# Works with both evaluate() and compare()
result = await evaluate(output, reference, middleware=pipeline)
comparison = await compare(output_a, output_b, middleware=pipeline)
```

**Built-in Middleware:**
- **LoggingMiddleware:** Detailed operation logging
- **MetricsMiddleware:** Performance and cost tracking
- **CachingMiddleware:** Result caching for efficiency
- **RateLimitingMiddleware:** API rate limiting
- **Custom:** Easy to create custom middleware

---

## 📚 Key Files & Documentation

### Core Documentation

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Project overview, installation, quick start | Users, newcomers |
| [DESIGN_SPEC.md](DESIGN_SPEC.md) | Architecture, design decisions, vision | Developers, contributors |
| [AGENTS.md](AGENTS.md) | AI agent development guide | AI assistants, developers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines | Contributors |

### Planning & Roadmap

| File | Purpose | Status |
|------|---------|--------|
| [ROADMAP.md](ROADMAP.md) | Multi-milestone roadmap | Active (Phase 3 starting) |
| [PROJECT_TODO.md](PROJECT_TODO.md) | Current milestone tasks | Active (Phase 2.5 tasks) |
| [PHASE2_REVIEW.md](PHASE2_REVIEW.md) | Phase 2 retrospective | Complete |
| [EVALUATOR_RECOMMENDATIONS.md](EVALUATOR_RECOMMENDATIONS.md) | Evaluator priorities | Reference |
| [docs/TOOLS_PLUGIN_ARCHITECTURE.md](docs/TOOLS_PLUGIN_ARCHITECTURE.md) | Tools & plugins for FactualityEvaluator | Phase 5 Design |

### AI-Generated Analysis (claudedocs/)

| File | Purpose | Date |
|------|---------|------|
| [IMPROVEMENTS_2025-11-14.md](claudedocs/IMPROVEMENTS_2025-11-14.md) | Code improvement analysis | 2025-11-14 |
| [PAIRWISE_MIDDLEWARE_IMPLEMENTATION.md](claudedocs/PAIRWISE_MIDDLEWARE_IMPLEMENTATION.md) | Middleware adapter implementation | 2025-11-14 |
| [CLEANUP_REPORT_2025-11-14.md](claudedocs/CLEANUP_REPORT_2025-11-14.md) | Workspace cleanup report | 2025-11-14 |

### Configuration Files

| File | Purpose |
|------|---------|
| pyproject.toml | Python project configuration, dependencies, tools |
| .env.example | Environment variable template |
| .gitignore | Git ignore patterns |

---

## 🔧 API Surface

### High-Level API (arbiter.api)

```python
from arbiter import evaluate, compare

# evaluate() - Single output evaluation
result: EvaluationResult = await evaluate(
    output: str,
    reference: Optional[str] = None,
    evaluators: List[str] = ["semantic"],
    criteria: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    middleware: Optional[MiddlewarePipeline] = None,
) -> EvaluationResult

# compare() - Pairwise comparison
comparison: ComparisonResult = await compare(
    output_a: str,
    output_b: str,
    criteria: Optional[str] = None,
    reference: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    middleware: Optional[MiddlewarePipeline] = None,
) -> ComparisonResult
```

### Evaluators (arbiter.evaluators)

```python
from arbiter import (
    SemanticEvaluator,
    CustomCriteriaEvaluator,
    PairwiseComparisonEvaluator,
)

# Direct usage (more control)
evaluator = SemanticEvaluator(llm_client=client)
score: Score = await evaluator.evaluate(output, reference)
```

### LLM Client (arbiter.core)

```python
from arbiter.core import LLMManager

# Automatic provider detection
client = await LLMManager.get_client(model="gpt-4o")

# Explicit provider
client = await LLMManager.get_client(
    provider="anthropic",
    model="claude-3-5-sonnet"
)
```

### Registry (arbiter.core.registry)

```python
from arbiter.core.registry import (
    register_evaluator,
    get_evaluator_class,
    get_available_evaluators,
)

# Register custom evaluator
register_evaluator("my_evaluator", MyEvaluatorClass)

# List available evaluators
evaluators = get_available_evaluators()  # ['semantic', 'custom_criteria', ...]
```

---

## 📖 Examples Guide

### Basic Examples

| Example | Description | File |
|---------|-------------|------|
| **Basic Evaluation** | Simple semantic similarity | [basic_evaluation.py](examples/basic_evaluation.py) |
| **Custom Criteria** | Domain-specific evaluation | [custom_criteria_example.py](examples/custom_criteria_example.py) |
| **Pairwise Comparison** | A/B testing and model comparison | [pairwise_comparison_example.py](examples/pairwise_comparison_example.py) |

### Advanced Examples

| Example | Description | File |
|---------|-------------|------|
| **Multiple Evaluators** | Combining multiple evaluators | [multiple_evaluators.py](examples/multiple_evaluators.py) |
| **Middleware Usage** | Logging, metrics, caching | [middleware_usage.py](examples/middleware_usage.py) |
| **Pairwise + Middleware** | A/B testing with observability | [pairwise_with_middleware.py](examples/pairwise_with_middleware.py) |
| **Error Handling** | Graceful error handling | [error_handling_example.py](examples/error_handling_example.py) |
| **Provider Switching** | Multi-provider usage | [provider_switching.py](examples/provider_switching.py) |
| **Evaluator Registry** | Custom evaluator registration | [evaluator_registry_example.py](examples/evaluator_registry_example.py) |
| **RAG Evaluation** | Retrieval-augmented generation | [rag_evaluation.py](examples/rag_evaluation.py) |

### Running Examples

```bash
# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Run any example
python examples/basic_evaluation.py
python examples/pairwise_with_middleware.py
```

---

## 🧪 Testing

### Test Organization

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   ├── test_api.py            # API function tests
│   ├── test_semantic.py       # SemanticEvaluator tests
│   ├── test_custom_criteria.py # CustomCriteriaEvaluator tests
│   ├── test_pairwise.py       # PairwiseComparisonEvaluator tests
│   ├── test_pairwise_middleware.py # Middleware integration tests
│   ├── test_registry.py       # Registry system tests
│   ├── test_models.py         # Data model tests
│   └── test_error_handling.py # Exception handling tests
└── integration/               # Integration tests (slower, real APIs)
```

### Running Tests

```bash
# All tests with coverage
pytest tests/ --cov=arbiter --cov-report=term-missing

# Unit tests only (fast)
pytest tests/unit/

# Specific test file
pytest tests/unit/test_pairwise_middleware.py -v

# With detailed output
pytest tests/ -vv --tb=short
```

### Test Coverage

**Target:** >80% coverage
**Current Status:** Active development (Phase 2.5)

---

## 🎨 Architecture Patterns

### 1. Template Method Pattern (Evaluators)

**Pattern:** BasePydanticEvaluator defines evaluation workflow

```python
class MyEvaluator(BasePydanticEvaluator):
    @property
    def name(self) -> str:
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        """Define evaluator behavior."""
        return "You are an expert evaluator..."

    def _get_user_prompt(self, output, reference, criteria) -> str:
        """Format evaluation request."""
        return f"Evaluate: {output}"

    def _get_response_type(self) -> Type[BaseModel]:
        """Define structured response model."""
        return MyResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract score from structured response."""
        return Score(name=self.name, value=response.score)
```

**Benefits:**
- Consistent evaluation workflow
- Type-safe responses (Pydantic)
- Automatic interaction tracking
- Reusable infrastructure

### 2. Middleware Pipeline Pattern

**Pattern:** Chain of responsibility for cross-cutting concerns

```python
class Middleware(ABC):
    @abstractmethod
    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable,
        context: MiddlewareContext,
    ) -> EvaluationResult:
        """Process request and call next middleware."""
        ...

class MiddlewarePipeline:
    async def execute(
        self,
        output: str,
        reference: Optional[str],
        final_handler: Callable,
        context: Optional[MiddlewareContext] = None,
    ) -> EvaluationResult:
        """Execute middleware chain."""
        ...
```

**Benefits:**
- Separation of concerns
- Composable functionality
- Easy to add new middleware
- Consistent application

### 3. Registry Pattern (Evaluators)

**Pattern:** Dynamic evaluator discovery and registration

```python
# Registration
AVAILABLE_EVALUATORS["semantic"] = SemanticEvaluator
AVAILABLE_EVALUATORS["custom_criteria"] = CustomCriteriaEvaluator

# Usage
evaluator_class = get_evaluator_class("semantic")
evaluator = evaluator_class(llm_client=client)
```

**Benefits:**
- Extensibility without code changes
- Type-safe evaluator access
- Dynamic evaluator discovery
- Easy custom evaluator integration

### 4. Adapter Pattern (Pairwise Middleware)

**Pattern:** Bridge signature mismatch between middleware and pairwise comparison

```python
async def execute_comparison(
    self,
    output_a: str,
    output_b: str,
    criteria: Optional[str],
    reference: Optional[str],
    final_handler: Callable,
    context: Optional[MiddlewareContext] = None,
) -> ComparisonResult:
    """Adapt pairwise signature to middleware pipeline."""
    # Package pairwise data in context
    context["is_pairwise_comparison"] = True
    context["pairwise_data"] = {
        "output_a": output_a,
        "output_b": output_b,
        "criteria": criteria,
    }
    # Execute middleware chain
    ...
```

**Benefits:**
- Consistent middleware application
- Backward compatibility
- Context-aware processing
- No code duplication

---

## 🚀 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter

# Create virtual environment (uv recommended)
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Git Workflow

```bash
# Always check status first
git status

# Create feature branch
git checkout -b feature/your-feature

# Make changes, then stage
git add <files>

# Run tests before committing
pytest tests/

# Commit with descriptive message
git commit -m "feat: add new feature"

# Push and create PR
git push -u origin feature/your-feature
```

### Code Quality Checks

```bash
# Format code
black arbiter/ tests/ examples/

# Lint code
ruff check arbiter/ --fix

# Type check
mypy arbiter/

# Run all checks
make lint  # if Makefile available
```

---

## 📊 Project Metrics

### Code Statistics

- **Main Package:** ~2,500 lines (arbiter/)
- **Tests:** ~1,500 lines (tests/)
- **Examples:** ~1,000 lines (examples/)
- **Documentation:** ~5,000 lines (*.md files)

### Test Coverage

- **Target:** >80%
- **Current:** Active development (Phase 2.5)

### Dependencies

**Core:**
- pydantic (2.12+) - Data validation
- pydantic-ai (1.14+) - Structured LLM outputs
- httpx (0.28+) - HTTP client
- python-dotenv (1.0+) - Environment variables

**LLM Providers:**
- openai (2.0+)
- anthropic (0.72+) [optional]
- google-generativeai (0.8.5+) [optional]
- mistralai (1.0+) [optional]
- cohere (5.0+) [optional]

**Development:**
- pytest (9.0+)
- mypy (1.18+)
- black (25.0+)
- ruff (0.14+)

---

## 🔮 Roadmap

### Phase 2.5 (Current - 80% Complete)
- ✅ Registry system for custom evaluators
- ✅ Pairwise comparison evaluator
- ✅ Middleware support for pairwise operations
- ⏳ Documentation improvements
- ⏳ Example expansion

### Phase 3 (Next - Planned)
- Vector database integration (Milvus)
- Embedding-based evaluators
- Advanced caching strategies
- Performance optimizations

### Phase 4+ (Future)
- Storage backends (Redis, PostgreSQL)
- Batch evaluation support
- Streaming evaluation
- Additional evaluators (factuality, toxicity, etc.)

See [ROADMAP.md](ROADMAP.md) for complete roadmap.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code of conduct
- Development setup
- Testing requirements
- PR process
- Code style guide

**Quick Links:**
- [Open Issues](https://github.com/evanvolgas/arbiter/issues)
- [Discussions](https://github.com/evanvolgas/arbiter/discussions)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

Built with:
- [PydanticAI](https://ai.pydantic.dev/) - Structured LLM outputs
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Milvus](https://milvus.io/) - Vector database (planned)

---

**Last Updated:** 2025-11-14
**Maintainer:** Evan Volgas
**Status:** Alpha (0.1.0) - Active Development
