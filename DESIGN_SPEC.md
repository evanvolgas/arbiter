# Arbiter Design Specification

**Version:** 1.0
**Status:** Phase 3 Complete
**Purpose:** Technical vision and architecture

---

## Vision

**Arbiter is production-grade LLM evaluation infrastructure for AI engineers.**

Simple enough for rapid adoption, powerful enough for scale, transparent enough for confidence.

### Core Values

1. **Simplicity** - 3-line API for common cases
2. **Observability** - Automatic interaction tracking
3. **Production-Ready** - Pooling, retry, middleware built-in
4. **Provider-Agnostic** - No vendor lock-in
5. **Extensibility** - Plugin system for customization

---

## Problem Statement

### The Evaluation Challenge

LLM evaluation is critical for production AI systems, but existing tools have gaps:

**Complexity:** LangChain evaluators require deep framework knowledge with steep learning curves.

**Poor Observability:** Black-box evaluations hide cost, token usage, and decision rationale.

**Vendor Lock-In:** OpenAI Evals only works with OpenAI; provider-specific tools create switching costs.

**Experimental Focus:** Most tools lack production features like retry, pooling, and rate limiting.

### Market Gap

Teams need:
- Simple API (like OpenAI Evals)
- Production features (like TruLens)
- Provider-agnostic (unlike OpenAI Evals)
- Evaluation-focused (unlike LangChain)
- Open source (unlike Braintrust)

**Arbiter fills this gap.**

---

## Target Users

### AI Engineers
**Need:** Evaluate LLM outputs quickly and reliably
**Value:** 3-line API with complete cost visibility

**Use Cases:**
- RAG system quality assessment
- Prompt optimization
- Model comparison
- Output validation

### MLOps/Platform Teams
**Need:** Production-ready evaluation infrastructure
**Value:** Built-in pooling, retry, middleware, streaming

**Use Cases:**
- Internal AI platform development
- Pipeline integration
- Production quality monitoring
- Cost optimization

### Researchers & Practitioners
**Need:** Flexible evaluation framework for experimentation
**Value:** Extensible plugin system and custom evaluators

**Use Cases:**
- Novel evaluator development
- Benchmark creation
- Academic research
- Tool comparison studies

---

## Key Features

### 1. Simple API

**Minimal Example:**
```python
from arbiter import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Passed: {result.passed}")
```

**Progressive Complexity:**
```python
# Advanced usage - explicit control
client = await LLMManager.get_client(
    provider="anthropic",
    model="claude-3-5-sonnet"
)

evaluator = SemanticEvaluator(
    client,
    middleware_pipeline=[LoggingMiddleware(), CachingMiddleware()]
)

score = await evaluator.evaluate(output, reference)
```

### 2. Complete Observability

**Automatic LLM Interaction Tracking:**

```python
# Every LLM call tracked automatically
for interaction in result.interactions:
    print(f"Purpose: {interaction.purpose}")
    print(f"Model: {interaction.model}")
    print(f"Tokens: {interaction.tokens_used}")
    print(f"Latency: {interaction.latency}s")
    print(f"Cost: ${interaction.tokens_used * 0.00001:.6f}")
```

**Capabilities:**
- Token usage and cost calculation
- Performance metrics per call
- Complete audit trails
- Confidence scoring with explanations

**Differentiator:** No other framework provides automatic interaction tracking at the evaluator level.

### 3. Provider-Agnostic Design

**Supported Providers:**
- OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3)
- Google (Gemini 1.5 Pro/Flash)
- Groq (Llama 3.1, Mixtral)
- Mistral AI
- Cohere

**Easy Switching:**
```python
# OpenAI
client = await LLMManager.get_client(
    provider="openai",
    model="gpt-4o-mini"
)

# Anthropic
client = await LLMManager.get_client(
    provider="anthropic",
    model="claude-3-5-sonnet"
)
```

### 4. Production Features

**Built-in Reliability:**

**Connection Pooling:**
- Reduces latency through connection reuse
- Automatic health checks
- Resource management

**Automatic Retry:**
- Exponential backoff for transient failures
- Configurable strategies (quick, standard, persistent)
- Selective retry (only for retryable errors)

**Middleware System:**
- Logging - Request/response tracking
- Metrics - Performance monitoring
- Caching - LRU with configurable TTL
- Rate Limiting - Token bucket algorithm

**Performance Monitoring:**
- Per-operation metrics
- Token tracking
- Cost analysis
- Error tracking

### 5. Extensible Architecture

**Custom Evaluators:**

Follow the template method pattern by implementing 4 methods:

```python
class MyEvaluator(BasePydanticEvaluator):
    def _get_system_prompt(self) -> str:
        return "You are an expert evaluator..."

    def _get_user_prompt(self, output, reference, criteria) -> str:
        return f"Evaluate: {output}"

    def _get_response_type(self) -> Type[BaseModel]:
        return MyResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        return Score(name=self.name, value=response.score)
```

**Plugin System:**
- Custom middleware
- Verification plugins (for FactualityEvaluator)
- Storage backends
- Custom providers

---

## Architecture

### Layered Design

```
┌─────────────────────────────────────┐
│       Public API (api.py)           │  Simple entry point
│  evaluate(), compare()              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Middleware Pipeline            │  Cross-cutting concerns
│  Logging, Metrics, Caching,         │
│  Rate Limiting                      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Evaluators (via LLM Clients)      │  Business logic
│  ├─ SemanticEvaluator               │
│  ├─ CustomCriteriaEvaluator         │
│  ├─ FactualityEvaluator             │
│  └─ Custom Evaluators               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   LLM Client + Connection Pool      │  Provider abstraction
│  (OpenAI, Anthropic, Google, etc.)  │
├─ PydanticAI Integration             │
├─ Automatic Retry Logic              │
└─ Token Tracking                     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│     External LLM APIs               │  External services
│  (OpenAI, Anthropic, Gemini, etc.)  │
└─────────────────────────────────────┘
```

### Design Principles

**SOLID:**
- Single Responsibility - Each class has one reason to change
- Open/Closed - Extensible via evaluator interface
- Liskov Substitution - Evaluators are substitutable
- Interface Segregation - Small, focused protocols
- Dependency Inversion - Depends on abstractions

**Design Patterns:**
- Template Method - `BasePydanticEvaluator` enforces structure
- Chain of Responsibility - Middleware pipeline
- Object Pool - `LLMClientPool` manages connections
- Strategy - Multiple evaluators, single interface
- Registry - Evaluator discovery and validation

**Type Safety:**
- Strict mypy enforcement
- Pydantic models throughout
- PydanticAI structured outputs
- Protocol-based duck typing

---

## Data Models

### Score
Individual evaluation metric:
```python
class Score(BaseModel):
    name: str              # "semantic_similarity"
    value: float           # 0.0 to 1.0
    confidence: float      # Evaluator confidence
    explanation: str       # Human-readable reasoning
    metadata: Dict         # Additional context
```

### LLMInteraction
Complete LLM call record:
```python
class LLMInteraction(BaseModel):
    prompt: str            # Sent to LLM
    response: str          # LLM's response
    model: str             # Model used
    tokens_used: int       # Token count
    latency: float         # Response time (seconds)
    timestamp: datetime    # When called
    purpose: str           # Evaluation purpose
    metadata: Dict         # Evaluator info
```

### EvaluationResult
Complete evaluation outcome:
```python
class EvaluationResult(BaseModel):
    # Inputs
    output: str
    reference: Optional[str]
    criteria: Optional[str]

    # Results
    scores: List[Score]
    overall_score: float   # 0.0 to 1.0
    passed: bool           # Threshold comparison

    # Metadata
    metrics: List[Metric]
    evaluator_names: List[str]
    total_tokens: int
    processing_time: float
    interactions: List[LLMInteraction]  # Complete audit trail
```

---

## Current Evaluators

### SemanticEvaluator
**Purpose:** Similarity scoring between output and reference

**Backends:**
- **LLM backend (default):** Rich explanations, slower (~2s), costs tokens
- **FAISS backend (optional):** Fast (~50ms), free, deterministic (requires `pip install arbiter[scale]`)

**Modes:**
- With reference - Compare semantic similarity
- With criteria - Evaluate against requirements
- Standalone - Assess output quality (LLM backend only)

### CustomCriteriaEvaluator
**Purpose:** Domain-specific evaluation

**Capabilities:**
- Single criterion assessment
- Multi-criteria evaluation
- Flexible prompt templates
- Metadata-rich results (criteria met/not met)

### PairwiseComparisonEvaluator
**Purpose:** A/B testing and model comparison

**Capabilities:**
- Side-by-side comparison
- Winner determination with confidence
- Aspect-level analysis
- Comprehensive comparison metadata

### FactualityEvaluator (Phase 3)
**Purpose:** Hallucination detection

**Capabilities:**
- Claim extraction
- Fact verification
- Source attribution (with plugins)
- Claim-level accuracy

See **ROADMAP.md** for upcoming evaluators.

---

## Competitive Differentiation

### Key Features

1. **Automatic Interaction Tracking** - Tracks all LLM calls with cost, latency, and token usage
2. **Simple + Production** - Easy 3-line API with production features (pooling, retry, middleware)
3. **Provider-Agnostic** - Works with OpenAI, Anthropic, Google, Groq, Mistral, Cohere
4. **Open Source Infrastructure** - MIT license, community-driven development

### Comparison with Alternatives

| Capability | Arbiter | LangChain | TruLens | OpenAI Evals |
|------------|---------|-----------|---------|--------------|
| API Simplicity | 3-line API | Requires framework integration | Moderate API | Simple config files |
| Observability | Automatic interaction tracking | Manual instrumentation | Comprehensive dashboard | Limited |
| Provider-Agnostic | 6 providers supported | Multiple providers | Multiple providers | OpenAI only |
| Production Features | Pooling, retry, middleware | Limited | Comprehensive | Minimal |

---

## Technology Stack

### Core Dependencies
- Python 3.10+ (modern type hints)
- Pydantic 2.12+ (validation)
- PydanticAI 1.14+ (structured LLM outputs)
- HTTPX 0.28+ (async HTTP)

### Provider SDKs
- OpenAI 2.0+
- Anthropic 0.72+
- Google (Gemini) 0.8.5+
- Groq, Mistral, Cohere (latest)

### Optional
- FAISS (`pip install arbiter[scale]`) - ✅ Implemented for SemanticEvaluator, planned for fact verification caching
- Redis 5.0+ (storage backend - deferred to v2.0)

### Development Tools
- pytest 9.0+ (testing)
- black 25.0+ (formatting)
- ruff 0.14+ (linting)
- mypy 1.18+ (strict type checking)

---

## Design Constraints

### Must Have

1. **Provider-Agnostic** - Must work with any LLM provider
2. **Type Safety** - Strict mypy, Pydantic throughout
3. **Production-Grade** - Retry, pooling, monitoring required
4. **Simple API** - 3-line usage for common cases
5. **Open Source** - MIT license, community-driven

### Will Not Do

1. **UI Dashboards** - Infrastructure-focused, not UI-focused
2. **Experiment Tracking** - Integrate with W&B, MLflow instead
3. **Prompt Engineering Tools** - Orthogonal concern
4. **LLM Training** - Evaluation only
5. **Data Labeling** - Not an annotation tool

---

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Framework Overhead | <50ms (p95) | ✅ <50ms |
| Test Coverage | >80% | ✅ 95% |
| Factuality with External Verification | Planned | 🚧 Phase 5 |
| Type Safety | 100% (strict mypy) | ✅ 100% |

---

## Integration Points

### Loom Integration

Arbiter serves as the evaluation engine for Loom AI pipelines:

```yaml
# Loom pipeline configuration
evaluate:
  evaluators:
    - name: factuality_check
      type: factuality
      threshold: 0.85
    - name: relevance_check
      type: relevance
      threshold: 0.80
  quality_gate: all_pass
```

### LangChain Integration (Planned)

Arbiter evaluators can be used in LangChain workflows:

```python
from langchain.evaluation import load_evaluator
from arbiter import SemanticEvaluator

# Use Arbiter evaluator in LangChain
evaluator = load_evaluator("arbiter", evaluator_class=SemanticEvaluator)
```

---

## Related Documentation

**Project Management:**
- **ROADMAP.md** - Development timeline and phases
- **PROJECT_TODO.md** - Current milestone tasks
- **DESIGN_DECISIONS.md** - Architectural choices and rationale

**User-Facing:**
- **README.md** - User-facing overview and quick start
- **examples/** - 15+ comprehensive examples demonstrating all evaluators

---

## Summary

Arbiter is **production-grade LLM evaluation infrastructure** that prioritizes:

**Simplicity** → 3-line API
**Observability** → Automatic interaction tracking
**Production-Ready** → Pooling, retry, middleware
**Provider-Agnostic** → Works with any LLM
**Extensibility** → Plugin system for customization

Built for AI engineers, MLOps teams, and researchers who need reliable evaluation without complexity.

---

**Last Updated:** 2025-11-16
