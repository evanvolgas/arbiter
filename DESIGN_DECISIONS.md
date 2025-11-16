# Design Decisions

**Purpose:** Record architectural choices, trade-offs, and technical rationale

**Audience:** Contributors, maintainers, and users evaluating Arbiter

---

## Core Architecture

### Provider-Agnostic Design

**Decision:** Abstract all LLM provider interactions behind `LLMClient` interface

**Rationale:**
- Users shouldn't be locked into a single provider
- Provider landscape changes rapidly (pricing, features, availability)
- Multi-provider support enables cost optimization and fallback strategies

**Trade-offs:**
- More complexity than OpenAI-only implementation
- Cannot use provider-specific features without abstraction
- Requires testing across multiple providers

**Status:** Implemented ✅

---

### PydanticAI for Structured Outputs

**Decision:** Use PydanticAI for all evaluator LLM interactions

**Rationale:**
- Type-safe structured outputs eliminate parsing errors
- Automatic validation reduces bugs
- Provider abstraction built-in (supports OpenAI, Anthropic, Google, etc.)
- Better developer experience with IDE autocomplete

**Trade-offs:**
- Dependency on relatively new library (PydanticAI 1.0+)
- Small performance overhead vs raw API calls
- Limited to providers PydanticAI supports

**Alternatives Considered:**
- Instructor: More mature but less provider-agnostic
- Raw API + manual parsing: Error-prone and verbose
- LangChain: Too heavy for our focused use case

**Status:** Implemented ✅

---

### Template Method Pattern for Evaluators

**Decision:** All evaluators extend `BasePydanticEvaluator` and implement 4 required methods

**Rationale:**
- Enforces consistency across evaluators
- Automatic interaction tracking without duplication
- Clear contract for custom evaluators
- Reduces boilerplate while maintaining flexibility

**Required Methods:**
1. `_get_system_prompt()` - Define expert role
2. `_get_user_prompt()` - Structure evaluation request
3. `_get_response_type()` - Specify Pydantic response model
4. `_compute_score()` - Calculate score from LLM response

**Trade-offs:**
- Less flexible than interface-only design
- Evaluators must fit the template (may not suit all use cases)
- Inheritance over composition

**Alternatives Considered:**
- Protocol-only design: More flexible but loses automatic features
- Composition: More modular but more boilerplate

**Status:** Implemented ✅

---

### Automatic LLM Interaction Tracking

**Decision:** Track every LLM call automatically in `BasePydanticEvaluator.evaluate()`

**Rationale:**
- Complete cost visibility crucial for production
- Audit trails required for compliance
- Debugging needs full interaction history
- No other framework provides this automatically

**Implementation:**
- Capture: model, provider, tokens, latency, purpose
- Store in `EvaluationResult.interactions` list
- Zero configuration required from users

**Trade-offs:**
- Small performance overhead (< 1ms per call)
- Memory usage for storing interactions
- Cannot disable (always on)

**Status:** Implemented ✅ (Unique differentiator)

---

## Evaluator Design

### Multi-Evaluator Support

**Decision:** Support running multiple evaluators in single `evaluate()` call

**Rationale:**
- Real-world evaluation needs multiple perspectives
- Combining semantic + factuality + relevance common pattern
- Single API call simpler than manual orchestration

**Implementation:**
```python
result = await evaluate(
    output="...",
    evaluators=["semantic", "factuality", "relevance"]
)
# Returns aggregated score + individual scores
```

**Trade-offs:**
- More complex error handling (partial failures)
- Increased latency (sequential by default)
- Could parallelize but adds complexity

**Status:** Implemented ✅

---

### Evaluator Registry System

**Decision:** Central registry for evaluator discovery and validation

**Rationale:**
- Type-safe evaluator names (Literal type hints)
- Early validation before expensive LLM calls
- Easy discovery of available evaluators
- Enables custom evaluator registration

**Implementation:**
- `AVAILABLE_EVALUATORS` dict in `registry.py`
- Auto-registration of built-in evaluators
- `register_evaluator()` for custom evaluators

**Trade-offs:**
- Global state (registry is singleton)
- Import-time registration overhead
- Must update registry for new evaluators

**Status:** Implemented ✅

---

## Infrastructure

### Middleware Pipeline

**Decision:** Composable middleware for cross-cutting concerns

**Rationale:**
- Logging, metrics, caching, rate limiting are universal needs
- Middleware pattern keeps evaluators focused
- Users can add custom middleware
- No framework lock-in (standard pattern)

**Built-in Middleware:**
- `LoggingMiddleware` - Request/response logging
- `MetricsMiddleware` - Performance tracking
- `CachingMiddleware` - LRU caching with configurable TTL
- `RateLimitingMiddleware` - Token bucket algorithm

**Trade-offs:**
- Adds abstraction layer
- Small performance overhead per middleware
- Order matters (middleware is sequential)

**Status:** Implemented ✅

---

### Connection Pooling

**Decision:** Automatic connection pooling for LLM clients

**Rationale:**
- Reduces latency (reuse connections)
- Better resource management
- Health checks prevent dead connections
- Transparent to users

**Implementation:**
- `LLMClientPool` manages connections per provider
- Automatic cleanup of idle connections
- Health checks on checkout

**Trade-offs:**
- Memory overhead (maintains pool)
- Complexity in async context
- Potential for connection leaks if not careful

**Status:** Implemented ✅

---

### Retry with Exponential Backoff

**Decision:** Automatic retry for transient failures

**Rationale:**
- LLM APIs have intermittent issues
- Exponential backoff prevents thundering herd
- Critical for production reliability

**Presets:**
- `quick` - 2 retries, 1s max delay
- `standard` - 3 retries, 10s max delay
- `persistent` - 5 retries, 60s max delay

**Trade-offs:**
- Increased latency on failures
- Could mask underlying issues
- Users must choose appropriate preset

**Status:** Implemented ✅

---

## Plugin Architecture

### FactVerificationPlugin Protocol

**Decision:** Protocol-based plugin system for external fact verification

**Rationale:**
- FactualityEvaluator benefits from external source verification
- Multiple verification sources needed (Tavily, Wikipedia, Wikidata)
- Users may need custom sources (internal knowledge bases)

**Design:**
```python
class FactVerificationPlugin(Protocol):
    async def verify_claim(self, claim: str) -> VerificationResult:
        ...
```

**Plugins Planned:**
- TavilyPlugin - Real-time web search ($0.001/search)
- WikidataPlugin - Structured knowledge (free)
- WikipediaPlugin - Encyclopedia (free)
- WolframPlugin - Mathematical/scientific ($0.003/query)

**Trade-offs:**
- External API dependencies
- Cost per verification
- Latency increase (network calls)
- Requires API keys for some plugins

**Status:** Planned (Phase 5)

---

### Vector Cache for Fact Verification

**Decision:** FAISS for v1.0, Milvus deferred to v2.0+

**Rationale:**
- FAISS provides 90% of benefits with 10% of complexity for v1.0 users
- No server infrastructure required (pip install only)
- Significantly faster than LLM-based similarity, zero operational cost
- Facts are relatively stable (cache long-term)
- Semantic similarity enables fuzzy matching
- Significant speedup on cache hits (embeddings computed once)
- Reduced costs on cached verifications (no repeated LLM calls)

**v1.0 Implementation (FAISS):**
- Optional install: `pip install arbiter[scale]` (adds faiss-cpu + sentence-transformers)
- Embed claims using sentence transformers
- Search for similar claims (cosine similarity > 0.95)
- Local file-based persistence

**v2.0+ Path (Milvus):**
- Milvus targets enterprise users (1M+ evaluations/day)
- Distributed vector search at massive scale
- Requires Docker/server infrastructure
- Deferred to preserve simplicity for 90% of users

**Trade-offs:**
- FAISS: Limited to single-machine scale, no distributed search
- Milvus: More powerful but adds operational complexity
- Choice: Optimize for ease of adoption in v1.0

**Status:** ✅ FAISS implemented (Phase 2.5), Milvus deferred (v2.0+)

---

## Scope Decisions

### What We Include

**Core Evaluators (Built-in):**
- Semantic similarity
- Custom criteria (domain-specific)
- Pairwise comparison
- Factuality (hallucination detection)
- Groundedness (RAG validation)
- Relevance (query alignment)

**Infrastructure:**
- Provider-agnostic LLM client
- Middleware pipeline
- Connection pooling
- Automatic retry
- Interaction tracking

**Rationale:** Focus on production-grade evaluation infrastructure, not experiment tracking or prompt engineering tools.

---

### What We Exclude

**Out of Scope:**
- UI dashboards (infrastructure-focused, not UI-focused)
- Prompt engineering tools (orthogonal concern)
- LLM training/fine-tuning (different domain)
- Storage backends for results (users can persist themselves)
- Streaming support initially (niche use case, high complexity)

**Deferred to Community/v2.0:**
- Additional evaluators beyond core 6
- Quality assurance tools (calibration, agreement metrics)
- Advanced storage (Redis, PostgreSQL)
- Real-time streaming evaluation

**Rationale:** Stay focused on core value proposition. Defer nice-to-haves to maintain momentum toward v1.0.

---

## Performance Targets

### Latency

**Decision:** Target p95 latency < 200ms for semantic evaluation (excluding LLM call)

**Rationale:**
- Evaluation shouldn't significantly slow down development loops
- Network latency to LLM APIs dominates (500-2000ms)
- Framework overhead should be negligible

**Status:** Achieved ✅ (measured < 50ms overhead)

---

### Test Coverage

**Decision:** Maintain >80% test coverage

**Rationale:**
- Production-grade framework requires high reliability
- Type safety (mypy strict) + tests catch most bugs
- Confidence for contributors and users

**Current:** 95% coverage

**Status:** Achieved ✅

---

### Accuracy

**Decision:** Use external verification plugins to improve factuality evaluation accuracy

**Rationale:**
- Pure LLM-based evaluation is limited by model knowledge and hallucination risks
- External verification (web search, knowledge bases) provides evidence-based fact-checking
- Multi-plugin approach increases reliability through source redundancy

**Status:** Planned (Phase 5 - plugin implementation)

---

## API Design

### Simplicity First

**Decision:** 3-line API for common cases, progressive complexity for advanced

**Example:**
```python
# Simple (3 lines)
result = await evaluate(
    output="Paris is the capital of France",
    evaluators=["semantic"]
)

# Advanced (explicit control)
client = await LLMManager.get_client(provider="anthropic", model="claude-3-5-sonnet")
evaluator = SemanticEvaluator(client, middleware_pipeline=[...])
score = await evaluator.evaluate(output, reference, criteria)
```

**Rationale:**
- Low barrier to entry for beginners
- Power users can opt into complexity
- Matches user mental model (simple → advanced)

**Status:** Implemented ✅

---

### Async-First

**Decision:** All evaluation APIs are `async`

**Rationale:**
- LLM calls are inherently I/O-bound
- Enables efficient batch evaluation
- Modern Python pattern for network operations
- Better performance with `asyncio.gather()`

**Trade-offs:**
- Requires `async`/`await` knowledge
- More verbose for simple scripts
- No synchronous wrapper (could add later)

**Status:** Implemented ✅

---

## Versioning & Compatibility

### Semantic Versioning

**Decision:** Follow semver (MAJOR.MINOR.PATCH)

**Policy:**
- MAJOR: Breaking API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

**Status:** Currently v0.1.0-alpha

---

### Backward Compatibility

**Decision:** Maintain backward compatibility within major versions

**Rationale:**
- Production users cannot tolerate frequent breaking changes
- Deprecation warnings for 1 minor version before removal
- Serious about being production-grade

**Example:**
- v0.1.0: Introduce feature
- v0.2.0: Deprecate old API (warning), add new API
- v1.0.0: Remove deprecated API

**Status:** Committed policy

---

## Development Workflow

### Type Safety (Strict)

**Decision:** mypy strict mode enforced

**Configuration:**
```toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

**Rationale:**
- Catch bugs at development time, not runtime
- Better IDE support and documentation
- Enforces code quality

**Trade-offs:**
- More verbose code (explicit types)
- Learning curve for contributors
- Occasional need for `cast()` workarounds

**Status:** Implemented ✅

---

### No Placeholders

**Decision:** Never commit TODO comments or incomplete implementations

**Rationale:**
- Production-grade code, not scaffolding
- Clear signal of what's ready vs not
- Forces completion before commit

**Exceptions:** None

**Status:** Enforced (zero TODOs in codebase)

---

## Summary

Arbiter prioritizes:

1. **Simplicity** - 3-line API for common cases
2. **Observability** - Automatic interaction tracking
3. **Production-Ready** - Pooling, retry, middleware built-in
4. **Provider-Agnostic** - No vendor lock-in
5. **Extensibility** - Plugin system for custom evaluators and tools
6. **Type Safety** - Strict mypy, Pydantic models throughout
7. **Quality** - >80% test coverage, no placeholders

These decisions align with the vision: *pragmatic, production-grade LLM evaluation for AI engineers*.

---

## Related Documentation

- **DESIGN_SPEC.md** - Vision, architecture, and competitive analysis
- **ROADMAP.md** - Development timeline and phases
- **AGENTS.md** - Development workflow and patterns
- **PROJECT_TODO.md** - Current milestone tasks
- **README.md** - User-facing overview

---

**Last Updated:** 2025-11-15
