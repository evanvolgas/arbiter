# Arbiter Roadmap

**Current Phase:** Phase 3 - Core Evaluators (Starting Dec 2025)

**Status:** Phase 2.5 Complete | 3 evaluators implemented | v0.1.0-alpha

**Timeline to v1.0:** ~4-5 months

---

## Current Status

**Completed ✅**
- Phase 1: Foundation (Nov 1-14)
- Phase 2: Core Evaluation Engine (Nov 15-21)
- Phase 2.5: Critical Gaps (Nov 22 - Dec 12)

**Active 🚧**
- Phase 3: Core Evaluators (Dec 15 - Jan 5)

**Upcoming ⏳**
- Phase 4: Batch Evaluation (Jan 2026)
- Phase 5: Enhanced Factuality (Feb-Mar 2026)
- Phase 6: Polish & v1.0 Release (Apr 2026)

---

## Completed Phases

### Phase 1: Foundation ✅

**Duration:** 2 weeks

**Delivered:**
- Project structure and tooling (pytest, mypy, ruff, black)
- Core infrastructure (LLMClient, middleware, retry, pooling)
- Exception hierarchy (8 exception types)
- Pydantic models (EvaluationResult, Score, Metric, LLMInteraction)
- Provider-agnostic design (OpenAI, Anthropic, Google, Groq, Mistral, Cohere)

### Phase 2: Core Evaluation Engine ✅

**Duration:** 1 week

**Delivered:**
- `BasePydanticEvaluator` template method pattern
- SemanticEvaluator (similarity scoring)
- `evaluate()` public API with automatic client management
- Automatic LLM interaction tracking (unique feature)
- Middleware integration (logging, metrics, caching, rate limiting)
- 95% test coverage

### Phase 2.5: Critical Gaps ✅

**Duration:** 3 weeks

**Delivered:**
- CustomCriteriaEvaluator (single + multi-criteria modes)
- PairwiseComparisonEvaluator + `compare()` API
- Evaluator registry system
- Multi-evaluator error handling (partial results)
- **FAISS backend for SemanticEvaluator** (optional via `pip install arbiter[scale]`)
  - Significantly faster than LLM-based similarity
  - Zero cost for embeddings (local sentence-transformers)
  - Deterministic results, no Docker/server required
- 10+ example files covering key use cases
- Comprehensive documentation

**Current Evaluators:**
1. SemanticEvaluator - Similarity scoring with pluggable backends (LLM or FAISS)
2. CustomCriteriaEvaluator - Domain-specific criteria
3. PairwiseComparisonEvaluator - A/B testing

---

## Active Phase

### Phase 3: Core Evaluators 🚧

**Duration:** 3 weeks (Dec 15, 2025 - Jan 5, 2026)

**Goal:** Build production-ready evaluators for real-world AI pipelines

**Priority Evaluators:**

#### Week 1: FactualityEvaluator
**Purpose:** Hallucination detection and fact verification

**Capabilities:**
- Extract atomic claims from outputs
- Classify claims (factual, subjective, opinion)
- Verify claims against reference or LLM knowledge
- Score based on factual accuracy

**Output:**
- Overall factuality score
- Lists of factual/non-factual/uncertain claims
- Detailed explanation

#### Week 2: GroundednessEvaluator
**Purpose:** RAG system validation (source attribution)

**Capabilities:**
- Identify statements requiring sources
- Map statements to source documents
- Detect ungrounded claims
- Track citation accuracy

**Output:**
- Groundedness score
- Grounded vs ungrounded statements
- Citation mapping (statement → source)
- Attribution rate

#### Week 3: RelevanceEvaluator
**Purpose:** Query-output alignment assessment

**Capabilities:**
- Analyze query requirements
- Identify addressed vs missing points
- Detect off-topic content
- Score relevance to query

**Output:**
- Relevance score
- Addressed points
- Missing points
- Irrelevant content

**Success Criteria:**
- [ ] 3 evaluators fully implemented
- [ ] Template method pattern followed
- [ ] >80% test coverage each
- [ ] Example file + API docs for each
- [ ] Registered in evaluator registry

---

## Upcoming Phases

### Phase 4: Batch Evaluation

**Duration:** 1 week (Jan 2026)

**Goal:** Production-scale evaluation API

**Deliverables:**
- `batch_evaluate()` API function
- Parallel processing with `asyncio.gather()`
- Progress tracking (optional callback)
- Partial failure handling
- Result aggregation
- Performance benchmarks

**Example:**
```python
results = await batch_evaluate(
    outputs=["output1", "output2", ...],
    evaluators=["semantic", "factuality"],
    progress_callback=lambda p: print(f"{p}% complete")
)
```

---

### Phase 5: Enhanced Factuality

**Duration:** 6 weeks (Feb-Mar 2026)

**Goal:** Production-grade fact verification with external sources

**Motivation:** External verification (web search, knowledge bases) can improve fact-checking accuracy beyond pure LLM-based approaches.

**Components:**

#### 5.1: Plugin Infrastructure (Week 1-2)
- `FactVerificationPlugin` protocol
- TavilyPlugin (real-time web search)
- Plugin registry and priority system
- Cost/latency tracking per plugin

**Expected Benefit:** Improved fact-checking accuracy through external source verification

#### 5.2: Vector Cache for Fact Verification (Week 3-4)
- FAISS-based semantic caching of verified facts
- Cache similar claims using cosine similarity
- TTL-based expiration (configurable, default 7 days)
- Reduced latency on cache hits (no repeated API calls)
- Cost savings on cached verifications (avoid redundant lookups)

**Note:** FAISS backend for SemanticEvaluator already implemented in Phase 2.5

#### 5.3: Atomic Claim Decomposition (Week 5)
- LLM-based claim extraction
- Claim-level verification
- Per-claim source attribution
- Granular hallucination detection

#### 5.4: Additional Plugins (Week 6+)
- WikidataPlugin (structured knowledge, free)
- WikipediaPlugin (encyclopedia, free)
- WolframPlugin (math/science, $0.003/query)
- PubMedPlugin (medical literature, free)
- ArXivPlugin (research papers, free)

**Expected Benefit:** Improved accuracy through multi-source verification and plugin redundancy

**Configuration Example:**
```python
from arbiter import FactualityEvaluator
from arbiter.plugins import TavilyPlugin, WikidataPlugin

evaluator = FactualityEvaluator(
    llm_client,
    plugins=[
        TavilyPlugin(api_key=os.getenv("TAVILY_API_KEY")),
        WikidataPlugin()
    ],
    use_cache=True
)

result = await evaluator.evaluate(
    output="The Eiffel Tower is 330 meters tall",
    reference="The Eiffel Tower is 300 meters tall"
)

# result.value: 0.5 (detected hallucination)
# result.metadata['non_factual_claims']: ["330 meters tall (actual: 300m)"]
# result.metadata['sources']: [{"url": "...", "snippet": "300 metres tall..."}]
```

---

### Phase 6: Polish & v1.0 Release

**Duration:** 2 weeks (Apr 2026)

**Goal:** Production release preparation

**Deliverables:**

#### PyPI Package
- Finalize package metadata
- Publish to PyPI (`pip install arbiter`)
- Version v1.0.0 tag
- Semantic versioning policy

#### CI/CD Pipeline
- GitHub Actions workflows
- Automated testing on PRs
- Type checking (mypy)
- Linting (ruff)
- Coverage reporting
- Release automation

#### Documentation Site
- MkDocs or Sphinx setup
- Auto-generated API reference
- Hosted on GitHub Pages / ReadTheDocs
- Custom domain (optional)

#### Launch Materials
- CHANGELOG.md
- Release notes
- Migration guide
- Announcement

**Success Criteria:**
- [ ] Package on PyPI
- [ ] CI/CD running
- [ ] Documentation site live
- [ ] Release announced

---

## Deferred Features

These features are valuable but deferred to maintain focus on v1.0:

### Milvus Vector Database (→ v2.0+)
**Why Deferred:** Requires Docker/server infrastructure. FAISS provides 90% of benefits with 10% of complexity for v1.0 users.

**Strategic Rationale:**
- FAISS satisfies scale needs for v1.0 (faster than LLM-based, zero cost, simple install)
- Milvus targets enterprise users (1M+ evaluations/day) - defer to v2.0+
- Preserves simplicity for 90% of users while providing scale path

**Planned:** Optional Milvus backend for distributed vector search at massive scale

### Storage Backends (→ v2.0)
**Why Deferred:** Users can persist results themselves. Adds complexity without core value.

**Planned:** MemoryStorage, FileStorage, RedisStorage

### Quality Assurance Tools (→ v2.0)
**Why Deferred:** High complexity (labeled datasets, statistical rigor). Can be community-driven.

**Planned:**
- `validate_evaluator()` - Agreement with human labels
- `check_evaluator_agreement()` - Inter-evaluator correlation
- `test_evaluator_consistency()` - Variance testing
- Calibration analysis tools

### Streaming Support (→ v2.0 or community)
**Why Deferred:** Niche use case with high complexity (ByteWax learning curve).

**Planned:** ByteWax adapter for real-time stream evaluation

### Additional Evaluators (→ community)
**Why Deferred:** Core 6 evaluators cover 80% of use cases. Community can contribute specialized evaluators via registry.

**Ideas:**
- ToxicityEvaluator (Perspective API integration)
- ConsistencyEvaluator (internal contradiction detection)
- CoherenceEvaluator (logical flow assessment)
- BiasEvaluator (demographic bias detection)

---

## Development Principles

**Quality Over Speed:**
- >80% test coverage (currently 95%)
- Mypy strict mode enforced
- No TODO/FIXME comments
- Complete implementations only

**Simplicity First:**
- 3-line API for common cases
- Progressive complexity for advanced use
- Sensible defaults
- Clear error messages

**Production-Ready:**
- Connection pooling
- Automatic retry
- Middleware pipeline
- Comprehensive observability

**Community-Driven:**
- Open source (MIT license)
- Extensible plugin system
- Custom evaluator registry
- Responsive to user feedback

---

## Success Metrics

**Technical:**
- 6 core evaluators implemented
- >80% test coverage maintained
- <200ms framework overhead (p95)
- Enhanced factuality through external verification

**Adoption:**
- PyPI package published
- Documentation site live
- Example integrations (Loom, LangChain)
- Community evaluators contributed

---

## Version History

### v0.1.0-alpha (Current)
**Released:** Nov 2025

**Features:**
- 3 evaluators (semantic, custom_criteria, pairwise)
- Provider-agnostic infrastructure
- Automatic interaction tracking
- Middleware pipeline
- Evaluator registry

**Status:** Alpha - functional but early-stage

### v1.0.0 (Planned)
**Target:** Apr 2026

**Features:**
- 6+ core evaluators
- Enhanced factuality with plugins
- Batch evaluation API
- Production-grade reliability
- Comprehensive documentation
- PyPI distribution

**Status:** In development

---

## Contributing

See **AGENTS.md** for development workflow and **DESIGN_DECISIONS.md** for architectural rationale.

**Current Focus:** Phase 3 - Core Evaluators (Factuality, Groundedness, Relevance)

**Ways to Contribute:**
- Implement evaluators (follow template method pattern)
- Add verification plugins (FactVerificationPlugin protocol)
- Write examples and documentation
- Report bugs and suggest features

---

## Related Documentation

- **DESIGN_SPEC.md** - Vision, architecture, and competitive analysis
- **DESIGN_DECISIONS.md** - Architectural choices and rationale
- **AGENTS.md** - Development workflow and patterns
- **PROJECT_TODO.md** - Current milestone tasks
- **README.md** - User-facing overview

---

**Last Updated:** 2025-11-15 | **Next Milestone:** Phase 3 Week 1 (Dec 15, 2025)
