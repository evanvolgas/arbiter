# Arbiter Project Plan

**Version:** 2.0 (Revised Scope)
**Timeline:** 4-5 months to v1.0 (revised from 7 months)
**Current Status:** Phase 2 Complete, Phase 2.5 Starting
**Last Updated:** 2025-11-12

---

## Overview

This document provides the complete multi-milestone roadmap for Arbiter, from foundation through v1.0 release.

**Goal:** Build production-grade LLM evaluation infrastructure with simple API, complete observability, and provider-agnostic design.

**Strategic Focus:** Play to strengths - simplicity, observability, production-readiness. Don't compete on feature count.

**Key Metrics (Revised):**
- 5-7 evaluators by Month 4 (focused on quality)
- >80% test coverage (maintain current)
- <200ms p95 latency (maintain current)
- Batch evaluation API (production necessity)
- PyPI package published (adoption blocker)
- 100+ GitHub stars by Month 3

---

## Timeline Summary (Revised - Value-First Roadmap)

| Phase | Duration | Status | Completion |
|-------|----------|--------|------------|
| Phase 1: Foundation | 2 weeks | ✅ Done | 100% |
| Phase 2: Core Engine | 1 week | ✅ Done | 100% |
| **Phase 2.5: Critical Gaps** | **2-3 weeks** | **✅ Complete** | **100%** |
| **Phase 3: Core Evaluators** | **3 weeks** | **⏳ Next** | **0%** |
| Phase 4: Batch Evaluation | 1 week | ⏳ Planned | 0% |
| Phase 5: Vector Similarity (Milvus) | 2 weeks | ⏳ Planned | 0% |
| Phase 6: Polish & Release | 2 weeks | ⏳ Planned | 0% |

**Total:** ~3-4 months to v1.0 (optimized)

**Roadmap Change (Nov 15, 2025):** Prioritized Core Evaluators (was Phase 5) before Milvus (now Phase 5) to deliver user value faster. Value-first over infrastructure-first.

**Deferred:**
- ⏸️ Storage Backends (Phase 4 original) - Deferred to Phase 2.0
- ⏸️ Quality Assurance (Phase 6 original) - Deferred to Phase 2.0
- ⏸️ Streaming Support (Phase 7) - Deferred indefinitely

---

## Strategic Scope Decisions (November 2025)

### Core Strengths (Play to These)

1. **Automatic LLM Interaction Tracking** ⭐⭐⭐⭐⭐
   - **Unique in market** - No other framework provides this
   - **High value** - Complete cost/quality visibility
   - **Already implemented** - Working production feature

2. **Simple API + Production Features** ⭐⭐⭐⭐⭐
   - **Rare combination** - Most tools choose one or the other
   - **3-line usage** - Delivered and working
   - **Production-ready** - Middleware, retry, pooling built-in

3. **Provider-Agnostic Design** ⭐⭐⭐⭐
   - **Future-proofs** user investments
   - **Competitive advantage** vs OpenAI Evals
   - **6 providers supported** - OpenAI, Anthropic, Google, Groq, Mistral, Cohere

### Scope Revision Rationale

**What We DON'T Compete On:**
- ❌ **Feature Count** - DeepEval has 15+ evaluators (we focus on 5-7 quality evaluators)
- ❌ **Specialized Integrations** - Streaming, storage backends (adds complexity without core value)
- ❌ **UI/Dashboards** - Infrastructure-focused, not UI-focused

**What We Compete On:**
- ✅ **Simplicity** - 3 lines vs 30+ (LangChain)
- ✅ **Observability** - Automatic tracking (UNIQUE)
- ✅ **Production-Ready** - Built-in retry, pooling, middleware
- ✅ **Provider-Agnostic** - Switch providers easily

**Result:** 4-5 months to v1.0 (vs 7 months) with clearer focus and higher success probability.

### Deferred Features (Detailed Rationale)

**Storage Backends** ⏸️
- **Value:** Low (memory sufficient for MVP)
- **Complexity:** Medium (Redis, File backends require maintenance)
- **User Workaround:** Users can persist results themselves
- **Timeline:** Defer to Phase 2.0 or community

**Quality Assurance Tools** ⏸️
- **Value:** High but not blocking (differentiator for Phase 2.0)
- **Complexity:** High (labeled datasets, statistical rigor)
- **User Workaround:** Manual validation for now
- **Timeline:** Defer to Phase 2.0 or separate project

**Streaming Support** ⏸️
- **Value:** Low-Medium (niche use case)
- **Complexity:** High (ByteWax learning curve, state management)
- **User Workaround:** Manual integration possible
- **Timeline:** Defer indefinitely or community contribution

**Exotic Evaluators** ⏸️
- **Value:** Medium (nice-to-have after core 5-7)
- **Complexity:** Medium per evaluator
- **User Workaround:** Registry enables custom evaluators
- **Timeline:** Community contributions or Phase 2.0

---

## Milestone Details

## Phase 1: Foundation ✅ COMPLETE

**Duration:** 2 weeks (Nov 1-14, 2025)
**Status:** ✅ 100% Complete
**Goal:** Project setup and core infrastructure

### Objectives
1. Set up project structure
2. Configure dependencies and tooling
3. Establish development workflow
4. Create initial documentation

### Deliverables

#### ✅ Project Setup
- [x] Create project structure (arbiter/, tests/, examples/)
- [x] Initialize git repository
- [x] Configure pyproject.toml with dependencies
- [x] Create package structure (__init__.py files)
- [x] Add supporting files (README, LICENSE, .gitignore)
- [x] Initial commit

#### ✅ Core Infrastructure
- [x] Middleware system (from Sifaka)
  - LoggingMiddleware
  - MetricsMiddleware
  - CachingMiddleware
  - RateLimitingMiddleware
- [x] LLM client infrastructure
  - core/llm_client.py (provider-agnostic)
  - core/llm_client_pool.py (connection pooling)
- [x] Retry logic with exponential backoff
- [x] Core models (EvaluationResult, Score, Metric, LLMInteraction)
- [x] Exception hierarchy (8 exception types)
- [x] Type definitions and protocols

### Success Criteria
- ✅ Project builds successfully
- ✅ All type checks pass (mypy strict)
- ✅ Git repository initialized
- ✅ Dependencies installed

### Lessons Learned
- Strong foundation enables rapid development
- PydanticAI integration smooth
- Early type safety prevents later issues

---

## Phase 2: Core Evaluation Engine ✅ COMPLETE

**Duration:** 1 week (Nov 15-21, 2025)
**Status:** ✅ 100% Complete + Critical Fixes
**Goal:** Working evaluation with SemanticEvaluator

### Objectives
1. Implement evaluation flow
2. Create PydanticAI evaluator base class
3. Build SemanticEvaluator
4. Design public API
5. Automatic LLM interaction tracking

### Deliverables

#### ✅ Evaluation Engine
- [x] BaseEvaluator protocol (core/interfaces.py)
- [x] BasePydanticEvaluator template method class
  - _get_system_prompt()
  - _get_user_prompt()
  - _get_response_type()
  - _compute_score()
- [x] Automatic LLM interaction tracking
- [x] Score aggregation and confidence calculation

#### ✅ SemanticEvaluator
- [x] Implementation in evaluators/semantic.py
- [x] SemanticResponse model (score, confidence, explanation, similarities/differences)
- [x] Handles 3 modes: with reference, criteria only, standalone

#### ✅ Public API
- [x] evaluate() function in api.py
- [x] Automatic client management
- [x] Input validation
- [x] Error handling
- [x] Middleware integration

#### ✅ Critical Fixes (completed during Phase 2)
- [x] Token counting fixed (extract from PydanticAI)
- [x] Middleware integrated into evaluate()
- [x] Input validation added

### Success Criteria
- ✅ Can evaluate text with simple API
- ✅ SemanticEvaluator returns structured scores
- ✅ Complete LLM interaction tracking works
- ✅ Example code demonstrates usage
- ✅ All tests pass

### Key Achievements
- **Automatic interaction tracking** - Novel feature, well-implemented
- **Simple 3-line API** - Competitive with best-in-class
- **Template method pattern** - Forces consistency

### Gaps Identified
- Only one evaluator (semantic)
- No custom criteria support
- No comparison mode
- Limited documentation

---

## Phase 2.5: Fill Critical Gaps 🚧 CURRENT

**Duration:** 2-3 weeks (Nov 22 - Dec 12, 2025)
**Status:** 🚧 0% Complete (Starting)
**Priority:** 🔴 CRITICAL
**Goal:** Make Phase 2 production-ready with key missing features

### Objectives
1. Add CustomCriteriaEvaluator (highest ROI)
2. Add PairwiseComparisonEvaluator (A/B testing)
3. Improve multi-evaluator error handling
4. Expand documentation significantly

### Deliverables

#### 🔴 CustomCriteriaEvaluator (2-3 days)
**Priority:** HIGHEST
**Why:** Multiplies framework usefulness immediately

**Features:**
- Single criteria evaluation
- Multi-criteria evaluation (dict input)
- Flexible prompt templates
- Domain-specific criteria support

**Example:**
```python
result = await evaluate(
    output="Medical advice...",
    criteria="Medical accuracy, HIPAA compliance, appropriate tone",
    evaluators=["custom_criteria"],
    model="gpt-4o"
)
```

**Deliverables:**
- [ ] CustomCriteriaEvaluator class
- [ ] CustomCriteriaResponse model
- [ ] Single criteria mode
- [ ] Multi-criteria mode (returns multiple scores)
- [ ] Tests (>80% coverage)
- [ ] Example code
- [ ] Documentation

#### 🔴 PairwiseComparisonEvaluator (2-3 days)
**Priority:** HIGHEST
**Why:** Essential for A/B testing and model comparison

**Features:**
- Compare two outputs
- Aspect-level comparison
- Winner determination
- Confidence scoring

**Example:**
```python
comparison = await compare(
    output_a="GPT-4 response",
    output_b="Claude response",
    criteria="helpfulness, accuracy, clarity",
    reference="User question"
)
print(f"Winner: {comparison.winner}")
```

**Deliverables:**
- [ ] PairwiseComparisonEvaluator class
- [ ] ComparisonResult model
- [ ] compare() API function
- [ ] Aspect-level scoring
- [ ] Tests (>80% coverage)
- [ ] Example code
- [ ] Documentation

#### 🟡 Multi-Evaluator Error Handling (1 day)
**Priority:** MEDIUM

**Features:**
- Partial results when one evaluator fails
- Clear error messages
- Error tracking in result
- Graceful degradation

**Deliverables:**
- [ ] Partial result support
- [ ] Error field in EvaluationResult
- [ ] Error handling tests
- [ ] Documentation

#### 🟡 Evaluator Registry & Validation (1 day)
**Priority:** MEDIUM

**Features:**
- Registry of available evaluators
- Validate evaluator names early
- Better error messages
- Type-safe evaluator names (Literal)

**Deliverables:**
- [ ] AVAILABLE_EVALUATORS registry
- [ ] Validation in evaluate()
- [ ] Type hints (Literal["semantic", "custom_criteria", ...])
- [ ] Tests

#### 🟢 Documentation & Examples (5-7 days)
**Priority:** HIGH

**Deliverables:**
- [ ] 10-15 usage examples
  - Basic evaluation
  - Custom criteria
  - Pairwise comparison
  - Multiple evaluators
  - Middleware usage
  - Cost tracking
  - Error handling
  - Batch (manual loop)
  - Provider switching
  - Advanced configuration
- [ ] API reference documentation
- [ ] Integration guides
- [ ] Performance best practices
- [ ] Troubleshooting guide

### Success Criteria
- ✅ CustomCriteriaEvaluator working with tests
- ✅ PairwiseComparisonEvaluator working with compare() API
- ✅ 10+ examples covering key use cases
- ✅ Comprehensive API documentation
- ✅ Error handling tested and robust

### Risks
- **Scope creep:** Focus on must-haves only
- **Documentation time:** May take longer than estimated

### Dependencies
- None (can start immediately)

---

## Phase 3: Core Evaluators ⏳ NEXT (Moved from Phase 5)

**Duration:** 3 weeks (Dec 15, 2025 - Jan 5, 2026)
**Status:** ⏳ Next - Ready to Start
**Goal:** Build production-ready evaluators for real-world use cases
**Priority:** 🔴 HIGHEST - Delivers immediate user value

**Strategic Focus:** Focus on TOP 3 high-impact evaluators that make Arbiter production-ready.

### Why This Matters

Value-first roadmap prioritization:
- **Immediate Value:** Users need evaluators NOW, not infrastructure later
- **Loom Unblocked:** Enables Loom integration with robust evaluation
- **Production Ready:** Factuality, Groundedness, Relevance cover 80% of use cases
- **Faster v1.0:** 3 weeks vs 4-5 weeks (focused scope)

### Objectives
1. Implement **top 3** production evaluators
2. Ensure high quality and reliability (>80% coverage)
3. Comprehensive examples and documentation

### Deliverables

#### Week 1: FactualityEvaluator (HIGHEST PRIORITY)
- [ ] **FactualityEvaluator** implementation
  - LLM-based claim extraction and verification
  - Verification against reference text
  - Output: score, confidence, factual_claims, non_factual_claims, uncertain_claims
- [ ] Comprehensive tests (>80% coverage)
- [ ] Example: examples/factuality_example.py
- [ ] Documentation in docs/api/evaluators/factuality.md

**Why First:** Hallucination detection is critical for all LLM outputs

#### Week 2: GroundednessEvaluator (RAG-SPECIFIC)
- [ ] **GroundednessEvaluator** implementation
  - Source attribution checking for RAG systems
  - Unsupported claims detection
  - Citation mapping
  - Output: score, confidence, grounded_statements, ungrounded_statements, citations
- [ ] Comprehensive tests (>80% coverage)
- [ ] Example: examples/groundedness_example.py
- [ ] Documentation in docs/api/evaluators/groundedness.md

**Why Second:** RAG is the most common LLM pattern - needs validation

#### Week 3: RelevanceEvaluator (QUERY ALIGNMENT)
- [ ] **RelevanceEvaluator** implementation
  - Query-output alignment assessment
  - Addressed vs. missed aspects detection
  - Off-topic content detection
  - Output: score, confidence, addressed_points, missing_points, irrelevant_content
- [ ] Comprehensive tests (>80% coverage)
- [ ] Example: examples/relevance_example.py
- [ ] Documentation in docs/api/evaluators/relevance.md

**Why Third:** Ensures outputs stay on-topic and answer the actual question

### Success Criteria
- ✅ 3 evaluators fully implemented
- ✅ Each evaluator >80% test coverage
- ✅ Complete API documentation for each
- ✅ Working examples for each use case
- ✅ All following template method pattern
- ✅ Registered in evaluator registry
- ✅ Updated README with new evaluators

### Risks & Mitigation
- **Complexity variation:** Factuality is hardest → Start with it to learn
- **Prompt engineering:** LLM evaluations need good prompts → Iterate based on testing
- **Time estimates:** 1 week per evaluator is tight → Focus on MVP, defer nice-to-haves

### Dependencies
- Phase 2.5 complete ✅ (CustomCriteria provides template pattern)
- Template method pattern established ✅
- Evaluator registry working ✅

### Deferred Evaluators (Phase 2.0 or Community)
- ⏸️ **ToxicityEvaluator** - Requires Perspective API integration
- ⏸️ **ConsistencyEvaluator** - Lower priority, can be community-built
- ⏸️ **ContextRelevanceEvaluator** - Overlaps with RelevanceEvaluator

---

## Phase 4: Batch Evaluation ⏳ PLANNED (Revised Scope)

**Duration:** 1 week (reduced from 2 weeks)
**Status:** ⏳ Planned
**Goal:** Batch evaluation API for production scale

**Note:** Storage backends deferred to Phase 2.0 or community. Focus on batch evaluation only.

### Objectives
1. Add batch evaluation API
2. Parallel processing with asyncio.gather
3. Error handling for partial failures
4. Progress tracking

### Deliverables

#### Batch Operations
- [ ] batch_evaluate() API function
- [ ] Parallel processing of multiple outputs (asyncio.gather)
- [ ] Progress tracking (optional callback)
- [ ] Error handling for partial failures
- [ ] Result aggregation
- [ ] Example: batch_evaluation_example.py

#### ⏸️ DEFERRED: Storage Backends
- ⏸️ MemoryStorage (deferred to Phase 2.0)
- ⏸️ FileStorage (deferred to Phase 2.0)
- ⏸️ RedisStorage (deferred to Phase 2.0)
- ⏸️ Result serialization (users can handle)

**Rationale:** Memory sufficient for MVP. Users can persist results themselves. Reduces complexity.

### Success Criteria
- ✅ batch_evaluate() handles 100+ items
- ✅ Parallel processing faster than sequential
- ✅ Partial failure handling tested
- ✅ Example demonstrating batch usage

### Risks
- **Batch complexity:** Error handling edge cases
- **Memory usage:** Large batches may require optimization

### Dependencies
- Phase 2.5 complete
- Phase 3 optional (can parallelize)

---

## Phase 5: Vector Similarity (Milvus) ⏳ PLANNED (Moved from Phase 3)

**Duration:** 2 weeks (Jan 6-19, 2026)
**Status:** ⏳ Planned - Deferred for value-first approach
**Goal:** Milvus integration for vector-based semantic comparison
**Priority:** 🟡 MEDIUM - Nice-to-have optimization

**Strategic Context:** Deferred from Phase 3 because core evaluators provide more immediate user value.

### Why Deferred

Infrastructure-second prioritization:
- **Core evaluators more critical:** Users need Factuality/Groundedness/Relevance first
- **Current semantic works:** Existing SemanticEvaluator is functional without vectors
- **Optimization not blocker:** Vector similarity is performance enhancement, not core capability
- **Can be optional:** Make Milvus opt-in for users who need it

### Objectives
1. Set up Milvus vector database (optional dependency)
2. Implement embedding generation pipeline
3. Build vector similarity scoring
4. Hybrid evaluation (LLM + vector similarity)

### Deliverables

#### Vector Storage
- [ ] Milvus client (core/milvus_client.py)
- [ ] Embedding generation (core/embeddings.py)
  - OpenAI embeddings
  - Sentence transformers
  - Custom embedding models
- [ ] Vector storage schema design
- [ ] Collection management

#### Enhanced SemanticEvaluator
- [ ] Hybrid mode (LLM + vector similarity)
- [ ] Vector-only mode (faster, cheaper)
- [ ] Embedding caching
- [ ] Similarity threshold configuration
- [ ] Graceful fallback when Milvus unavailable

#### Infrastructure
- [ ] Milvus connection pooling
- [ ] Embedding cache
- [ ] Vector search optimization
- [ ] Optional dependency (pymilvus not required by default)

### Success Criteria
- ✅ Milvus integration is **optional** (doesn't break without it)
- ✅ Embeddings generated and stored when available
- ✅ Vector similarity scoring works
- ✅ Hybrid evaluation faster than LLM-only
- ✅ Tests for Milvus integration
- ✅ Clear docs on when to use Milvus

### Risks
- **Milvus complexity:** May require Docker setup → Mitigation: Make optional
- **Performance:** Embedding generation adds latency → Mitigation: Cache aggressively
- **Storage:** Vector storage size management → Mitigation: Configurable retention

### Dependencies
- Phase 3 complete (Core Evaluators)
- Phase 4 optional (Batch evaluation can happen in parallel)

---

## Phase 6: Polish & Release ⏳ PLANNED (Moved Up)

**Duration:** 2 weeks
**Status:** ⏳ Planned
**Goal:** Production release preparation

**Note:** Moved up from Phase 7-9 to enable earlier adoption. Critical for community growth.

### Objectives
1. Publish to PyPI
2. Set up CI/CD pipeline
3. Create documentation site
4. Prepare release materials

### Deliverables

#### PyPI Package
- [ ] Finalize package metadata
- [ ] Create PyPI account
- [ ] Write release checklist
- [ ] Publish 0.1.0-alpha

#### CI/CD Pipeline
- [ ] GitHub Actions workflow
- [ ] Automated testing
- [ ] Code quality checks
- [ ] Release automation

#### Documentation Site
- [ ] Set up MkDocs or Sphinx
- [ ] Auto-generate API reference
- [ ] Host on GitHub Pages or ReadTheDocs
- [ ] Link from README

#### Release Materials
- [ ] CHANGELOG.md
- [ ] Release notes
- [ ] Migration guide (if needed)
- [ ] Announcement post

### Success Criteria
- [ ] Package on PyPI (pip install arbiter works) - NOT YET AVAILABLE
- [ ] CI/CD pipeline running
- [ ] Documentation site live
- [ ] Release checklist complete

### Dependencies
- Phase 5 complete (evaluators done)

---

## ⏸️ DEFERRED Phases

### Phase 6 (Original): Evaluation Quality Assurance ⏸️ DEFERRED

**Duration:** 2-3 weeks
**Status:** ⏸️ Deferred to Phase 2.0 or community
**Goal:** Tools to validate evaluator quality

**Rationale:**
- High complexity (labeled datasets, statistical rigor)
- Can be community-driven
- Not blocking for v1.0
- **New Timeline:** Phase 2.0 or separate project

### Objectives
1. Build evaluator validation tools
2. Create calibration metrics
3. Measure agreement with human judgment
4. Test consistency and robustness

### Deliverables

#### Validation Tools
- [ ] validate_evaluator() - Agreement with human labels
  - Kendall tau correlation
  - Spearman correlation
  - Accuracy metrics
  - Bias analysis

- [ ] check_evaluator_agreement() - Inter-evaluator agreement
  - Compare multiple evaluators
  - Identify discrepancies
  - Consensus scoring

- [ ] test_evaluator_consistency() - Consistency testing
  - Same input multiple times
  - Measure variance
  - Confidence calibration

- [ ] adversarial_test() - Robustness testing
  - Typos
  - Reordering
  - Paraphrasing
  - Perturbation resilience

- [ ] calibrate_evaluator() - Calibration analysis
  - Confidence accuracy
  - Over/under-confidence detection
  - Calibration curves

#### Documentation
- [ ] Quality assurance guide
- [ ] Validation best practices
- [ ] Calibration datasets
- [ ] Benchmark results

### Success Criteria
- ✅ 5 validation tools implemented
- ✅ Human agreement >0.75 for core evaluators
- ✅ Consistency variance <0.1
- ✅ Calibration documented
- ✅ Published benchmarks

### Risks
- **Human labeling:** Need labeled datasets
- **Metrics complexity:** Statistical rigor required

### Dependencies
- Phase 5 complete (need evaluators to validate)

---

### Phase 7: Streaming Support ⏸️ DEFERRED

**Duration:** 1 week
**Status:** ⏸️ Deferred indefinitely
**Goal:** ByteWax integration for streaming data

**Rationale:**
- Niche use case (most users don't need it)
- High complexity (ByteWax learning curve)
- Users can integrate manually
- **New Timeline:** Indefinite defer or community contribution

**Original Plan:** ByteWax adapter, streaming examples, Kafka integration

---

### Phase 8: Testing & Documentation ⏸️ DEFERRED

**Duration:** 2 weeks
**Status:** ⏸️ Deferred (integrated into Phase 6)
**Goal:** Comprehensive testing and documentation

**Rationale:**
- Testing already at >80% coverage (maintain current)
- Documentation integrated into Phase 6 (Polish & Release)
- **New Timeline:** Integrated into Phase 6

**Original Plan:** Unit/integration tests, API reference, user guides, architecture docs

---

### Phase 9 (Original): Polish & Release ⏸️ MOVED UP

**Duration:** 2 weeks
**Status:** ⏸️ Moved up to Phase 6
**Goal:** Prepare for v1.0 release

**Rationale:**
- Critical for adoption (PyPI, CI/CD, docs)
- Should happen sooner, not later
- **New Timeline:** Phase 6 (moved up from Phase 9)

### Objectives
1. Set up CI/CD
2. Package for PyPI
3. Create documentation site
4. Announce release

### Deliverables

#### CI/CD
- [ ] GitHub Actions workflows
  - Test automation
  - Type checking
  - Linting
  - Coverage reporting
- [ ] Pre-commit hooks
  - Black formatting
  - Ruff linting
  - Mypy type checking

#### Package
- [ ] Build wheels
- [ ] Test installation
- [ ] PyPI upload
- [ ] Version tagging

#### Documentation Site
- [ ] MkDocs configuration
- [ ] Deploy to GitHub Pages
- [ ] Custom domain (optional)

#### Launch
- [ ] Blog post
- [ ] Reddit/HN announcement
- [ ] Twitter launch
- [ ] Documentation push

### Success Criteria
- ✅ CI/CD running
- ✅ Package on PyPI
- ✅ Docs site live
- ✅ Launch announced

### Risks
- **PyPI issues:** First-time package upload
- **Documentation site:** May need custom theme

### Dependencies
- All phases complete

---

## Resource Requirements

### Development Time

| Phase | Duration | FTE |
|-------|----------|-----|
| Phase 1-2 | 3 weeks | 1 |
| Phase 2.5 | 2-3 weeks | 1 |
| Phase 3 | 2 weeks | 1 |
| Phase 4 | 2 weeks | 1 |
| Phase 5 | 6 weeks | 1 |
| Phase 6 | 2-3 weeks | 1 |
| Phase 7 | 1 week | 0.5 |
| Phase 8 | 2 weeks | 1 |
| Phase 9 | 2 weeks | 0.5 |
| **Total** | **~7 months** | **1 FTE** |

### Infrastructure

**Development:**
- GitHub repository (free)
- Local development environment

**Testing:**
- LLM API credits ($100-500/month)
  - OpenAI GPT-4o-mini for testing
  - Multiple providers for validation
- Milvus (local Docker or cloud)
  - Vector storage
  - ~10GB for development

**Production (Post-Release):**
- Documentation hosting (GitHub Pages - free)
- Optional: Redis for storage backend
- Optional: ByteWax for streaming

### Estimated Costs

**Development Phase (7 months):**
- LLM API credits: $500-2000
- Infrastructure: $0-500 (if using cloud Milvus/Redis)
- **Total:** $500-2500

**Post-Release (Monthly):**
- Hosting: $0 (GitHub Pages)
- LLM credits (testing): $50-200
- Infrastructure (optional): $0-100

---

## Risk Management

### High-Priority Risks

#### 1. Evaluator Coverage Gap 🔴
**Risk:** Only semantic evaluator blocks adoption
**Impact:** HIGH
**Mitigation:** Phase 2.5 adds CustomCriteria (covers long tail)
**Status:** Being addressed

#### 2. Documentation Gap 🟡
**Risk:** Poor docs slow adoption
**Impact:** MEDIUM
**Mitigation:** Phase 2.5 adds 10+ examples
**Status:** Being addressed

#### 3. PydanticAI Breaking Changes 🟡
**Risk:** New library, may have breaking changes
**Impact:** MEDIUM
**Mitigation:** Abstract behind interfaces
**Status:** Monitored

### Medium-Priority Risks

#### 4. LLM Cost at Scale 🟡
**Risk:** Expensive with GPT-4
**Impact:** MEDIUM
**Mitigation:** Caching, cheaper models, batch ops
**Status:** Mitigated (caching exists)

#### 5. Competition 🟡
**Risk:** Established players have network effects
**Impact:** MEDIUM
**Mitigation:** Focus on observability + production
**Status:** Differentiation clear

### Low-Priority Risks

#### 6. Milvus Complexity 🟢
**Risk:** Vector DB adds complexity
**Impact:** LOW
**Mitigation:** Make optional, good docs
**Status:** Contained to Phase 3

---

## Decision Log

### Major Architectural Decisions

| Date | Decision | Rationale | Trade-offs |
|------|----------|-----------|------------|
| 2025-11-01 | Use PydanticAI | Structured outputs, type safety | Dependency on new library |
| 2025-11-01 | Provider-agnostic | Future-proof, flexibility | More complexity than OpenAI-only |
| 2025-11-15 | Template method pattern | Consistency, automatic tracking | Less flexibility for unusual evaluators |
| 2025-11-15 | Automatic interaction tracking | Complete observability | Slight performance overhead |
| 2025-11-15 | LLM-as-judge | Nuanced evaluation | Cost and latency |

### Implementation Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-15 | Fix token counting in Phase 2 | Critical for cost tracking |
| 2025-11-15 | Integrate middleware in Phase 2 | Core feature, needed immediately |
| 2025-11-22 | Add Phase 2.5 | Address critical gaps before Phase 3 |
| 2025-11-22 | CustomCriteria highest priority | Multiplies usefulness immediately |

---

## Success Metrics & Tracking

### Technical Metrics

| Metric | Target | Current | Phase |
|--------|--------|---------|-------|
| Evaluator Count | 8+ | 1 | Phase 5 |
| Test Coverage | >80% | TBD | Phase 8 |
| p95 Latency | <200ms | TBD | Phase 8 |
| Human Agreement | >0.85 | TBD | Phase 6 |

### Adoption Metrics

| Metric | Target | Deadline | Status |
|--------|--------|----------|--------|
| GitHub Stars | 100+ | Month 3 (Feb 2026) | 0 |
| Production Users | 10+ | Month 6 (May 2026) | 0 |
| Community Evaluators | 5+ | Month 9 (Aug 2026) | 0 |
| Framework Integrations | 2+ | Month 6 (May 2026) | 0 |

### Quality Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Documentation Coverage | 100% public APIs | TBD |
| Performance Benchmarks | Published | TBD |
| Calibration Data | Available | TBD |
| Community Engagement | Active (issues, PRs) | TBD |

---

## Next Steps

### Immediate (This Week)
1. ✅ Start Phase 2.5
2. 🚧 Implement CustomCriteriaEvaluator
3. ⏳ Design PairwiseComparisonEvaluator

### Short-Term (Next Month)
1. Complete Phase 2.5 (all tasks)
2. Start Phase 3 (Milvus integration)
3. Build 3 core evaluators (Factuality, Relevance, Toxicity)

### Medium-Term (Months 2-3)
1. Complete Phase 3-4
2. Complete Phase 5 (all evaluators)
3. Start Phase 6 (quality assurance)

### Long-Term (Months 4-7)
1. Complete Phase 6-9
2. Launch v1.0
3. Build community

---

## Appendix

### Related Documents
- **DESIGN_SPEC.md** - Vision and architecture
- **AGENTS.md** - How to work with this repo
- **PROJECT_TODO.md** - Current milestone tracker
- **PHASE2_REVIEW.md** - Phase 2 assessment
- **EVALUATOR_RECOMMENDATIONS.md** - Evaluator priorities

### External References
- [PydanticAI](https://ai.pydantic.dev/)
- [Milvus](https://milvus.io/)
- [LangChain Evaluators](https://python.langchain.com/docs/guides/evaluation/)
- [OpenAI Evals](https://github.com/openai/evals)

### Version History
- **v1.0** (2025-11-12) - Initial project plan

---

**End of Project Plan**
