# FactualityEvaluator Enhancement Plan

**Version:** 1.0.0
**Date:** 2025-11-15
**Status:** Approved for Implementation

---

## Executive Summary

Transform FactualityEvaluator from basic LLM-based verification (70-80% accuracy) to production-grade external fact verification system (90-98% accuracy) through plugin architecture and vector caching.

**Impact on Loom**: Loom pipelines can use enhanced FactualityEvaluator in quality gates for hallucination detection with citations.

### Current State (v0.1.0)

```python
# Pure LLM-based verification
evaluator = FactualityEvaluator(llm_client)
result = await evaluator.evaluate(
    output="The Eiffel Tower is 330 meters tall",
    reference="The Eiffel Tower is 300 meters tall"
)
# result.value: 0.67 (detects hallucination but no citations)
```

**Limitations:**
- 70-80% accuracy (depends on LLM knowledge cutoff)
- No external verification or citations
- Cannot verify recent facts or real-time data
- No source attribution for claims

### Proposed State (v0.2.0+)

```python
# External verification with citations
from arbiter import FactualityEvaluator
from arbiter.plugins import TavilyPlugin

evaluator = FactualityEvaluator(
    llm_client,
    plugins=[TavilyPlugin()],
    use_cache=True
)

result = await evaluator.evaluate(
    output="The Eiffel Tower is 330 meters tall",
    reference="The Eiffel Tower is 300 meters tall"
)

# result.value: 0.50
# result.metadata['non_factual_claims']: ["330 meters tall (actual: 300 meters)"]
# result.metadata['sources']: [
#     {"url": "wikipedia.org/Eiffel_Tower", "snippet": "300 metres (984 ft) tall..."}
# ]
```

**Benefits:**
- 90-98% accuracy (real-time web + structured knowledge verification)
- Full source citations with URLs and snippets
- Claim-level attribution and verification
- Recent/real-time fact checking via Tavily web search

---

## Architecture: Plugin System + Vector Cache

### Core Protocol

```python
# arbiter/core/interfaces.py
from typing import Protocol, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class Source(BaseModel):
    """External source for fact verification."""
    url: str
    title: str
    snippet: str = Field(max_length=500)
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.8)
    published_date: Optional[datetime] = None

class VerificationResult(BaseModel):
    """Result of external fact verification."""
    status: Literal["verified", "refuted", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cost_usd: float = Field(default=0.0)

class FactVerificationPlugin(Protocol):
    """Protocol for pluggable fact verification sources."""

    async def verify_claim(
        self,
        claim: str,
        context: Optional[str] = None
    ) -> VerificationResult:
        """Verify a single atomic claim.

        Args:
            claim: Atomic factual claim to verify
            context: Optional context for disambiguation

        Returns:
            VerificationResult with status, confidence, evidence, sources
        """
        ...

    @property
    def name(self) -> str:
        """Plugin identifier (e.g., 'tavily', 'wikidata')."""
        ...

    @property
    def cost_per_verification(self) -> float:
        """Estimated cost per claim verification in USD."""
        ...
```

### Enhanced Evaluator

```python
# arbiter/evaluators/factuality.py (enhanced)
class FactualityEvaluator(BasePydanticEvaluator):
    """Evaluates factual accuracy with optional external verification.

    Supports three modes:
    1. Pure LLM (default, backward compatible)
    2. LLM + plugins (external verification with citations)
    3. LLM + plugins + cache (production-grade performance)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        plugins: Optional[List[FactVerificationPlugin]] = None,
        use_cache: bool = False,
        cache_backend: Optional[VectorDB] = None,
        plugin_weight: float = 0.7  # Trust plugins more than pure LLM
    ):
        super().__init__(llm_client)
        self.plugins = plugins or []  # Default: pure LLM mode
        self.cache = FactCache(cache_backend) if use_cache else None
        self.plugin_weight = plugin_weight

    async def _verify_claim_with_plugins(
        self,
        claim: str,
        context: Optional[str] = None
    ) -> VerificationResult:
        """Verify claim using plugins with caching."""

        # Check cache first
        if self.cache:
            cached = await self.cache.get(claim)
            if cached:
                return cached

        # Verify with all plugins in parallel
        if self.plugins:
            results = await asyncio.gather(*[
                plugin.verify_claim(claim, context)
                for plugin in self.plugins
            ])

            # Aggregate results (majority vote weighted by confidence)
            aggregated = self._aggregate_verifications(results)

            # Cache result
            if self.cache:
                await self.cache.set(claim, aggregated)

            return aggregated

        # Fallback: LLM-only verification
        return await self._verify_with_llm(claim, context)
```

---

## Implementation Roadmap

### Phase 1: Plugin Infrastructure (Week 1-2) ⚡ HIGH PRIORITY

**Goal**: Add Tavily plugin for real-time web verification

**Why Tavily First?**
- Already available as MCP server
- Best cost/accuracy ratio ($0.001 per search)
- 90-95% accuracy with citations
- Real-time web search capability

**Implementation:**

1. **Create Plugin Infrastructure**
   - File: `arbiter/plugins/__init__.py`
   - File: `arbiter/plugins/base.py` (FactVerificationPlugin protocol)
   - File: `arbiter/plugins/tavily.py` (TavilyPlugin implementation)

2. **TavilyPlugin Implementation**
```python
# arbiter/plugins/tavily.py
class TavilyPlugin:
    """Real-time web search verification using Tavily API."""

    def __init__(self, max_results: int = 5, api_key: Optional[str] = None):
        self.max_results = max_results
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    async def verify_claim(
        self,
        claim: str,
        context: Optional[str] = None
    ) -> VerificationResult:
        """Verify claim using Tavily web search."""

        # Search query optimization
        query = f'verify fact: "{claim}"'
        if context:
            query += f" context: {context[:100]}"

        # Call Tavily API (via MCP server or direct API)
        results = await self._search_tavily(query)

        # Extract sources
        sources = [
            Source(
                url=r["url"],
                title=r["title"],
                snippet=r["content"][:200],
                credibility_score=r.get("score", 0.8)
            )
            for r in results["results"][:self.max_results]
        ]

        # Determine verification status
        if self._claim_supported_by_sources(claim, sources):
            return VerificationResult(
                status="verified",
                confidence=0.9,
                evidence=[s.snippet for s in sources[:3]],
                sources=sources,
                reasoning=f"Claim verified across {len(sources)} independent web sources",
                cost_usd=0.001
            )
        elif self._claim_contradicted_by_sources(claim, sources):
            return VerificationResult(
                status="refuted",
                confidence=0.85,
                evidence=[s.snippet for s in sources[:3]],
                sources=sources,
                reasoning=f"Claim contradicted by {len(sources)} sources",
                cost_usd=0.001
            )
        else:
            return VerificationResult(
                status="uncertain",
                confidence=0.6,
                sources=sources,
                reasoning="Insufficient evidence to verify or refute claim",
                cost_usd=0.001
            )

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def cost_per_verification(self) -> float:
        return 0.001  # $0.001 per Tavily search
```

3. **Update FactualityEvaluator**
   - Add `plugins` parameter to `__init__()`
   - Implement `_verify_claim_with_plugins()` method
   - Update `_compute_score()` to include sources in metadata
   - Maintain backward compatibility (plugins=None → pure LLM mode)

4. **Testing**
   - Unit tests with mocked Tavily responses
   - Integration tests with real Tavily API (optional, gated by API key)
   - Test backward compatibility (pure LLM mode still works)

**Deliverables:**
- [ ] `arbiter/plugins/` directory structure
- [ ] `FactVerificationPlugin` protocol
- [ ] `TavilyPlugin` implementation
- [ ] Enhanced `FactualityEvaluator` with plugins support
- [ ] Unit tests (mocked Tavily responses)
- [ ] Integration tests (real Tavily API)
- [ ] Example: `examples/factuality_with_tavily.py`
- [ ] Documentation update

**Success Metrics:**
- Accuracy: 70-80% → 90-95% with Tavily
- Cost: $0.01 → $0.015 per evaluation (50% increase)
- Latency: 500ms → 1500ms (3x slower, acceptable for quality gain)
- Citations: 0 → 3-5 sources per evaluation

---

### Phase 2: Vector Cache (Week 3-4) 🚄 PERFORMANCE

**Goal**: Add Milvus vector cache for fact verification results

**Why Cache?**
- 30x speedup: 1500ms → 50ms on cache hits
- 90% cost reduction: $0.015 → $0.0015 for cached claims
- Critical for production scale

**Implementation:**

1. **FactCache with Milvus**
```python
# arbiter/core/cache.py
from pymilvus import MilvusClient, DataType

class FactCache:
    """Vector-based fact verification cache using Milvus."""

    def __init__(self, milvus_uri: str = "memory"):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = "fact_verifications"
        self._ensure_collection()

    async def get(self, claim: str) -> Optional[VerificationResult]:
        """Retrieve cached verification by semantic similarity."""

        # Embed claim
        embedding = await self._embed(claim)

        # Search for similar claims (>0.95 cosine similarity)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=1,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )

        if results and results[0].distance > 0.95:
            cached_data = results[0].entity.get("verification")

            # Check if expired
            expires_at = datetime.fromisoformat(cached_data["expires_at"])
            if datetime.utcnow() < expires_at:
                return VerificationResult(**cached_data["result"])

        return None

    async def set(
        self,
        claim: str,
        result: VerificationResult,
        ttl_days: int = 7
    ):
        """Cache verification result with TTL."""

        embedding = await self._embed(claim)

        self.client.insert(
            collection_name=self.collection_name,
            data={
                "embedding": embedding,
                "claim": claim,
                "verification": {
                    "result": result.model_dump(),
                    "expires_at": (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
                }
            }
        )
```

2. **Update FactualityEvaluator**
   - Add `use_cache` and `cache_backend` parameters
   - Check cache before plugin calls
   - Cache plugin results after verification

3. **Testing**
   - Test cache hit/miss logic
   - Test semantic similarity threshold
   - Test TTL expiration
   - Benchmark performance improvement

**Deliverables:**
- [ ] `arbiter/core/cache.py` with FactCache implementation
- [ ] Milvus integration
- [ ] Cache hit/miss tracking in metadata
- [ ] Performance benchmarks
- [ ] Documentation

**Success Metrics:**
- Cache hit rate: Target 50-80% after warmup
- Latency on hit: 1500ms → 50ms (30x improvement)
- Cost on hit: $0.015 → $0.0015 (90% reduction)
- Overall cost at 50% hit: $0.015 → $0.008 (47% reduction)

---

### Phase 3: Atomic Claim Decomposition (Week 5) 🔬 PRECISION

**Goal**: Break outputs into atomic claims for granular verification

**Why Atomic Claims?**
- Precise hallucination detection (identify specific false claims)
- Better feedback for prompt improvement
- Claim-level citations and attribution

**Implementation:**

```python
class AtomicClaim(BaseModel):
    """Single verifiable factual claim."""
    text: str
    type: Literal["factual", "subjective", "opinion"]
    verifiability: float = Field(ge=0.0, le=1.0)

class FactualityEvaluator(BasePydanticEvaluator):
    async def _extract_atomic_claims(self, output: str) -> List[AtomicClaim]:
        """Decompose output into verifiable atomic claims using LLM."""

        decomposition_prompt = f"""Extract atomic factual claims from this text.

Rules:
- Each claim must be independently verifiable
- No subjective statements or opinions
- Break compound claims into separate claims
- Mark subjective/opinion statements as non-factual

Text: {output}

Return list of atomic claims with type classification."""

        agent = self.llm_client.create_agent(
            AtomicClaimList,
            system_prompt="You are a claim extraction expert"
        )
        result = await agent.run(decomposition_prompt)

        return [c for c in result.data.claims if c.type == "factual"]

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None
    ) -> Score:
        """Enhanced evaluation with atomic claim verification."""

        # Extract atomic claims
        claims = await self._extract_atomic_claims(output)

        # Verify each claim
        verifications = []
        for claim in claims:
            verification = await self._verify_claim_with_plugins(
                claim.text,
                context=reference
            )
            verifications.append((claim, verification))

        # Categorize claims
        verified = [(c, v) for c, v in verifications if v.status == "verified"]
        refuted = [(c, v) for c, v in verifications if v.status == "refuted"]
        uncertain = [(c, v) for c, v in verifications if v.status == "uncertain"]

        # Calculate score
        total = len(verified) + len(refuted)
        score = len(verified) / total if total > 0 else 1.0

        # Aggregate sources from all verifications
        all_sources = []
        for _, verification in verifications:
            all_sources.extend(verification.sources)

        return Score(
            name=self.name,
            value=score,
            confidence=sum(v.confidence for _, v in verifications) / len(verifications),
            explanation=self._build_explanation(verified, refuted, uncertain),
            metadata={
                "factual_claims": [c.text for c, _ in verified],
                "non_factual_claims": [c.text for c, _ in refuted],
                "uncertain_claims": [c.text for c, _ in uncertain],
                "sources": [s.model_dump() for s in all_sources[:10]],  # Top 10 sources
                "total_verifications": len(verifications),
                "total_cost_usd": sum(v.cost_usd for _, v in verifications)
            }
        )
```

**Deliverables:**
- [ ] `_extract_atomic_claims()` implementation
- [ ] Claim-level verification logic
- [ ] Source aggregation across claims
- [ ] Cost tracking per evaluation
- [ ] Example with claim-level attribution

**Success Metrics:**
- Claim extraction accuracy: >90%
- Claim-level precision: >95%
- User feedback: Clear identification of specific hallucinations

---

### Phase 4: Additional Plugins (Week 6+) 🎯 SPECIALIZATION

**Goal**: Add domain-specific verification plugins for maximum accuracy

**Plugin Roadmap:**

1. **WikidataPlugin** (Structured Knowledge)
   - Entities, dates, places, people
   - SPARQL queries for structured facts
   - Cost: Free
   - Accuracy: 95%+ for entities

2. **WolframPlugin** (Mathematical/Scientific)
   - Mathematical calculations
   - Scientific constants and formulas
   - Unit conversions
   - Cost: $0.003 per query
   - Accuracy: 99%+ for math/science

3. **PubMedPlugin** (Medical/Health)
   - Medical literature verification
   - Drug information, clinical studies
   - Health claims validation
   - Cost: Free
   - Accuracy: 95%+ for medical claims

4. **ArXivPlugin** (Research Papers)
   - Scientific paper citations
   - Research methodology validation
   - Academic claim verification
   - Cost: Free
   - Accuracy: 95%+ for research claims

**Multi-Plugin Usage:**
```python
# Domain-specific configurations
medical_evaluator = FactualityEvaluator(
    llm_client,
    plugins=[
        TavilyPlugin(),      # General web search
        PubMedPlugin(),      # Medical literature
        WikidataPlugin()     # Medical entities
    ],
    use_cache=True
)

science_evaluator = FactualityEvaluator(
    llm_client,
    plugins=[
        WolframPlugin(),     # Math/science
        ArXivPlugin(),       # Research papers
        WikidataPlugin()     # Scientific entities
    ],
    use_cache=True
)
```

---

## Integration with Loom

### Loom Pipeline Usage

Loom orchestrates AI pipelines and uses Arbiter for evaluation. Enhanced FactualityEvaluator can be used in Loom's quality gates:

```yaml
# Loom pipeline: pipelines/customer_qa.yaml
name: customer_qa_generation
version: 1.0.0

extract:
  source: postgres://customers/questions

transform:
  type: ai
  prompt: prompts/answer_customer_question.txt
  model: gpt-4o-mini
  context:
    - context/product_docs.md
    - context/faq.md

evaluate:
  evaluators:
    # Semantic similarity check
    - name: semantic_check
      type: semantic
      threshold: 0.8

    # FACTUALITY CHECK with external verification
    - name: factuality_check
      type: factuality
      config:
        plugins:
          - name: tavily
            max_results: 5
        use_cache: true
        reference_field: product_docs  # Verify against source docs
        threshold: 0.85

    # Custom criteria
    - name: quality_check
      type: custom_criteria
      criteria: "Helpful, accurate, no hallucination"
      threshold: 0.75

  quality_gate: all_pass  # All evaluators must pass

load:
  destination: postgres://customer_service/qa_responses
  on_failure: quarantine  # Quarantine hallucinated responses
```

### Benefits for Loom Users

1. **Hallucination Prevention**: Quality gates catch hallucinations before reaching customers
2. **Source Attribution**: Responses include citations from Tavily verification
3. **Cost Tracking**: Total cost includes both LLM and verification costs
4. **Audit Trail**: Every claim verified with external sources
5. **Confidence Scoring**: Per-claim confidence enables fine-grained quality control

### Example Loom Output

```python
# Loom pipeline execution result
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
            "sources": [
                {
                    "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                    "title": "Eiffel Tower - Wikipedia",
                    "snippet": "The tower is 300 metres (984 ft) tall..."
                }
            ]
        },
        "quality_check": {"passed": True, "score": 0.88}
    },

    "quality_gate": "PASSED",  # All evaluators passed
    "total_cost_usd": 0.016,  # $0.01 (LLM) + $0.001 (Tavily) + $0.005 (Arbiter evals)
    "loaded_to": "postgres://customer_service/qa_responses"
}
```

---

## Cost & Performance Analysis

### Comparison Matrix

| Configuration | Accuracy | Cost/Eval | Latency | Use Case |
|---------------|----------|-----------|---------|----------|
| **Pure LLM** (current) | 70-80% | $0.01 | 500ms | Quick checks, low stakes |
| **+ Tavily** | 90-95% | $0.015 | 1500ms | Production, citations needed |
| **+ Cache (50% hit)** | 90-95% | $0.008 | 775ms | High-volume, repeated claims |
| **+ Cache (80% hit)** | 90-95% | $0.004 | 340ms | Steady-state production |
| **+ Multi-plugin** | 95-98% | $0.025 | 2000ms | Critical applications, medical/legal |
| **+ Multi + Cache (50%)** | 95-98% | $0.013 | 1025ms | Enterprise production |

### Cost Breakdown (Per Evaluation)

**Pure LLM Mode:**
- LLM API: $0.01 (100 tokens @ gpt-4o-mini)
- Total: **$0.01**

**With Tavily Plugin:**
- LLM claim extraction: $0.01
- Tavily search: $0.001
- LLM verification: $0.004
- Total: **$0.015**

**With Cache (50% hit rate):**
- Cache hit (50%): $0.0015
- Cache miss (50%): $0.015
- Average: **$0.008**

**With Cache (80% hit rate):**
- Cache hit (80%): $0.0015
- Cache miss (20%): $0.015
- Average: **$0.004** (60% cost reduction!)

### Performance Projections

**1,000 evaluations/day:**
- Pure LLM: $10/day, 8.3 minutes
- + Tavily: $15/day, 25 minutes
- + Cache (50%): $8/day, 13 minutes
- + Cache (80%): $4/day, 5.7 minutes

**10,000 evaluations/day:**
- Pure LLM: $100/day
- + Tavily: $150/day
- + Cache (50%): $80/day
- + Cache (80%): $40/day (60% savings!)

**Key Insight**: Vector cache is critical. At scale (10K+ evals/day), cache transforms economics from $150/day to $40/day.

---

## Migration Path

### Backward Compatibility (CRITICAL)

All enhancements must maintain backward compatibility. Existing code continues to work:

```python
# Existing code (v0.1.0) - still works in v0.2.0+
evaluator = FactualityEvaluator(llm_client)
score = await evaluator.evaluate(output, reference)
# Pure LLM mode, no breaking changes
```

### Opt-In Enhancement

Users opt into plugins and caching:

```python
# Enhanced mode (opt-in)
from arbiter import FactualityEvaluator
from arbiter.plugins import TavilyPlugin

evaluator = FactualityEvaluator(
    llm_client,
    plugins=[TavilyPlugin(api_key=os.getenv("TAVILY_API_KEY"))],
    use_cache=True
)

score = await evaluator.evaluate(output, reference)
# Now includes citations in metadata['sources']!
```

### Migration Steps for Users

1. **Phase 1**: Add Tavily API key to environment
2. **Phase 2**: Update FactualityEvaluator initialization with plugins
3. **Phase 3**: Enable caching with Milvus backend
4. **Phase 4**: Access sources in evaluation metadata

---

## Testing Strategy

### Unit Tests (Mock External APIs)

```python
# tests/unit/test_factuality_plugins.py
@pytest.mark.asyncio
async def test_tavily_plugin_verify_claim():
    """Test TavilyPlugin with mocked API response."""

    mock_response = {
        "results": [{
            "title": "Eiffel Tower - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
            "content": "The Eiffel Tower is 300 meters tall...",
            "score": 0.95
        }]
    }

    with patch('arbiter.plugins.tavily.tavily_search', return_value=mock_response):
        plugin = TavilyPlugin()
        result = await plugin.verify_claim("The Eiffel Tower is 300 meters tall")

        assert result.status == "verified"
        assert result.confidence >= 0.9
        assert len(result.sources) > 0
```

### Integration Tests (Real API Calls)

```python
# tests/integration/test_factuality_with_tavily.py
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="Tavily API key required")
async def test_factuality_with_real_tavily():
    """Test with real Tavily API."""

    evaluator = FactualityEvaluator(
        llm_client=await LLMManager.get_client(model="gpt-4o-mini"),
        plugins=[TavilyPlugin(api_key=os.getenv("TAVILY_API_KEY"))]
    )

    # Known hallucination
    result = await evaluator.evaluate(
        output="The Eiffel Tower is 500 meters tall",
        reference="The Eiffel Tower is 300 meters tall"
    )

    assert result.value < 0.7  # Should detect hallucination
    assert "500 meters" in str(result.metadata['non_factual_claims'])
    assert len(result.metadata['sources']) > 0
```

### Performance Tests

```python
# tests/performance/test_cache_performance.py
@pytest.mark.asyncio
async def test_cache_speedup():
    """Verify cache provides significant speedup."""

    cache = FactCache(MilvusClient(uri="memory"))
    evaluator = FactualityEvaluator(llm_client, use_cache=True, cache_backend=cache)

    # First evaluation - cache miss
    start = time.time()
    result1 = await evaluator.evaluate(output="Paris is the capital of France")
    time1 = time.time() - start

    # Second evaluation - cache hit
    start = time.time()
    result2 = await evaluator.evaluate(output="Paris is the capital of France")
    time2 = time.time() - start

    assert result1.value == result2.value
    assert time2 < time1 * 0.2  # Cache hit should be 5x+ faster
```

---

## Documentation Requirements

### User-Facing Documentation

1. **Enhanced README.md section**:
   - Quick example with plugins
   - Performance comparison table
   - Cost analysis

2. **New example files**:
   - `examples/factuality_with_tavily.py`
   - `examples/factuality_with_cache.py`
   - `examples/factuality_multi_plugin.py`
   - `examples/loom_factuality_integration.py`

3. **API Documentation**:
   - Plugin protocol documentation
   - Cache configuration guide
   - Source attribution guide

### Technical Documentation

1. **FACTUALITY_ENHANCEMENT_PLAN.md** (this file)
2. **Plugin development guide**: How to create custom plugins
3. **Cache configuration guide**: Milvus setup and tuning
4. **Loom integration guide**: Using FactualityEvaluator in pipelines

---

## Success Criteria

### Phase 1 Success (Tavily Plugin)
- [ ] TavilyPlugin passes all unit tests
- [ ] Integration test with real Tavily API works
- [ ] Accuracy improves from 70-80% to 90-95%
- [ ] Sources included in evaluation metadata
- [ ] Backward compatibility maintained
- [ ] Example code demonstrates citations
- [ ] Documentation updated

### Phase 2 Success (Vector Cache)
- [ ] Cache achieves 30x speedup on hits
- [ ] Cache hit rate reaches 50%+ after warmup
- [ ] Cost reduced by 40%+ in steady state
- [ ] TTL expiration works correctly
- [ ] Semantic similarity threshold validated
- [ ] Performance benchmarks documented

### Phase 3 Success (Atomic Claims)
- [ ] Claim extraction accuracy >90%
- [ ] Claim-level verification precision >95%
- [ ] Users can identify specific hallucinations
- [ ] Claim-level citations available
- [ ] Cost tracking per claim works

### Overall Success Metrics
- Accuracy: 70-80% → 90-98%
- Cost efficiency: $0.01 → $0.004 (with cache)
- Performance: 500ms → 50ms (cached)
- User satisfaction: Hallucination detection clarity
- Loom integration: Quality gates prevent hallucinated responses

---

## Next Steps

1. **Immediate**: Create `arbiter/plugins/` directory structure
2. **Week 1**: Implement TavilyPlugin with tests
3. **Week 2**: Integrate plugins into FactualityEvaluator
4. **Week 3**: Implement vector cache with Milvus
5. **Week 4**: Performance benchmarks and optimization
6. **Week 5**: Atomic claim decomposition
7. **Week 6+**: Additional plugins (Wikidata, Wolfram, PubMed)

---

**Status**: Ready for implementation
**Approval**: Pending user confirmation
**Last Updated**: 2025-11-15
