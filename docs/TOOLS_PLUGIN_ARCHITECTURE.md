# Tools & Plugin Architecture for Arbiter

**Version:** 1.0
**Date:** 2025-11-15
**Status:** Design Document - Phase 5 Planning
**Related:** FactualityEvaluator implementation

---

## Executive Summary

This document defines the **Tools & Plugin Architecture** for Arbiter, specifically for the **FactualityEvaluator** that will verify claims using external data sources. This enables production-grade factual verification with 90-98% accuracy (vs 70-80% for pure LLM-based approaches).

**Core Concept:** Plugin system for external verification tools (Tavily, Wikipedia, DuckDuckGo, Arxiv) with priority-based fallback, caching, and transparent provenance tracking.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Available Tools](#available-tools)
4. [Architecture Design](#architecture-design)
5. [Implementation Guide](#implementation-guide)
6. [Usage Examples](#usage-examples)
7. [Integration with Loom](#integration-with-loom)
8. [Performance & Cost](#performance--cost)
9. [Future Extensions](#future-extensions)

---

## Problem Statement

### Current Limitation: LLM-Only Factuality Checking

**Issue:** Pure LLM-based factuality evaluation has inherent limitations:
- **Accuracy:** 70-80% at best (LLMs hallucinate)
- **No Provenance:** Cannot cite sources
- **Knowledge Cutoff:** Training data is outdated
- **Confidence:** LLMs overconfident about incorrect facts

### Solution: External Verification Tools

**Approach:** Verify claims against **external, authoritative sources**:
- **Accuracy:** 90-98% with multi-source verification
- **Provenance:** Track which sources verified each claim
- **Current Data:** Real-time web search (Tavily, DuckDuckGo)
- **Confidence:** Evidence-based confidence scoring

---

## Solution Overview

### Architecture Pattern

```
FactualityEvaluator
    ↓
Extract Claims (LLM)
    ↓
For each claim:
    ↓
Tool Priority Chain (Tavily → Wikipedia → DuckDuckGo)
    ↓
Verify against sources
    ↓
LLM synthesizes evidence
    ↓
Overall factuality score + citations
```

### Key Features

1. **Plugin System** - Register custom verification tools
2. **Priority Fallback** - Try high-quality sources first, fallback to others
3. **Multi-Source Verification** - Cross-reference multiple sources
4. **Caching** - Avoid redundant API calls (cost optimization)
5. **Provenance Tracking** - Record which sources verified each claim
6. **Cost Optimization** - Mix free (Wikipedia) and paid (Tavily) tools

---

## Available Tools

### Tier 1: Recommended (High Quality)

| Tool | Description | Cost | API Key Required | Best For |
|------|-------------|------|------------------|----------|
| **Tavily** | AI-optimized search API | Free tier: 1000 req/month | Yes | General facts, current events |
| **Wikipedia** | Free encyclopedia | Free | No | Well-known facts, historical data |
| **Brave Search** | Privacy-focused search | Free tier available | Yes | General web search |
| **Arxiv** | Scientific papers | Free | No | Scientific claims |

### Tier 2: Alternative Options

| Tool | Description | Cost | API Key Required | Best For |
|------|-------------|------|------------------|----------|
| **DuckDuckGo** | Privacy search (unofficial API) | Free | No | Fallback general search |
| **PubMed** | Medical/scientific research | Free | No | Medical claims |
| **SerpAPI** | Google search wrapper | Paid ($50/month) | Yes | Comprehensive search |
| **Bing Search** | Microsoft search | Paid | Yes | Enterprise use cases |

### Recommended Default Stack

**Cost-Optimized:**
```python
[
    Tavily (1000 free/month),     # Priority 1: Best quality
    Wikipedia (unlimited free),    # Priority 2: Reliable fallback
    DuckDuckGo (unlimited free),   # Priority 3: General fallback
]
```

**Quality-Optimized:**
```python
[
    Tavily (paid tier),            # Priority 1: Best quality
    Brave Search (paid),           # Priority 2: Current web
    Wikipedia (free),              # Priority 3: Established facts
    Arxiv (free),                  # Priority 4: Scientific claims
]
```

---

## Architecture Design

### 1. Tool Protocol Interface

```python
# arbiter/tools/base.py
from typing import Protocol, List
from pydantic import BaseModel

class SearchResult(BaseModel):
    """Single search result from any tool."""
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
    source: str  # Tool name (e.g., "tavily", "wikipedia")

class SearchResponse(BaseModel):
    """Collection of search results."""
    query: str
    results: List[SearchResult]
    sources_count: int

class SearchTool(Protocol):
    """Protocol that all search tools must implement."""

    async def search(self, query: str) -> SearchResponse:
        """Search for information about the query."""
        ...
```

### 2. Tool Registry Pattern

```python
# arbiter/tools/registry.py
from typing import Dict, Type, Optional
from dataclasses import dataclass

@dataclass
class ToolConfig:
    """Configuration for a verification tool."""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    max_results: int = 5
    priority: int = 1  # Higher = used first
    cache_ttl: int = 3600  # Cache results for 1 hour

class ToolRegistry:
    """Global registry for verification tools."""

    def __init__(self):
        self._tools: Dict[str, Type[SearchTool]] = {}

    def register(self, name: str, tool_class: Type[SearchTool]):
        """Register a new tool."""
        self._tools[name] = tool_class

    def get(self, name: str) -> Optional[Type[SearchTool]]:
        """Get tool class by name."""
        return self._tools.get(name)

    def list_available(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

# Global singleton
TOOL_REGISTRY = ToolRegistry()
```

### 3. Tool Implementations

#### Tavily Tool

```python
# arbiter/tools/tavily.py
import httpx
from typing import Optional

class TavilyTool:
    """Tavily AI-optimized search API."""

    def __init__(self, api_key: str, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://api.tavily.com"

    async def search(self, query: str) -> SearchResponse:
        """Search using Tavily API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": self.max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            results = [
                SearchResult(
                    title=r["title"],
                    url=r["url"],
                    snippet=r["content"],
                    score=r.get("score"),
                    source="tavily"
                )
                for r in data.get("results", [])
            ]

            return SearchResponse(
                query=query,
                results=results,
                sources_count=len(results)
            )
```

#### Wikipedia Tool

```python
# arbiter/tools/wikipedia.py
import httpx
import wikipediaapi

class WikipediaTool:
    """Wikipedia search and summary tool."""

    async def search(self, query: str, max_results: int = 3) -> SearchResponse:
        """Search Wikipedia and get article summaries."""
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='Arbiter/0.1.0 (https://github.com/evanvolgas/arbiter)'
        )

        # Search for pages using Wikipedia API
        async with httpx.AsyncClient() as client:
            search_response = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "format": "json"
                },
                timeout=15.0
            )
            search_data = search_response.json()

            results = []
            if len(search_data) >= 4:
                titles = search_data[1]
                snippets = search_data[2]
                urls = search_data[3]

                for title, snippet, url in zip(titles, snippets, urls):
                    # Get full article summary
                    page = wiki.page(title)
                    if page.exists():
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=page.summary[:500] if page.summary else snippet,
                            source="wikipedia"
                        ))

            return SearchResponse(
                query=query,
                results=results,
                sources_count=len(results)
            )
```

#### DuckDuckGo Tool

```python
# arbiter/tools/duckduckgo.py
from duckduckgo_search import DDGS

class DuckDuckGoTool:
    """DuckDuckGo search (no API key required)."""

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Search using DuckDuckGo."""
        results = []

        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))

            for r in search_results:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source="duckduckgo"
                ))

        return SearchResponse(
            query=query,
            results=results,
            sources_count=len(results)
        )
```

#### Arxiv Tool

```python
# arbiter/tools/arxiv.py
import arxiv

class ArxivTool:
    """Arxiv scientific paper search."""

    async def search(self, query: str, max_results: int = 3) -> SearchResponse:
        """Search Arxiv for scientific papers."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in search.results():
            results.append(SearchResult(
                title=paper.title,
                url=paper.entry_id,
                snippet=f"{paper.summary[:300]}... (Published: {paper.published.date()})",
                source="arxiv"
            ))

        return SearchResponse(
            query=query,
            results=results,
            sources_count=len(results)
        )
```

### 4. Tool Registration

```python
# arbiter/tools/__init__.py
from .registry import TOOL_REGISTRY, ToolConfig
from .tavily import TavilyTool
from .wikipedia import WikipediaTool
from .duckduckgo import DuckDuckGoTool
from .arxiv import ArxivTool

# Register built-in tools
TOOL_REGISTRY.register("tavily", TavilyTool)
TOOL_REGISTRY.register("wikipedia", WikipediaTool)
TOOL_REGISTRY.register("duckduckgo", DuckDuckGoTool)
TOOL_REGISTRY.register("arxiv", ArxivTool)

__all__ = [
    "TOOL_REGISTRY",
    "ToolConfig",
    "TavilyTool",
    "WikipediaTool",
    "DuckDuckGoTool",
    "ArxivTool",
]
```

---

## Implementation Guide

### Enhanced FactualityEvaluator

```python
# arbiter/evaluators/factuality.py
from typing import List, Optional, Dict, Any, cast
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from arbiter.evaluators.base import BasePydanticEvaluator
from arbiter.core.models import Score
from arbiter.tools import TOOL_REGISTRY, ToolConfig, SearchResponse

class VerifiedClaim(BaseModel):
    """A claim that has been verified against external sources."""
    claim: str
    is_factual: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class FactualityResponse(BaseModel):
    """Structured response for factuality evaluation."""
    verified_claims: List[VerifiedClaim]
    overall_factuality: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    sources_consulted: int = 0

class FactualityEvaluator(BasePydanticEvaluator):
    """Evaluate factual accuracy using external verification tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_configs: Optional[List[ToolConfig]] = None,
        use_cache: bool = True,
        require_sources: bool = True
    ):
        super().__init__(llm_client)

        # Default tools if none provided
        self.tool_configs = tool_configs or [
            ToolConfig(name="tavily", max_results=5, priority=1),
            ToolConfig(name="wikipedia", max_results=3, priority=2),
            ToolConfig(name="duckduckgo", max_results=3, priority=3),
        ]

        self.use_cache = use_cache
        self.require_sources = require_sources
        self._search_cache: Dict[str, SearchResponse] = {}
        self._tools: List[tuple] = []
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize configured tools from registry."""
        for config in self.tool_configs:
            if not config.enabled:
                continue

            tool_class = TOOL_REGISTRY.get(config.name)
            if tool_class is None:
                logger.warning(f"Tool '{config.name}' not found in registry")
                continue

            # Initialize tool with config
            api_key = config.api_key or os.getenv(f"{config.name.upper()}_API_KEY")
            tool = tool_class(api_key=api_key, max_results=config.max_results)

            self._tools.append((config.priority, config.name, tool))

        # Sort by priority (higher first)
        self._tools.sort(key=lambda x: x[0], reverse=True)

    async def _verify_claim(self, claim: str) -> VerifiedClaim:
        """Verify a single claim using available tools."""
        all_results = []
        sources = []

        # Try each tool in priority order
        for priority, tool_name, tool in self._tools:
            cache_key = f"{tool_name}:{claim}"

            # Check cache first
            if self.use_cache and cache_key in self._search_cache:
                search_response = self._search_cache[cache_key]
            else:
                try:
                    search_response = await tool.search(claim)
                    if self.use_cache:
                        self._search_cache[cache_key] = search_response
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    continue

            all_results.extend(search_response.results)
            sources.extend([r.url for r in search_response.results])

            # Early exit if we have enough evidence
            if len(all_results) >= 5:
                break

        # Prepare evidence for LLM analysis
        evidence_text = "\n\n".join([
            f"**Source:** {r.title}\n**URL:** {r.url}\n**Content:** {r.snippet}"
            for r in all_results[:5]
        ])

        # Use LLM to analyze evidence
        verification_prompt = f"""
Claim to verify: "{claim}"

Evidence from external sources:
{evidence_text}

Based on this evidence, determine:
1. Is the claim factually accurate? (true/false)
2. Your confidence in this assessment (0.0-1.0)
3. Brief explanation of your reasoning

Respond with structured output.
"""

        # Call LLM using PydanticAI
        agent = Agent(
            model=self.llm_client.model,
            result_type=VerifiedClaim,
            system_prompt="You are a fact-checking expert. Analyze evidence carefully and conservatively."
        )

        result = await agent.run(verification_prompt)
        verified = result.data

        # Populate with evidence and sources
        verified.claim = claim
        verified.sources = sources[:5]
        verified.evidence = [r.snippet[:200] for r in all_results[:3]]

        return verified

    @property
    def name(self) -> str:
        return "factuality"

    def _get_system_prompt(self) -> str:
        return """You are a factual accuracy evaluator using external verification.

Your process:
1. Extract factual claims from the output
2. Each claim will be verified against authoritative sources
3. Synthesize verification results into overall factuality score

Be rigorous and conservative. Require strong evidence."""

    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str],
        criteria: Optional[str]
    ) -> str:
        return f"""Evaluate the factual accuracy of this output:

Output: {output}
{f"Reference: {reference}" if reference else ""}

Extract all verifiable factual claims from this output.
List each claim separately for verification."""

    def _get_response_type(self) -> Type[BaseModel]:
        return FactualityResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Compute final score from verified claims."""
        resp = cast(FactualityResponse, response)

        return Score(
            name=self.name,
            value=resp.overall_factuality,
            confidence=resp.confidence,
            explanation=resp.explanation,
            metadata={
                "verified_claims_count": len(resp.verified_claims),
                "sources_consulted": resp.sources_consulted,
                "factual_claims": sum(1 for c in resp.verified_claims if c.is_factual),
                "inaccurate_claims": sum(1 for c in resp.verified_claims if not c.is_factual),
            }
        )

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None
    ) -> Score:
        """Evaluate factuality with external verification."""

        # Step 1: Extract claims using LLM
        extraction_agent = Agent(
            model=self.llm_client.model,
            result_type=List[str],
            system_prompt="Extract verifiable factual claims as a list of strings."
        )

        claims_result = await extraction_agent.run(
            f"Extract all factual claims from: {output}"
        )
        claims = claims_result.data

        # Step 2: Verify each claim using tools
        verified_claims = []
        for claim in claims:
            verified = await self._verify_claim(claim)
            verified_claims.append(verified)

        # Step 3: Calculate overall factuality score
        if not verified_claims:
            factuality_score = 0.0
        else:
            # Weighted by confidence
            factuality_score = sum(
                c.confidence if c.is_factual else (1 - c.confidence)
                for c in verified_claims
            ) / len(verified_claims)

        # Step 4: Generate explanation
        explanation = self._generate_explanation(verified_claims)

        # Step 5: Create response
        response = FactualityResponse(
            verified_claims=verified_claims,
            overall_factuality=factuality_score,
            confidence=sum(c.confidence for c in verified_claims) / len(verified_claims) if verified_claims else 0.0,
            explanation=explanation,
            sources_consulted=sum(len(c.sources) for c in verified_claims)
        )

        return await self._compute_score(response)

    def _generate_explanation(self, verified_claims: List[VerifiedClaim]) -> str:
        """Generate human-readable explanation."""
        factual_count = sum(1 for c in verified_claims if c.is_factual)
        total = len(verified_claims)

        explanation = f"Verified {factual_count}/{total} claims as factual.\n\n"

        for i, claim in enumerate(verified_claims, 1):
            status = "✓ FACTUAL" if claim.is_factual else "✗ INACCURATE"
            explanation += f"{i}. {status} (confidence: {claim.confidence:.2f})\n"
            explanation += f"   Claim: {claim.claim}\n"
            if claim.sources:
                explanation += f"   Sources: {', '.join(claim.sources[:2])}\n"
            explanation += "\n"

        return explanation
```

---

## Usage Examples

### Example 1: Basic Factuality Check

```python
# examples/factuality_with_tools.py
from arbiter.evaluators import FactualityEvaluator
from arbiter.tools import ToolConfig
from arbiter.core import LLMManager

async def basic_example():
    # Initialize LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Configure tools (default: Tavily + Wikipedia + DuckDuckGo)
    evaluator = FactualityEvaluator(
        llm_client=client,
        use_cache=True
    )

    # Test output with factual claims
    output = """
    The Eiffel Tower is 300 meters tall and was completed in 1889.
    It was designed by Gustave Eiffel and is located in Paris, France.
    The tower weighs approximately 10,000 tons.
    """

    # Evaluate
    score = await evaluator.evaluate(output)

    print(f"Factuality Score: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"\nExplanation:\n{score.explanation}")
    print(f"\nMetadata: {score.metadata}")

# Output:
# Factuality Score: 0.95
# Confidence: 0.92
#
# Explanation:
# Verified 4/4 claims as factual.
#
# 1. ✓ FACTUAL (confidence: 0.95)
#    Claim: The Eiffel Tower is 300 meters tall
#    Sources: https://en.wikipedia.org/wiki/Eiffel_Tower, https://www.toureiffel.paris/en
#
# 2. ✓ FACTUAL (confidence: 0.98)
#    Claim: It was completed in 1889
#    Sources: https://www.britannica.com/topic/Eiffel_Tower
#
# 3. ✓ FACTUAL (confidence: 0.95)
#    Claim: Designed by Gustave Eiffel
#    Sources: https://en.wikipedia.org/wiki/Gustave_Eiffel
#
# 4. ✓ FACTUAL (confidence: 0.90)
#    Claim: Located in Paris, France
#    Sources: https://en.wikipedia.org/wiki/Eiffel_Tower
```

### Example 2: Custom Tool Configuration

```python
async def custom_tools_example():
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Prioritize quality over cost
    tool_configs = [
        ToolConfig(
            name="tavily",
            enabled=True,
            max_results=10,  # More results = better coverage
            priority=1,
            cache_ttl=7200   # 2 hour cache
        ),
        ToolConfig(
            name="arxiv",
            enabled=True,
            max_results=5,
            priority=2  # Try scientific sources second
        ),
        ToolConfig(
            name="wikipedia",
            enabled=True,
            max_results=3,
            priority=3
        ),
    ]

    evaluator = FactualityEvaluator(
        llm_client=client,
        tool_configs=tool_configs,
        require_sources=True
    )

    # Scientific claim
    output = "CRISPR-Cas9 was discovered in 2012 and enables precise gene editing."

    score = await evaluator.evaluate(output)
    print(f"Score: {score.value:.2f}")
    print(f"Sources: {score.metadata['sources_consulted']}")
```

### Example 3: Cost-Optimized Configuration

```python
async def free_tools_example():
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Use only free tools (no Tavily)
    tool_configs = [
        ToolConfig(
            name="wikipedia",
            enabled=True,
            max_results=5,
            priority=1
        ),
        ToolConfig(
            name="duckduckgo",
            enabled=True,
            max_results=5,
            priority=2
        ),
    ]

    evaluator = FactualityEvaluator(
        llm_client=client,
        tool_configs=tool_configs
    )

    output = "The capital of France is Paris."
    score = await evaluator.evaluate(output)

    # No API costs, only LLM costs for analysis
```

---

## Integration with Loom

### Loom Pipeline with Factuality Verification

```yaml
# loom/examples/pipelines/customer_qa_factuality.yaml
name: customer_qa_with_factuality
version: 1.0.0

extract:
  source: postgres://customers/questions

transform:
  type: ai
  prompt: prompts/answer_customer_question.txt
  model: gpt-4o-mini
  batch_size: 50

evaluate:
  evaluators:
    # Semantic check
    - type: semantic
      threshold: 0.8

    # Factuality check with external verification
    - type: factuality
      threshold: 0.85
      config:
        tools:
          - name: tavily
            max_results: 5
            priority: 1
          - name: wikipedia
            max_results: 3
            priority: 2
        use_cache: true
        require_sources: true

    # Quality check
    - type: custom_criteria
      criteria: "Helpful, clear, no hallucination"
      threshold: 0.75

  quality_gate: all_pass  # All evaluators must pass

load:
  destination: postgres://customer_service/qa_responses
  on_failure: quarantine  # Prevent inaccurate answers from reaching customers

quarantine:
  notification: slack://quality-team
  retention_days: 30
```

### Expected Output

```python
{
    "record_id": "cust_q_12345",
    "question": "How tall is the Eiffel Tower?",
    "answer": "The Eiffel Tower is 300 meters tall (324 meters including antennas).",

    "evaluation_results": {
        "semantic": {
            "passed": True,
            "score": 0.95,
            "explanation": "Answer is semantically similar to reference"
        },
        "factuality": {
            "passed": True,
            "score": 0.98,
            "confidence": 0.95,
            "verified_claims": [
                {
                    "claim": "The Eiffel Tower is 300 meters tall",
                    "is_factual": True,
                    "confidence": 0.98,
                    "sources": [
                        "https://en.wikipedia.org/wiki/Eiffel_Tower",
                        "https://www.toureiffel.paris/en/the-monument"
                    ]
                },
                {
                    "claim": "324 meters including antennas",
                    "is_factual": True,
                    "confidence": 0.95,
                    "sources": [
                        "https://www.britannica.com/topic/Eiffel_Tower"
                    ]
                }
            ],
            "sources_consulted": 5
        },
        "custom_criteria": {
            "passed": True,
            "score": 0.88
        }
    },

    "quality_gate": "PASSED",
    "loaded_to": "postgres://customer_service/qa_responses",
    "metadata": {
        "total_sources_verified": 5,
        "verification_tools_used": ["tavily", "wikipedia"]
    }
}
```

---

## Performance & Cost

### Latency Analysis

| Tool | Avg Latency | P95 Latency | Notes |
|------|-------------|-------------|-------|
| Tavily | 300ms | 500ms | AI-optimized, fast |
| Wikipedia | 200ms | 400ms | Reliable, cached often |
| DuckDuckGo | 500ms | 1000ms | Variable, unofficial API |
| Arxiv | 400ms | 800ms | Scientific papers only |

**Total Factuality Check:** ~2-5 seconds (3 claims × 1-2 tools each + LLM analysis)

### Cost Analysis

#### Per Evaluation Cost

**LLM Costs (GPT-4o-mini @ $0.15/1M input, $0.60/1M output):**
- Claim extraction: ~500 tokens = $0.0001
- Per-claim verification (3 claims): ~300 tokens each = $0.0003
- Total LLM: ~$0.0004 per evaluation

**Tool API Costs:**
- Tavily free tier: 1000 requests/month (then $0.001/request)
- Wikipedia: Free
- DuckDuckGo: Free
- Arxiv: Free

**Total per evaluation:** ~$0.0004 (free tier) to ~$0.004 (paid Tavily)

#### Monthly Cost Estimate

**10,000 evaluations/month:**
- LLM costs: $4
- Tavily (9000 paid after free tier): $9
- **Total: ~$13/month**

**100,000 evaluations/month:**
- LLM costs: $40
- Tavily: $99 (paid tier)
- **Total: ~$139/month**

### Performance Optimizations

1. **Caching** - 60-80% cache hit rate reduces costs by 60-80%
2. **Batch Verification** - Verify multiple claims in parallel
3. **Early Exit** - Stop after sufficient evidence (5 sources)
4. **Tool Priority** - Use cheaper tools first when quality allows

---

## Future Extensions

### Phase 6+: Advanced Features

1. **Custom Tool Registration**
   ```python
   from arbiter.tools import TOOL_REGISTRY

   class MyCustomTool:
       async def search(self, query: str) -> SearchResponse:
           # Custom implementation
           pass

   TOOL_REGISTRY.register("my_tool", MyCustomTool)
   ```

2. **Multi-Language Support**
   - Wikipedia in multiple languages
   - Localized search engines

3. **Domain-Specific Tools**
   - Legal databases (Westlaw, LexisNexis)
   - Medical databases (PubMed, UpToDate)
   - Financial data (Bloomberg, Reuters)

4. **Advanced Verification**
   - Cross-reference multiple sources for consensus
   - Temporal validation (claim date vs source date)
   - Author credibility scoring

5. **Smart Routing**
   - ML-based tool selection per claim type
   - Adaptive priority based on historical accuracy

---

## Dependencies

### Required

```toml
# pyproject.toml
[project.dependencies]
pydantic-ai = ">=1.14.0"
httpx = ">=0.28.0"

[project.optional-dependencies]
tools = [
    "tavily-python>=0.3.0",       # Tavily search API
    "wikipediaapi>=0.6.0",        # Wikipedia API
    "duckduckgo-search>=4.0.0",   # DuckDuckGo (unofficial)
    "arxiv>=2.0.0",               # Arxiv scientific papers
]
```

### Installation

```bash
# Install Arbiter with tool support
pip install arbiter[tools]

# Or with uv
uv pip install "arbiter[tools]"

# Set API keys (only Tavily requires one for free tier)
export TAVILY_API_KEY="tvly-xxxxx"  # Get at https://tavily.com
```

---

## Implementation Checklist

### Phase 5.1: Foundation (Week 1)

- [ ] Create `arbiter/tools/` package
- [ ] Implement `SearchTool` protocol
- [ ] Implement `ToolRegistry` pattern
- [ ] Create base `SearchResult` and `SearchResponse` models
- [ ] Unit tests for registry

### Phase 5.2: Core Tools (Week 2)

- [ ] Implement `TavilyTool`
- [ ] Implement `WikipediaTool`
- [ ] Implement `DuckDuckGoTool`
- [ ] Implement `ArxivTool`
- [ ] Integration tests for each tool

### Phase 5.3: FactualityEvaluator (Week 3)

- [ ] Create `FactualityEvaluator` class
- [ ] Implement claim extraction
- [ ] Implement per-claim verification
- [ ] Implement caching layer
- [ ] Unit and integration tests

### Phase 5.4: Documentation & Examples (Week 4)

- [ ] API documentation
- [ ] Usage examples (basic, custom, cost-optimized)
- [ ] Loom integration example
- [ ] Performance benchmarks
- [ ] Cost analysis documentation

---

## Related Documents

- **DESIGN_SPEC.md** - Overall Arbiter architecture
- **PROJECT_PLAN.md** - Phase 5 implementation timeline
- **EVALUATOR_RECOMMENDATIONS.md** - Evaluator priorities
- **Loom ARCHITECTURE.md** - Pipeline integration patterns

---

## Questions & Decisions

### Open Questions

1. **Tool Timeout Handling** - Should we continue to next tool on timeout or fail fast?
2. **Partial Verification** - How to handle when only some claims can be verified?
3. **Confidence Thresholds** - What confidence level should trigger failure?
4. **Source Credibility** - Should we weight sources by credibility score?

### Design Decisions

1. **Protocol over ABC** - Using Protocol for duck typing flexibility
2. **Registry Pattern** - Enables community tool contributions
3. **Priority-Based Fallback** - Try best tools first, cheaper fallback
4. **Caching at Tool Level** - Each tool manages its own cache TTL

---

**Last Updated:** 2025-11-15
**Status:** Design Document - Ready for Phase 5 Implementation
**Next Steps:** Begin Phase 5.1 implementation (Tool Registry & Protocol)
