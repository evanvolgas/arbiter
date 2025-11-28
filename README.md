<div align="center">
  <h1>Arbiter</h1>

  <p><strong>The only LLM evaluation framework that shows you exactly what your evaluations cost</strong></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/ashita-ai/arbiter"><img src="https://img.shields.io/badge/version-0.1.1-blue" alt="Version"></a>
    <a href="https://ai.pydantic.dev"><img src="https://img.shields.io/badge/PydanticAI-native-purple" alt="PydanticAI"></a>
    <a href="https://codecov.io/gh/ashita-ai/arbiter"><img src="https://codecov.io/gh/ashita-ai/arbiter/branch/main/graph/badge.svg" alt="Coverage"></a>
  </p>
</div>

---

## Why Arbiter?

Most evaluation frameworks tell you if your outputs are good. **Arbiter tells you that AND exactly what it cost.** Every evaluation automatically tracks tokens, latency, and real dollar costs across any provider - no manual instrumentation required.

```python
from arbiter_ai import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Cost: ${await result.total_llm_cost():.6f}")  # Real pricing data
print(f"Calls: {len(result.interactions)}")           # Every LLM interaction
```

### What Makes Arbiter Different?

**1. Automatic Cost Transparency** (Unique)
- Real-time cost calculation using live pricing data
- Cost breakdown by evaluator and model
- No guessing - see exactly what each evaluation costs

**2. PydanticAI Native**
- Built on PydanticAI - same patterns you already know
- Type-safe structured outputs
- If you use PydanticAI, Arbiter feels familiar

**3. Pure Library Philosophy**
- No platform signup required
- No server to run
- Just `pip install` and go
- No vendor lock-in to SaaS platforms

**4. Complete Observability**
- Every LLM interaction automatically tracked
- Prompts, responses, tokens, latency - all visible
- Perfect for debugging evaluation issues

### How Arbiter Compares

| Feature | Arbiter | LangSmith | Ragas | DeepEval |
|---------|---------|-----------|-------|----------|
| **Automatic cost tracking** | ✅ Real-time | ❌ | ❌ | ❌ |
| **Pure library** (no platform) | ✅ | ❌ Platform | ✅ | ✅ |
| **Provider-agnostic** | ✅ 6 providers | ✅ | ✅ | ✅ |
| **Interaction visibility** | ✅ Every call | Manual | ❌ | Manual |
| **PydanticAI native** | ✅ | ❌ | ❌ | ❌ |
| **Built-in evaluators** | 6 | - | 8 | 20+ |
| **Type-safe** (strict mypy) | ✅ | ❌ | ❌ | ❌ |

**Best for**: Teams who need cost transparency, complete observability, and no platform lock-in.

## Installation

**Requirements**: Python 3.11+

```bash
# Clone the repository
git clone https://github.com/ashita-ai/arbiter.git
cd arbiter

# Install with uv (recommended - handles environment automatically)
uv sync

# Or with pip (manual environment management)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Setup

Arbiter requires API keys for the LLM providers you want to use. Configure them using a `.env` file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# At minimum, add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

The `.env.example` file includes placeholders for all supported providers:
- `OPENAI_API_KEY` - For GPT models (required for examples)
- `ANTHROPIC_API_KEY` - For Claude models
- `GOOGLE_API_KEY` - For Gemini models
- `GROQ_API_KEY` - For Groq (fast inference)
- `MISTRAL_API_KEY` - For Mistral models
- `COHERE_API_KEY` - For Cohere models
- `TAVILY_API_KEY` - For SearchVerifier (web search fact verification, optional)

**Optional Features:**
```bash
# For web search fact verification (SearchVerifier)
pip install arbiter-ai[verifiers]

# For fast semantic evaluation (FAISS backend)
pip install arbiter-ai[scale]

# For result persistence (PostgreSQL + Redis)
pip install arbiter-ai[storage]
```

**Note:** All examples automatically load environment variables from `.env` using `python-dotenv`.

## Quick Start

```python
from arbiter_ai import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

# See everything that happened
print(f"Score: {result.overall_score:.2f}")
print(f"Cost: ${await result.total_llm_cost():.6f}")
print(f"Time: {result.processing_time:.2f}s")
print(f"LLM Calls: {len(result.interactions)}")

# Get detailed cost breakdown
breakdown = await result.cost_breakdown()
print(f"\nBy evaluator: {breakdown['by_evaluator']}")
print(f"By model: {breakdown['by_model']}")
```

### Cost Comparison Example

Compare evaluation costs across different models:

```python
# Test with expensive model
result_gpt4 = await evaluate(
    output=output, reference=reference,
    model="gpt-4o", evaluators=["semantic"]
)

# Test with cheaper model
result_mini = await evaluate(
    output=output, reference=reference,
    model="gpt-4o-mini", evaluators=["semantic"]
)

cost_gpt4 = await result_gpt4.total_llm_cost()
cost_mini = await result_mini.total_llm_cost()

print(f"GPT-4o: ${cost_gpt4:.6f}")
print(f"GPT-4o-mini: ${cost_mini:.6f}")
print(f"Savings: {((cost_gpt4 - cost_mini) / cost_gpt4 * 100):.1f}%")
print(f"Score difference: {abs(result_gpt4.overall_score - result_mini.overall_score):.3f}")
```

**Result**: GPT-4o-mini often gives similar quality at 80%+ cost savings.

## Key Features

- **Simple API**: Evaluate LLM outputs with 3 lines of code
- **Automatic Observability**: Automatic LLM interaction tracking with cost and performance metrics
- **Provider-Agnostic**: Works with any model from OpenAI, Anthropic, Google, Groq, Mistral, or Cohere (via PydanticAI)
- **Middleware Pipeline**: Logging, metrics, caching, rate limiting
- **Semantic Evaluation**: Similarity scoring with LLM or FAISS backends (significantly faster, zero cost for embeddings)
- **Custom Criteria**: Domain-specific evaluation (medical, technical, brand voice)
- **Comparison Mode**: A/B testing with `compare()` API for pairwise evaluation
- **Multiple Evaluators**: Combine semantic, custom_criteria, pairwise, factuality, groundedness, and relevance evaluators
- **Registry System**: Register custom evaluators for extensibility
- **Factuality Evaluation**: Hallucination detection and fact verification
- **Groundedness Evaluation**: RAG system validation (source attribution)
- **Relevance Evaluation**: Query-output alignment assessment

## Core Concepts

### Evaluators

Evaluators assess LLM outputs against criteria:

**Semantic Similarity**
```python
from arbiter_ai import evaluate

# LLM backend (default) - Rich explanations
result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

# FAISS backend (optional) - significantly faster, zero cost for embeddings
# Requires: pip install arbiter-ai[scale]
from arbiter_ai import SemanticEvaluator, LLMManager

client = await LLMManager.get_client(model="gpt-4o-mini")
evaluator = SemanticEvaluator(client, backend="faiss")
score = await evaluator.evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris"
)
print(f"Similarity: {score.value:.2f}")  # Fast, free, deterministic
```

**Custom Criteria** (No reference needed!)
```python
# Evaluate against domain-specific criteria
result = await evaluate(
    output="Medical advice about diabetes management",
    criteria="Medical accuracy, HIPAA compliance, appropriate tone for patients",
    evaluators=["custom_criteria"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Criteria met: {result.scores[0].metadata['criteria_met']}")
print(f"Criteria not met: {result.scores[0].metadata['criteria_not_met']}")
```

**Pairwise Comparison** (A/B Testing)
```python
from arbiter_ai import compare, PairwiseComparisonEvaluator, LLMManager

# Option 1: High-level API
comparison = await compare(
    output_a="GPT-4 response",
    output_b="Claude response",
    criteria="accuracy, clarity, completeness",
    model="gpt-4o-mini"
)
print(f"Winner: {comparison.winner}")  # output_a, output_b, or tie
print(f"Confidence: {comparison.confidence:.2f}")

# Option 2: Direct evaluator (supports evaluate() too)
client = await LLMManager.get_client(model="gpt-4o-mini")
evaluator = PairwiseComparisonEvaluator(client)

# Pattern 1: compare() for explicit A/B comparison
comparison = await evaluator.compare(output_a="...", output_b="...")

# Pattern 2: evaluate() for output vs reference
score = await evaluator.evaluate(output="...", reference="...")
print(f"Score: {score.value:.2f}")  # High if output > reference, low if reference > output
```

**Multiple Evaluators**
```python
# Combine multiple evaluators for comprehensive assessment
result = await evaluate(
    output="Your LLM output",
    reference="Expected output",
    criteria="Accuracy, clarity, completeness",
    evaluators=["semantic", "custom_criteria"],
    model="gpt-4o-mini"
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Individual Scores: {len(result.scores)}")
for score in result.scores:
    print(f"  {score.name}: {score.value:.2f}")
```

### Batch Evaluation

Evaluate multiple outputs in parallel with built-in progress tracking and concurrency control:

```python
from arbiter_ai import batch_evaluate

# Efficient batch processing
items = [
    {"output": "Paris is capital of France", "reference": "Paris is France's capital"},
    {"output": "Tokyo is capital of Japan", "reference": "Tokyo is Japan's capital"},
    {"output": "Berlin is capital of Germany", "reference": "Berlin is Germany's capital"},
]

result = await batch_evaluate(
    items=items,
    evaluators=["semantic"],
    model="gpt-4o-mini",
    max_concurrency=5  # Control parallel execution
)

print(f"Success: {result.successful_items}/{result.total_items}")
print(f"Total cost: ${await result.total_llm_cost():.4f}")

# With progress tracking
def on_progress(completed, total, latest):
    print(f"Progress: {completed}/{total}")

result = await batch_evaluate(
    items=items,
    progress_callback=on_progress
)

# Access individual results
for i, eval_result in enumerate(result.results):
    if eval_result:
        print(f"Item {i}: {eval_result.overall_score:.2f}")
    else:
        error = result.get_error(i)
        print(f"Item {i}: FAILED - {error['error']}")
```

See [examples/batch_evaluation_example.py](examples/batch_evaluation_example.py) for comprehensive patterns including error handling and cost breakdown.

### Result Persistence (Storage Backends)

Arbiter supports optional storage backends for persisting evaluation results:

**PostgreSQL** (persistent storage):
```bash
pip install arbiter[postgres]
```

```python
from arbiter_ai import evaluate
from arbiter_ai.storage import PostgresStorage

storage = PostgresStorage()  # Uses DATABASE_URL from environment

async with storage:
    result = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        model="gpt-4o-mini"
    )

    # Save to PostgreSQL
    result_id = await storage.save_result(result)

    # Retrieve later
    retrieved = await storage.get_result(result_id)
```

**Redis** (fast caching with TTL):
```bash
pip install arbiter[redis]
```

```python
from arbiter_ai.storage import RedisStorage

storage = RedisStorage(ttl=3600)  # 1 hour cache

async with storage:
    result = await evaluate(...)
    result_id = await storage.save_result(result)

    # Fast retrieval from cache
    cached = await storage.get_result(result_id)
```

**Setup**:
1. Set `DATABASE_URL` and/or `REDIS_URL` in your `.env` file
2. For PostgreSQL: Run migrations with `alembic upgrade head`
3. Use storage backends in your evaluation code

See [examples/storage_postgres_example.py](examples/storage_postgres_example.py) and [examples/storage_redis_example.py](examples/storage_redis_example.py) for complete examples.

### RAG System Evaluation

Evaluate Retrieval-Augmented Generation systems comprehensively:

```python
from arbiter_ai import evaluate

# Evaluate RAG response with multiple aspects
result = await evaluate(
    output=rag_answer,
    reference=expected_answer,
    criteria="Accuracy, completeness, source attribution, no hallucination",
    evaluators=["semantic", "custom_criteria"],
    model="gpt-4o-mini"
)

# Check for hallucinations and source attribution
if result.scores[0].metadata.get("criteria_not_met"):
    print("WARNING: Potential hallucination detected")
```

See [examples/rag_evaluation.py](examples/rag_evaluation.py) for complete RAG evaluation patterns.

## Architecture

Built on proven patterns with type-safe foundations:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Public API                              │
│                  evaluate() | compare() | batch_evaluate()      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    Middleware Pipeline                          │
│         Logging → Metrics → Caching → Rate Limiting             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Evaluators                               │
│  Semantic | CustomCriteria | Pairwise | Factuality | ...        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              BasePydanticEvaluator                      │    │
│  │  Template Method: 4 abstract methods per evaluator      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     LLM Client Layer                            │
│              Provider-Agnostic (via PydanticAI)                 │
│    OpenAI | Anthropic | Google | Groq | Mistral | Cohere        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Infrastructure                             │
│  Cost Calculator | Circuit Breaker | Retry | Monitoring         │
│  Storage: PostgreSQL | Redis                                    │
└─────────────────────────────────────────────────────────────────┘
```

- **PydanticAI 1.14+**: Structured LLM interactions with type safety
- **Template Method Pattern**: Consistent evaluator implementation
- **Middleware Pipeline**: Composable logging, metrics, caching, rate limiting
- **Provider-Agnostic Design**: Works with any LLM provider

## Examples

18 comprehensive examples demonstrating all features.

**Running Examples:**
```bash
# With uv (recommended)
uv run python examples/basic_evaluation.py

# Or if you activated venv manually
python examples/basic_evaluation.py
```

**Getting Started:**
- [Debugging Multi-Call Systems](examples/debugging_multi_call.py) - The black box problem solved
- [Basic Evaluation](examples/basic_evaluation.py) - Simple semantic evaluation with cost tracking
- [Cost Comparison](examples/cost_comparison.py) - Model cost/quality analysis
- [Multiple Evaluators](examples/multiple_evaluators.py) - Combining evaluators

**Evaluators:**
- [Semantic Similarity](examples/faiss_semantic_example.py) - LLM and FAISS backends
- [Custom Criteria](examples/custom_criteria_example.py) - Domain-specific evaluation
- [Pairwise Comparison](examples/pairwise_comparison_example.py) - A/B testing with compare()
- [Pairwise evaluate()](examples/pairwise_evaluate_example.py) - Output vs reference comparison
- [Factuality](examples/factuality_example.py) - Hallucination detection
- [Groundedness](examples/groundedness_example.py) - RAG validation
- [Relevance](examples/relevance_example.py) - Query alignment

**Advanced Features:**
- [Batch Evaluation](examples/batch_evaluation_example.py) - Parallel processing with progress tracking
- [RAG Evaluation](examples/rag_evaluation.py) - Complete RAG system evaluation
- [Observability](examples/observability_example.py) - Interaction tracking and debugging
- [Error Handling](examples/error_handling_example.py) - Handling failures gracefully
- [Middleware Usage](examples/middleware_usage.py) - Logging, metrics, caching
- [Circuit Breaker](examples/circuit_breaker_example.py) - Fault tolerance patterns
- [Provider Switching](examples/provider_switching.py) - Multi-provider support
- [Evaluator Registry](examples/evaluator_registry_example.py) - Custom evaluators
- [Advanced Config](examples/advanced_config.py) - Temperature, retries, custom clients

## Development

```bash
# Clone and setup
git clone https://github.com/ashita-ai/arbiter.git
cd arbiter

# Install with development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Or use make commands
make test          # Run tests
make test-cov      # Tests with coverage
make lint          # Check code quality
make format        # Format code
make type-check    # Type checking
```

## Roadmap

**Note:** This is a personal project. Roadmap items are ideas and explorations, not commitments. Priorities and timelines may change based on what's useful.

**Completed**
- [x] Core evaluation engine with PydanticAI
- [x] SemanticEvaluator, CustomCriteriaEvaluator, PairwiseComparisonEvaluator
- [x] FactualityEvaluator, GroundednessEvaluator, RelevanceEvaluator
- [x] Batch evaluation API
- [x] Automatic cost tracking and observability
- [x] FAISS backend for faster semantic evaluation
- [x] Storage backends (PostgreSQL + Redis)
- [x] PyPI package publication (arbiter-ai)
- [x] Enhanced factuality with external verification plugins (SearchVerifier, CitationVerifier, KnowledgeBaseVerifier)

**Future Ideas** (No timeline, exploring as needed)
- [ ] Additional evaluators for specific domains (medical, legal, technical writing)

**Contributions welcome!** This is a personal project, but if you find it useful and want to contribute, pull requests are appreciated.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

**Areas where we'd love help:**
- Additional evaluators for specific domains
- Performance optimizations
- Documentation improvements
- Integration examples with popular frameworks
