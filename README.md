<div align="center">
  <h1>Arbiter</h1>

  <p><strong>Native PydanticAI evaluation with automatic cost tracking</strong></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/evanvolgas/arbiter"><img src="https://img.shields.io/badge/version-0.1.0--alpha-blue" alt="Version"></a>
    <a href="https://ai.pydantic.dev"><img src="https://img.shields.io/badge/PydanticAI-native-purple" alt="PydanticAI"></a>
  </p>

  <p><em>Alpha Software: Early development stage. Use for evaluation and experimentation.</em></p>
</div>

---

## Why Arbiter?

Stop guessing what your LLM evaluations cost. Arbiter shows you exactly what happened in every evaluation - with **automatic cost tracking**, complete interaction visibility, and native PydanticAI integration.

```python
from arbiter import evaluate

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

**Status**: Alpha software (v0.1.0-alpha). Functional but early-stage. Best suited for evaluation, experimentation, and development. Not recommended for mission-critical use yet.

## Installation

```bash
# Clone the repository
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter

# Install with uv (recommended)
uv pip install -e .

# Or with standard pip
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

**Note:** All examples automatically load environment variables from `.env` using `python-dotenv`.

## Quick Start

```python
from arbiter import evaluate

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
- **Provider-Agnostic**: OpenAI, Anthropic, Google, Groq, Mistral, Cohere support
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
from arbiter import evaluate

# LLM backend (default) - Rich explanations
result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

# FAISS backend (optional) - significantly faster, zero cost for embeddings
# Requires: pip install arbiter[scale]
from arbiter import SemanticEvaluator, LLMManager

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
from arbiter import batch_evaluate

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

**Note:** Result persistence (storage backends) is deferred to Phase 2.0. For now, you can persist results manually:
```python
import json
from arbiter import evaluate

result = await evaluate(output="...", reference="...")
# Save results yourself
with open("results.json", "w") as f:
    json.dump(result.model_dump(), f, indent=2)
```

### RAG System Evaluation

Evaluate Retrieval-Augmented Generation systems comprehensively:

```python
from arbiter import evaluate

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

- **PydanticAI 1.14+**: Structured LLM interactions with type safety
- **Template Method Pattern**: Consistent evaluator implementation
- **Middleware Pipeline**: Composable logging, metrics, caching, rate limiting
- **Provider-Agnostic Design**: Works with any LLM provider

## Examples

17 comprehensive examples demonstrating all features:

**Getting Started:**
- [Basic Evaluation](examples/basic_evaluation.py) - Simple semantic evaluation with cost tracking
- [Cost Comparison](examples/cost_comparison.py) - NEW: Model cost/quality analysis
- [Multiple Evaluators](examples/multiple_evaluators.py) - Combining evaluators

**Evaluators:**
- [Semantic Similarity](examples/faiss_semantic_example.py) - LLM and FAISS backends
- [Custom Criteria](examples/custom_criteria_example.py) - Domain-specific evaluation
- [Factuality](examples/factuality_example.py) - Hallucination detection
- [Groundedness](examples/groundedness_example.py) - RAG validation
- [Relevance](examples/relevance_example.py) - Query alignment
- [Pairwise Comparison](examples/pairwise_comparison_example.py) - A/B testing

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
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter
pip install -e ".[dev]"
pytest
```

## Roadmap

**Phase 1 - Foundation** (Completed)
- [x] Project setup and structure
- [x] Core infrastructure (LLM client, middleware, monitoring)
- [x] Exception handling and retry logic

**Phase 2 - Core Evaluation** (Completed)
- [x] Core evaluation engine with PydanticAI
- [x] BasePydanticEvaluator with automatic LLM tracking
- [x] SemanticEvaluator implementation
- [x] Main evaluate() API
- [x] Complete observability (interaction tracking)

**Phase 2.5 - Fill Critical Gaps** (Completed)
- [x] CustomCriteriaEvaluator (domain-specific evaluation)
- [x] PairwiseComparisonEvaluator (A/B testing)
- [x] FAISS backend for SemanticEvaluator (faster than LLM-based, zero cost for embeddings)
- [x] Multi-evaluator error handling with partial results
- [x] 12 comprehensive examples (basic, custom criteria, pairwise, batch, advanced config, RAG evaluation, etc.)
- [x] Complete API documentation (16 API reference pages + MkDocs setup)
- [x] Evaluator registry system for extensibility

**Phase 3 - Core Evaluators** (Complete - 2 days, accelerated from 3 weeks)
- [x] FactualityEvaluator (hallucination detection) - 100% coverage
- [x] GroundednessEvaluator (RAG validation) - 100% coverage
- [x] RelevanceEvaluator (query alignment) - 100% coverage

**Phase 4 - Batch Evaluation** (Complete - 1 day, accelerated from 1 week)
- [x] Batch evaluation API (`batch_evaluate()` function)
- [x] Parallel processing with progress tracking
- [x] Concurrency control and error handling
- **Storage backends deferred to v2.0** (users can persist results manually)

**Phase 5 - Enhanced Factuality** (Planned - 6 weeks)
- [ ] Plugin infrastructure for external verification
- [ ] TavilyPlugin (web search for fact-checking)
- [ ] FAISS-based fact verification caching (reuses existing FAISS backend from Phase 2.5)
- [ ] Atomic claim decomposition
- [ ] Additional plugins (Wikidata, Wolfram, PubMed)
- **Milvus deferred to v2.0+** (FAISS covers scale needs for v1.0)

**Phase 6 - Polish & v1.0 Release** (Planned - 2 weeks)
- [ ] PyPI package publication
- [ ] CI/CD pipeline setup
- [ ] Documentation site deployment
- [ ] v1.0 release announcement

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

**Areas where we'd love help:**
- Additional evaluators for specific domains
- Performance optimizations
- Documentation improvements
- Integration examples with popular frameworks
