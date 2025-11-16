<div align="center">
  <h1>Arbiter</h1>

  <p><strong>LLM Evaluations with automatic interaction tracking, multiple evaluators, and extensible architecture</strong></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/evanvolgas/arbiter"><img src="https://img.shields.io/badge/version-0.1.0--alpha-blue" alt="Version"></a>
  </p>

  <p><em>⚠️ Alpha Software: Early development stage. Use for evaluation and experimentation.</em></p>
</div>

---

## Why Arbiter?

**The Problem:** Teams building with LLMs need to evaluate output quality at scale. Manual review becomes impractical beyond a few examples, and writing custom evaluation code for every use case is time-consuming.

**What Arbiter Provides:**
- **Simple API**: Evaluate LLM outputs with 3 lines of code
- **Automatic Tracking**: Complete visibility into LLM interactions, costs, and performance
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq, Mistral, and Cohere
- **Production-Ready**: Built-in retry logic, connection pooling, and middleware support

**Use Case Example:**
A customer support chatbot needs quality evaluation. Arbiter provides:
1. Automated evaluation against defined criteria
2. Detailed scoring with explanations
3. Complete audit trail of all LLM interactions
4. Cost and performance metrics per evaluation

---

## What is Arbiter?

Arbiter is an LLM evaluation framework that provides simple APIs, automatic observability, and provider-agnostic infrastructure.

Evaluate LLM outputs with 3 lines of code while maintaining visibility into cost, quality, and decision-making processes.

**Core Value**: A pragmatic approach to LLM evaluation without complexity, vendor lock-in, or hidden costs.

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

# Simple evaluation with automatic client management
result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Passed: {result.passed}")
print(f"Interactions: {len(result.interactions)}")
```

Or use evaluators directly for more control:

```python
from arbiter import SemanticEvaluator, LLMManager

# Get LLM client
client = await LLMManager.get_client(model="gpt-4o-mini")

# Create and use evaluator
evaluator = SemanticEvaluator(client)
score = await evaluator.evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris"
)

print(f"Semantic similarity: {score.value:.2f}")
print(f"Confidence: {score.confidence:.2f}")
```

## Key Features

- **✅ Simple API**: Evaluate LLM outputs with 3 lines of code
- **✅ Automatic Observability**: Automatic LLM interaction tracking with cost and performance metrics
- **✅ Provider-Agnostic**: OpenAI, Anthropic, Google, Groq, Mistral, Cohere support
- **✅ Middleware Pipeline**: Logging, metrics, caching, rate limiting
- **✅ Semantic Evaluation**: Similarity scoring with LLM or FAISS backends (significantly faster, zero cost for embeddings)
- **✅ Custom Criteria**: Domain-specific evaluation (medical, technical, brand voice)
- **✅ Comparison Mode**: A/B testing with `compare()` API for pairwise evaluation
- **✅ Multiple Evaluators**: Combine semantic, custom_criteria, pairwise, factuality, groundedness, and relevance evaluators
- **✅ Registry System**: Register custom evaluators for extensibility
- **✅ Factuality Evaluation**: Hallucination detection and fact verification
- **✅ Groundedness Evaluation**: RAG system validation (source attribution)
- **✅ Relevance Evaluation**: Query-output alignment assessment

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

**Multiple Evaluators** ✅
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
    print("⚠️ Potential hallucination detected")
```

See [examples/rag_evaluation.py](examples/rag_evaluation.py) for complete RAG evaluation patterns.

## Architecture

Built on proven patterns with type-safe foundations:

- **PydanticAI 1.14+**: Structured LLM interactions with type safety
- **Template Method Pattern**: Consistent evaluator implementation
- **Middleware Pipeline**: Composable logging, metrics, caching, rate limiting
- **Provider-Agnostic Design**: Works with any LLM provider

## Examples

15 comprehensive examples demonstrating all evaluators:
- [Basic Evaluation](examples/basic_evaluation.py) - Simple semantic evaluation
- [Multiple Evaluators](examples/multiple_evaluators.py) - Combining evaluators
- [Custom Criteria](examples/custom_criteria_example.py) - Domain-specific evaluation
- [Pairwise Comparison](examples/pairwise_comparison_example.py) - A/B testing
- [Factuality Evaluation](examples/factuality_example.py) - Hallucination detection ✅
- [Groundedness Evaluation](examples/groundedness_example.py) - RAG validation ✅
- [Relevance Evaluation](examples/relevance_example.py) - Query alignment ✅
- [Batch Processing](examples/batch_manual.py) - Manual batching patterns
- [Advanced Config](examples/advanced_config.py) - Temperature, retries, custom clients
- [Interaction Tracking](examples/interaction_tracking_example.py) - Complete observability
- [RAG Evaluation](examples/rag_evaluation.py) - RAG system evaluation
- [Error Handling](examples/error_handling_example.py) - Handling failures gracefully
- [Middleware Usage](examples/middleware_usage.py) - Logging, metrics, caching
- [Provider Switching](examples/provider_switching.py) - Multi-provider support
- [Evaluator Registry](examples/evaluator_registry_example.py) - Custom evaluators

## Development

```bash
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter
pip install -e ".[dev]"
pytest
```

## Roadmap

**Phase 1 - Foundation** ✅ (Completed)
- [x] Project setup and structure
- [x] Core infrastructure (LLM client, middleware, monitoring)
- [x] Exception handling and retry logic

**Phase 2 - Core Evaluation** ✅ (Completed)
- [x] Core evaluation engine with PydanticAI
- [x] BasePydanticEvaluator with automatic LLM tracking
- [x] SemanticEvaluator implementation
- [x] Main evaluate() API
- [x] Complete observability (interaction tracking)

**Phase 2.5 - Fill Critical Gaps** ✅ (Completed)
- [x] CustomCriteriaEvaluator (domain-specific evaluation)
- [x] PairwiseComparisonEvaluator (A/B testing)
- [x] FAISS backend for SemanticEvaluator (faster than LLM-based, zero cost for embeddings)
- [x] Multi-evaluator error handling with partial results
- [x] 12 comprehensive examples (basic, custom criteria, pairwise, batch, advanced config, RAG evaluation, etc.)
- [x] Complete API documentation (16 API reference pages + MkDocs setup)
- [x] Evaluator registry system for extensibility

**Phase 3 - Core Evaluators** ✅ (Complete - 2 days, accelerated from 3 weeks)
- [x] FactualityEvaluator (hallucination detection) - 100% coverage
- [x] GroundednessEvaluator (RAG validation) - 100% coverage
- [x] RelevanceEvaluator (query alignment) - 100% coverage

**Phase 4 - Batch Evaluation** 📋 (1 week)
- [ ] Batch evaluation API (`batch_evaluate()` function)
- [ ] Parallel processing with progress tracking
- ⏸️ **Storage backends deferred to v2.0** (users can persist results manually)

**Phase 5 - Enhanced Factuality** 📋 (6 weeks)
- [ ] Plugin infrastructure for external verification
- [ ] TavilyPlugin (web search for fact-checking)
- [ ] FAISS-based fact verification caching (reuses existing FAISS backend from Phase 2.5)
- [ ] Atomic claim decomposition
- [ ] Additional plugins (Wikidata, Wolfram, PubMed)
- ⏸️ **Milvus deferred to v2.0+** (FAISS covers scale needs for v1.0)

**Phase 6 - Polish & v1.0 Release** 📋 (2 weeks)
- [ ] PyPI package publication
- [ ] CI/CD pipeline setup
- [ ] Documentation site deployment
- [ ] v1.0 release announcement

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with inspiration from [Sifaka](https://sifaka.ai) and leveraging proven patterns for AI systems.
