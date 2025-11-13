<div align="center">
  <img src="arbiter/arbiter.png" alt="Arbiter Logo" width="200"/>

  # Arbiter

  **Production-grade LLM evaluation framework with complete observability**

  [![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
  [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/evanvolgas/arbiter)
</div>

## What is Arbiter?

Arbiter is a production-grade LLM evaluation framework that provides simple APIs, complete observability, and provider-agnostic infrastructure for AI teams at scale.

Evaluate LLM outputs with 3 lines of code while maintaining full visibility into cost, quality, and decision-making processes.

**Core Value**: The pragmatic choice for teams that need reliable evaluation without complexity, vendor lock-in, or hidden costs.

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
- **✅ Complete Observability**: Automatic LLM interaction tracking (unique differentiator)
- **✅ Provider-Agnostic**: OpenAI, Anthropic, Google, Groq, Mistral, Cohere support
- **✅ Production-Ready**: Middleware for logging, metrics, caching, rate limiting
- **✅ Semantic Evaluation**: Similarity scoring with confidence levels
- **✅ Custom Criteria**: Domain-specific evaluation (medical, technical, brand voice)
- **✅ Comparison Mode**: A/B testing with `compare()` API for pairwise evaluation
- **📋 Multiple Evaluators**: Factuality, consistency, relevance (Phase 5+)

## Core Concepts

### Evaluators

Evaluators assess LLM outputs against criteria:

**Semantic Similarity**
```python
from arbiter import evaluate

# Compare output to reference text
result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)
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

**Multiple Evaluators** (Coming soon)
```python
# result = await evaluate(
#     output="Your LLM output",
#     reference="Expected output",
#     evaluators=["semantic", "custom_criteria", "factuality"],
#     model="gpt-4o-mini"
# )
```

### Batch Evaluation (Coming Soon)

Process multiple outputs efficiently:

```python
# Planned API for Phase 4
# from arbiter import batch_evaluate
#
# outputs = ["Output 1", "Output 2", "Output 3"]
# references = ["Reference 1", "Reference 2", "Reference 3"]
#
# results = await batch_evaluate(outputs, references, evaluators=["semantic"])
```

### Streaming (Optional)

Integrate with streaming pipelines:

```python
from arbiter.streaming import ByteWaxAdapter

async for batch in kafka_source:
    results = await evaluator.batch_score(batch)
    await sink.send(results)
```

## Architecture

Built on proven patterns with production-grade foundations:

- **PydanticAI 1.14+**: Structured LLM interactions with type safety
- **Template Method Pattern**: Consistent evaluator implementation
- **Middleware Pipeline**: Composable logging, metrics, caching, rate limiting
- **Provider-Agnostic Design**: Works with any LLM provider

See [DESIGN_SPEC.md](DESIGN_SPEC.md) for complete architecture details.

## Documentation

- **[DESIGN_SPEC.md](DESIGN_SPEC.md)** - Vision, architecture, and competitive analysis
- **[AGENTS.md](AGENTS.md)** - How to contribute and work with this repository
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Complete roadmap with all phases
- **[PROJECT_TODO.md](PROJECT_TODO.md)** - Current milestone tracker (Phase 2.5)
- **[PHASE2_REVIEW.md](PHASE2_REVIEW.md)** - Comprehensive Phase 2 assessment

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

**Phase 2.5 - Fill Critical Gaps** 🚧 (Current - Nov 22 to Dec 12)
- [x] CustomCriteriaEvaluator (domain-specific evaluation)
- [x] PairwiseComparisonEvaluator (A/B testing)
- [ ] Multi-evaluator error handling
- [ ] 10-15 comprehensive examples
- [ ] Complete API documentation

**Phase 3 - Semantic Comparison** 📋 (Next - 2 weeks)
- [ ] Milvus integration for vector storage
- [ ] Embedding generation pipeline
- [ ] Vector similarity scoring

**Phase 4 - Storage & Scale** 📋 (2-3 weeks)
- [ ] Storage backends (Memory, File, Redis)
- [ ] Batch operations with parallel processing
- [ ] ByteWax streaming adapter

**Phase 5 - Additional Evaluators** 📋 (4-5 weeks)
- [ ] Factuality evaluator (HIGHEST PRIORITY)
- [ ] Relevance evaluator
- [ ] Toxicity evaluator
- [ ] Groundedness evaluator
- [ ] Consistency evaluator
- [ ] ContextRelevance evaluator

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete 9-phase roadmap.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with inspiration from [Sifaka](https://sifaka.ai) and leveraging patterns for production-grade AI systems.
