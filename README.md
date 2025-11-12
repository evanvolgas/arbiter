<div align="center">
  <img src="arbiter/arbiter.png" alt="Arbiter Logo" width="200"/>

  # Arbiter

  **Streaming LLM evaluation framework with semantic comparison and composable metrics**

  [![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
  [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/evanvolgas/arbiter)
</div>

## What is Arbiter?

Arbiter provides composable evaluation primitives for assessing LLM outputs through research-backed metrics. Instead of simple string matching, Arbiter uses semantic comparison, factuality checking, and consistency verification to provide meaningful quality scores.

**Core Value**: Evaluate LLM outputs at scale with semantic understanding and complete observability.

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
from arbiter import evaluate, SemanticEvaluator

# Simple evaluation
evaluator = SemanticEvaluator(model="gpt-4o")

score = await evaluator.score(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    criteria="factuality"
)

print(f"Score: {score.value}")  # High score for semantic match
```

## Key Features

- **🎯 Semantic Comparison**: Milvus-backed vector similarity for deep comparison
- **📊 Multiple Metrics**: Factuality, consistency, semantic similarity, custom criteria
- **🔌 Composable**: Build custom evaluation pipelines with middleware
- **⚡ Streaming Ready**: Optional ByteWax integration for streaming data
- **👁️ Complete Observability**: Full audit trails and performance metrics
- **🧩 Extensible**: Plugin system for custom evaluators and storage

## Core Concepts

### Evaluators

Evaluators assess LLM outputs against criteria:

```python
# Semantic similarity
semantic_score = await evaluator.score(output, reference, "semantic_similarity")

# Factuality checking
factuality_score = await evaluator.score(output, reference, "factuality")

# Custom criteria
custom_score = await evaluator.score(output, reference, "matches company tone")
```

### Batch Evaluation

Process multiple outputs efficiently:

```python
outputs = ["Output 1", "Output 2", "Output 3"]
references = ["Reference 1", "Reference 2", "Reference 3"]

scores = await evaluator.batch_score(outputs, references)
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

Built on proven patterns from Sifaka with focus on evaluation:

- **PydanticAI**: Structured LLM interactions
- **Milvus**: Vector storage for semantic comparison
- **Middleware**: Logging, metrics, caching, rate limiting
- **Plugin System**: Extensible evaluators and storage backends

## Development

```bash
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter
pip install -e ".[dev]"
pytest
```

## Roadmap

**Phase 1 - Foundation** ✅
- [x] Project setup and structure
- [x] Core infrastructure (LLM client, middleware, monitoring)
- [x] Exception handling and retry logic

**Phase 2 - Core Evaluation** ✅
- [x] Core evaluation engine with PydanticAI
- [x] BasePydanticEvaluator with automatic LLM tracking
- [x] SemanticEvaluator implementation
- [x] Main evaluate() API
- [x] Complete observability (interaction tracking)

**Phase 3 - Semantic Comparison** 🚧
- [ ] Milvus integration for vector storage
- [ ] Embedding generation pipeline
- [ ] Vector similarity scoring

**Phase 4 - Storage & Scale** 📋
- [ ] Storage backends (Memory, File, Redis)
- [ ] Batch operations with parallel processing
- [ ] ByteWax streaming adapter

**Phase 5 - Additional Evaluators** 📋
- [ ] Factuality evaluator
- [ ] Consistency evaluator
- [ ] Relevance evaluator
- [ ] Custom criteria support

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with inspiration from [Sifaka](https://sifaka.ai) and leveraging patterns for production-grade AI systems.
