# Arbiter Documentation

Welcome to the Arbiter documentation!

**Arbiter** is a production-grade LLM evaluation framework that provides simple APIs, complete observability, and provider-agnostic infrastructure for AI teams at scale.

## Quick Links

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [API Reference](api/index.md) - Complete API documentation
- [Examples](examples/index.md) - Code examples for common use cases
- [Guides](guides/evaluator-registry.md) - In-depth guides and tutorials

## Key Features

- ✅ **Simple API**: Evaluate with 3 lines of code
- ✅ **Complete Observability**: Automatic LLM interaction tracking
- ✅ **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq, Mistral, Cohere
- ✅ **Production-Ready**: Middleware, error handling, retry logic
- ✅ **Extensible**: Registry system for custom evaluators

## Installation

**Note:** Arbiter is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter
pip install -e .
```

With uv (faster):

```bash
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter
uv pip install -e .
```

## Quick Example

```python
from arbiter import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Interactions: {len(result.interactions)}")
```

## Documentation Structure

- **Quick Start**: Get up and running quickly
- **Examples**: Real-world code examples
- **API Reference**: Complete API documentation
- **Guides**: In-depth tutorials and best practices
- **Development**: Contributing and architecture docs

## Need Help?

- Check the [Troubleshooting Guide](guides/troubleshooting.md)
- Review the [Examples](examples/index.md)
- Read the [API Reference](api/index.md)

