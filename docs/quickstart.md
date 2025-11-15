# Quick Start Guide

Get started with Arbiter in 5 minutes!

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

## Set Up API Keys

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
# Or for other providers:
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

## Your First Evaluation

```python
import asyncio
from arbiter import evaluate

async def main():
    result = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        model="gpt-4o-mini"
    )

    print(f"Score: {result.overall_score:.2f}")
    print(f"Passed: {result.passed}")

asyncio.run(main())
```

## Next Steps

- [Examples](examples/index.md) - See more examples
- [API Reference](api/index.md) - Complete API docs
- [Guides](guides/evaluator-registry.md) - In-depth tutorials

