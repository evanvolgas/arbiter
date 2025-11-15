# Troubleshooting Guide

Common issues and solutions when using Arbiter.

## Table of Contents

- [Installation Issues](#installation-issues)
- [API Key Problems](#api-key-problems)
- [Provider-Specific Errors](#provider-specific-errors)
- [Performance Issues](#performance-issues)
- [Evaluation Errors](#evaluation-errors)
- [Type Checking Issues](#type-checking-issues)

---

## Installation Issues

### ModuleNotFoundError: No module named 'arbiter'

**Symptoms:**
```python
ImportError: No module named 'arbiter'
```

**Solutions:**

1. Verify you're in the correct virtual environment:
   ```bash
   which python  # Should point to your venv
   ```

2. Reinstall the package:
   ```bash
   pip uninstall arbiter
   pip install -e .
   ```

3. Check Python version (3.10+ required):
   ```bash
   python --version  # Should be 3.10 or higher
   ```

### ImportError: cannot import name 'evaluate'

**Symptoms:**
```python
ImportError: cannot import name 'evaluate' from 'arbiter'
```

**Cause:** Old version or incomplete installation

**Solution:**
```bash
# Clean install
pip uninstall arbiter
rm -rf build/ dist/ *.egg-info
pip install -e .
```

### Dependency Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
```

**Solutions:**

1. Use a fresh virtual environment:
   ```bash
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. Or use `uv` (faster, better dependency resolution):
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

---

## API Key Problems

### ValidationError: API key not found

**Symptoms:**
```python
ValidationError: OPENAI_API_KEY not found in environment
```

**Solutions:**

1. Verify `.env` file exists in your working directory:
   ```bash
   ls -la .env
   ```

2. Check the `.env` format (no spaces, no quotes):
   ```bash
   # Good
   OPENAI_API_KEY=sk-abc123

   # Bad
   OPENAI_API_KEY = sk-abc123  # Extra spaces
   OPENAI_API_KEY="sk-abc123"  # Unnecessary quotes
   ```

3. Ensure `python-dotenv` is installed:
   ```bash
   pip install python-dotenv
   ```

4. Load environment variables manually if needed:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Load .env file

   from arbiter import evaluate
   ```

### Invalid API Key Error

**Symptoms:**
```
AuthenticationError: Invalid API key
```

**Solutions:**

1. Verify the API key is correct:
   ```bash
   echo $OPENAI_API_KEY  # Check the key
   ```

2. Generate a new key from the provider's dashboard
3. Check for extra characters (newlines, spaces):
   ```bash
   # Print key length (OpenAI keys are typically 51 chars)
   python -c "import os; print(len(os.getenv('OPENAI_API_KEY', '')))"
   ```

4. Test the key directly:
   ```python
   import os
   from openai import OpenAI

   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   # Should work without error
   ```

---

## Provider-Specific Errors

### OpenAI Rate Limiting

**Symptoms:**
```
RateLimitError: Rate limit exceeded
```

**Solutions:**

1. Use cheaper models for testing:
   ```python
   # Instead of gpt-4
   model="gpt-4o-mini"  # 15-60x cheaper
   ```

2. Add delays between requests:
   ```python
   import asyncio

   for text in texts:
       result = await evaluate(output=text, model="gpt-4o-mini")
       await asyncio.sleep(1)  # 1 second delay
   ```

3. Use batch processing (coming in Phase 4):
   ```python
   # For now, use asyncio.gather with rate limiting
   import asyncio

   async def evaluate_with_delay(text, delay=0.5):
       await asyncio.sleep(delay)
       return await evaluate(output=text, model="gpt-4o-mini")

   results = await asyncio.gather(*[
       evaluate_with_delay(text, i * 0.5)
       for i, text in enumerate(texts)
   ])
   ```

### Anthropic Context Length Error

**Symptoms:**
```
AnthropicError: prompt is too long
```

**Solutions:**

1. Shorten your input:
   ```python
   # Truncate long texts
   max_length = 5000
   output = text[:max_length] if len(text) > max_length else text
   ```

2. Use a model with larger context:
   ```python
   # Claude 3 models have 200K context
   model="claude-3-5-sonnet-20241022"  # 200K context
   ```

### Google AI (Gemini) Safety Filters

**Symptoms:**
```
GoogleAIError: Content blocked by safety filters
```

**Solutions:**

1. Review the content being evaluated
2. Try different phrasing
3. Use a different provider for sensitive content:
   ```python
   # Switch to OpenAI or Anthropic
   model="gpt-4o-mini"  # or "claude-3-5-sonnet"
   ```

### Groq Service Unavailable

**Symptoms:**
```
GroqError: Service temporarily unavailable
```

**Solutions:**

1. Groq is fast but can have outages - fallback to another provider:
   ```python
   try:
       result = await evaluate(output=text, model="llama-3.1-8b-instant")
   except Exception:
       # Fallback to OpenAI
       result = await evaluate(output=text, model="gpt-4o-mini")
   ```

---

## Performance Issues

### Slow Evaluation (High Latency)

**Symptoms:**
- Evaluations taking >30 seconds
- Timeouts on large batches

**Solutions:**

1. Use faster models:
   ```python
   # Fast models (ranked by speed)
   "llama-3.1-8b-instant"  # Groq - fastest
   "gpt-4o-mini"            # OpenAI - fast & cheap
   "gemini-1.5-flash"       # Google - fast
   "claude-3-5-haiku"       # Anthropic - fast (when available)
   ```

2. Run evaluations in parallel:
   ```python
   import asyncio

   # Parallel evaluation of multiple texts
   results = await asyncio.gather(*[
       evaluate(output=text, model="gpt-4o-mini")
       for text in texts
   ])
   ```

3. Enable caching (if using same texts):
   ```python
   from arbiter.core.middleware import CachingMiddleware

   # Caching reduces API calls for repeated evaluations
   # Automatically enabled in evaluate()
   ```

### High Cost

**Symptoms:**
- Unexpected API bills
- High token usage

**Solutions:**

1. Track token usage:
   ```python
   result = await evaluate(output=text, model="gpt-4o-mini")

   # Check LLM interactions
   for interaction in result.interactions:
       print(f"Tokens: {interaction.token_count}")
       print(f"Cost: ${interaction.cost:.4f}")

   # Total cost
   total_cost = result.total_llm_cost()
   print(f"Total: ${total_cost:.4f}")
   ```

2. Use cheaper models:
   ```python
   # Cost per 1M tokens (input + output)
   "gpt-4o-mini"            # $0.15 + $0.60 = $0.75
   "gemini-1.5-flash"       # $0.075 + $0.30 = $0.375
   "llama-3.1-8b-instant"   # Very cheap on Groq
   ```

3. Shorten prompts:
   ```python
   # Reduce reference text length
   reference = reference_text[:1000]  # First 1000 chars

   # Use criteria instead of long references
   result = await evaluate(
       output=text,
       criteria="accuracy, clarity",  # Instead of full reference
       evaluators=["custom_criteria"]
   )
   ```

### Memory Issues

**Symptoms:**
```
MemoryError: Unable to allocate memory
```

**Solutions:**

1. Process in smaller batches:
   ```python
   batch_size = 10

   for i in range(0, len(texts), batch_size):
       batch = texts[i:i+batch_size]
       results = await asyncio.gather(*[
           evaluate(output=text, model="gpt-4o-mini")
           for text in batch
       ])
       # Process results before next batch
   ```

2. Clear results after processing:
   ```python
   import gc

   for text in large_text_list:
       result = await evaluate(output=text, model="gpt-4o-mini")
       # Extract what you need
       scores.append(result.overall_score)
       # Clear result to free memory
       del result

   gc.collect()  # Force garbage collection
   ```

---

## Evaluation Errors

### Low Scores When Expecting High

**Symptoms:**
- Text seems good but scores low
- Inconsistent scoring

**Solutions:**

1. Check the evaluation criteria:
   ```python
   # Be specific about what "good" means
   result = await evaluate(
       output=text,
       criteria="factual accuracy, clarity, completeness",  # Explicit
       evaluators=["custom_criteria"]
   )

   # Check explanation
   print(result.scores[0].explanation)
   ```

2. Provide a reference for comparison:
   ```python
   result = await evaluate(
       output=output_text,
       reference=expected_text,  # Compare against known good
       evaluators=["semantic"]
   )
   ```

3. Try different models (they vary):
   ```python
   # Some models are stricter/lenient
   models = ["gpt-4o-mini", "claude-3-5-sonnet", "gemini-1.5-flash"]

   for model in models:
       result = await evaluate(output=text, model=model)
       print(f"{model}: {result.overall_score:.2f}")
   ```

### Partial Results (Some Evaluators Failed)

**Symptoms:**
```python
result.partial == True
result.errors == {"factuality": "Timeout error"}
```

**Solutions:**

1. Check which evaluator failed:
   ```python
   if result.partial:
       print(f"Failed evaluators: {list(result.errors.keys())}")
       print(f"Successful scores: {len(result.scores)}")

       for name, error in result.errors.items():
           print(f"{name}: {error}")
   ```

2. Retry failed evaluators:
   ```python
   failed_evaluators = list(result.errors.keys())

   if failed_evaluators:
       retry_result = await evaluate(
           output=text,
           evaluators=failed_evaluators,
           model="gpt-4o-mini"
       )
   ```

3. Set timeouts appropriately:
   ```python
   # Increase timeout for slow evaluators
   import asyncio

   async with asyncio.timeout(60):  # 60 seconds
       result = await evaluate(output=text, model="gpt-4o-mini")
   ```

### ValidationError: Unknown evaluator

**Symptoms:**
```python
ValidationError: Unknown evaluator: 'my_evaluator'
```

**Solutions:**

1. Register custom evaluators before use:
   ```python
   from arbiter import register_evaluator

   register_evaluator("my_evaluator", MyEvaluator)

   # Now can use it
   result = await evaluate(
       output=text,
       evaluators=["my_evaluator"]
   )
   ```

2. Check available evaluators:
   ```python
   from arbiter import get_available_evaluators

   print(get_available_evaluators())
   # ['semantic', 'custom_criteria', 'pairwise']
   ```

3. Use correct evaluator names:
   ```python
   # Good
   evaluators=["semantic", "custom_criteria"]

   # Bad (typo)
   evaluators=["semantics", "custom-criteria"]
   ```

---

## Type Checking Issues

### mypy Errors

**Symptoms:**
```
error: Argument 1 has incompatible type "str"; expected "Optional[str]"
```

**Solutions:**

1. Add type ignores for known safe cases:
   ```python
   result = await evaluate(
       output=text,
       model="gpt-4o-mini"  # type: ignore[arg-type]
   )
   ```

2. Be explicit with types:
   ```python
   from typing import Optional

   reference: Optional[str] = None
   result = await evaluate(output=text, reference=reference)
   ```

3. Update mypy configuration:
   ```toml
   [tool.mypy]
   disallow_untyped_defs = false  # If needed
   ```

### Pydantic ValidationError

**Symptoms:**
```
ValidationError: 1 validation error for Score
value
  ensure this value is less than or equal to 1.0
```

**Solutions:**

1. Ensure scores are in 0-1 range:
   ```python
   # Clamp scores
   score_value = max(0.0, min(1.0, raw_score))
   ```

2. Check response model constraints:
   ```python
   from pydantic import BaseModel, Field

   class MyResponse(BaseModel):
       score: float = Field(ge=0.0, le=1.0)  # Enforces 0-1 range
       confidence: float = Field(default=0.85, ge=0.0, le=1.0)
   ```

---

## Getting More Help

### Enable Debug Logging

```python
import logging

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("arbiter")
logger.setLevel(logging.DEBUG)

# Now evaluate - you'll see detailed logs
result = await evaluate(output=text, model="gpt-4o-mini")
```

### Check LLM Interactions

```python
result = await evaluate(output=text, model="gpt-4o-mini")

# Inspect what was sent to LLM
for interaction in result.interactions:
    print(f"Purpose: {interaction.purpose}")
    print(f"Model: {interaction.model}")
    print(f"Tokens: {interaction.token_count}")
    print(f"Latency: {interaction.latency_ms}ms")
    print(f"Cost: ${interaction.cost:.4f}")
    if interaction.error:
        print(f"Error: {interaction.error}")
```

### Minimal Reproduction

Create a minimal example that reproduces the issue:

```python
import asyncio
from arbiter import evaluate

async def minimal_repro():
    """Minimal reproduction of the issue."""
    result = await evaluate(
        output="Test text",
        reference="Reference text",
        evaluators=["semantic"],
        model="gpt-4o-mini"
    )
    print(result.overall_score)

asyncio.run(minimal_repro())
```

### Report Issues

If you still have problems:

1. Check [existing issues](https://github.com/evanvolgas/arbiter/issues)
2. Open a [new issue](https://github.com/evanvolgas/arbiter/issues/new) with:
   - Python version (`python --version`)
   - Arbiter version (`pip show arbiter`)
   - Minimal reproduction code
   - Error message (full traceback)
   - Expected vs actual behavior

---

## Quick Fixes

### "It's not working"

1. Check API key is set correctly
2. Verify internet connection
3. Try a simple example first
4. Enable debug logging

### "Scores don't make sense"

1. Read the explanation field
2. Check the LLM interactions
3. Try a different model
4. Be more explicit with criteria

### "Too slow"

1. Use `gpt-4o-mini` or `llama-3.1-8b-instant`
2. Run evaluations in parallel
3. Enable caching
4. Shorten inputs

### "Too expensive"

1. Track costs with `total_llm_cost()`
2. Use cheaper models
3. Reduce prompt length
4. Cache repeated evaluations

---

## Related Guides

- [Installation Guide](installation.md)
- [Custom Evaluators Guide](custom-evaluators.md)
- [API Documentation](../api/)
- [Examples](../../examples/)
