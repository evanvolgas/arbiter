# Installation Guide

Complete guide to installing and setting up Arbiter for LLM evaluation.

## Quick Install

**Note:** Arbiter is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter

# Using pip
pip install -e .

# Using uv (recommended for speed)
uv pip install -e .
```

## Development Installation

For contributing or development work:

```bash
# Clone the repository
git clone https://github.com/evanvolgas/arbiter.git
cd arbiter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

## API Keys Setup

Arbiter requires API keys for the LLM providers you want to use.

### Step 1: Create .env File

```bash
# Copy the example environment file
cp .env.example .env
```

### Step 2: Add Your API Keys

Edit `.env` and add your keys:

```bash
# Required for most examples
OPENAI_API_KEY=sk-...

# Optional: Add other providers as needed
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...
```

### Getting API Keys

**OpenAI** (required for examples)
- Go to https://platform.openai.com/api-keys
- Create a new API key
- Add to `.env` as `OPENAI_API_KEY=sk-...`

**Anthropic** (optional)
- Go to https://console.anthropic.com/settings/keys
- Create a new API key
- Add to `.env` as `ANTHROPIC_API_KEY=sk-ant-...`

**Google AI** (optional)
- Go to https://makersuite.google.com/app/apikey
- Create a new API key
- Add to `.env` as `GOOGLE_API_KEY=...`

**Groq** (optional - fast inference)
- Go to https://console.groq.com/keys
- Create a new API key
- Add to `.env` as `GROQ_API_KEY=...`

## Verification

Verify your installation works:

```python
import asyncio
from arbiter import evaluate

async def test_installation():
    """Test that Arbiter is installed and configured correctly."""
    result = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        model="gpt-4o-mini"
    )

    print(f"✅ Installation successful!")
    print(f"Score: {result.overall_score:.2f}")
    print(f"Passed: {result.passed}")
    return result

# Run the test
result = asyncio.run(test_installation())
```

Expected output:
```
✅ Installation successful!
Score: 0.95
Passed: True
```

## Troubleshooting Installation

### Problem: ModuleNotFoundError: No module named 'arbiter'

**Solution:**
```bash
# Make sure you're in the right environment
which python  # Should point to your venv

# Reinstall
pip install -e .
```

### Problem: ImportError: cannot import name 'evaluate'

**Solution:**
```bash
# You may have an old version
pip uninstall arbiter
pip install -e .
```

### Problem: API key not found

**Solution:**
1. Verify `.env` file exists in your working directory
2. Check the key name matches exactly (e.g., `OPENAI_API_KEY`)
3. Ensure no extra spaces or quotes around the key
4. Examples automatically load `.env` using `python-dotenv`

### Problem: Rate limit errors

**Solution:**
```bash
# Use a cheaper model for testing
model="gpt-4o-mini"  # Instead of gpt-4

# Or add rate limiting
from arbiter import evaluate

result = await evaluate(
    output="...",
    model="gpt-4o-mini",
    # Rate limiting handled automatically by middleware
)
```

## Next Steps

- Read the [Quick Start](../../README.md#quick-start) guide
- Try the [examples](../../examples/)
- Learn about [custom evaluators](custom-evaluators.md)
- Check the [API documentation](../api/)

## System Requirements

- Python 3.10 or higher
- Internet connection (for LLM API calls)
- API key for at least one LLM provider (OpenAI recommended)

## Optional Dependencies

### For Development
```bash
pip install -e ".[dev]"
```

Includes:
- pytest (testing)
- black (formatting)
- ruff (linting)
- mypy (type checking)

### For Documentation
```bash
pip install -e ".[docs]"
```

Includes:
- mkdocs (documentation site)
- mkdocs-material (theme)

## Platform-Specific Notes

### macOS/Linux
```bash
source .venv/bin/activate
```

### Windows
```bash
.venv\Scripts\activate
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

CMD ["python", "your_script.py"]
```

## Support

If you encounter issues:
1. Check the [troubleshooting guide](troubleshooting.md)
2. Search [existing issues](https://github.com/evanvolgas/arbiter/issues)
3. Open a new issue with your error message and environment details
