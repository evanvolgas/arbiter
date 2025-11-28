# How to Build a Custom Evaluator

This guide shows you how to create custom evaluators for domain-specific evaluation needs in Arbiter.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Template Method Pattern](#template-method-pattern)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Complete Example: ToneEvaluator](#complete-example-toneevaluator)
5. [Best Practices](#best-practices)
6. [Testing Your Evaluator](#testing-your-evaluator)
7. [Advanced Patterns](#advanced-patterns)

## Quick Start

Every custom evaluator extends `BasePydanticEvaluator` and implements 4 required methods:

```python
from arbiter_ai.evaluators.base import BasePydanticEvaluator
from arbiter_ai.core.models import Score
from pydantic import BaseModel, Field
from typing import Optional, Type

class MyEvaluator(BasePydanticEvaluator):
    """Your evaluator description."""

    @property
    def name(self) -> str:
        """Unique identifier for this evaluator."""
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        """Define the evaluator's role and behavior."""
        return "You are an expert at evaluating..."

    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str],
        criteria: Optional[str]
    ) -> str:
        """Create the evaluation prompt for this specific case."""
        return f"Evaluate: {output}"

    def _get_response_type(self) -> Type[BaseModel]:
        """Define the structured response format (optional)."""
        return EvaluatorResponse  # Use default or your custom model

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from the LLM's structured response."""
        return Score(
            name=self.name,
            value=response.score,
            confidence=response.confidence,
            explanation=response.explanation
        )
```

## Template Method Pattern

Arbiter uses the **Template Method Pattern** for consistency:

1. **`name`**: Unique identifier (e.g., "semantic", "custom_criteria", "factuality")
2. **`_get_system_prompt()`**: Establishes the LLM's role as an evaluator
3. **`_get_user_prompt()`**: Provides the specific content to evaluate
4. **`_get_response_type()`**: Defines the Pydantic model for structured output (optional)
5. **`_compute_score()`**: Transforms the LLM response into a Score object

The base class handles:
- LLM client management
- Automatic interaction tracking
- Cost calculation
- Error handling
- Token usage monitoring

## Step-by-Step Tutorial

### Step 1: Define Your Response Model

Create a Pydantic model for the LLM's structured response:

```python
from pydantic import BaseModel, Field

class ToneResponse(BaseModel):
    """Structured response for tone evaluation."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the tone matches criteria (0=poor, 1=excellent)"
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this assessment"
    )
    explanation: str = Field(
        ...,
        description="Why this score was assigned"
    )
    detected_tone: str = Field(
        ...,
        description="The tone detected in the output"
    )
    tone_issues: list[str] = Field(
        default_factory=list,
        description="Specific tone problems found"
    )
```

### Step 2: Create the Evaluator Class

```python
from arbiter_ai.evaluators.base import BasePydanticEvaluator
from arbiter_ai.core.models import Score
from typing import Optional, Type, cast

class ToneEvaluator(BasePydanticEvaluator):
    """Evaluates whether output matches the desired tone.

    Example:
        >>> evaluator = ToneEvaluator(model="gpt-4o-mini")
        >>> score = await evaluator.evaluate(
        ...     output="Hey there! This is totally awesome!!!",
        ...     criteria="Professional business tone, formal language"
        ... )
        >>> print(f"Score: {score.value:.2f}")
    """

    @property
    def name(self) -> str:
        return "tone"
```

### Step 3: Implement System Prompt

The system prompt defines the evaluator's expertise and approach:

```python
    def _get_system_prompt(self) -> str:
        return """You are an expert at evaluating writing tone and style.

Your task is to assess whether text matches a desired tone. Consider:
- Formality level (formal, casual, neutral)
- Language choice (professional, colloquial, technical)
- Emotional tone (enthusiastic, serious, empathetic)
- Audience appropriateness
- Cultural sensitivity

Provide:
- A score from 0.0 (completely wrong tone) to 1.0 (perfect tone match)
- Your confidence in this assessment
- Clear explanation of why the score was assigned
- The tone you detected
- Specific tone issues if the score is low

Be precise and constructive in your evaluation."""
```

### Step 4: Implement User Prompt

The user prompt provides the specific content to evaluate:

```python
    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str],
        criteria: Optional[str]
    ) -> str:
        if not criteria:
            raise ValueError(
                "ToneEvaluator requires criteria specifying desired tone. "
                "Example: criteria='Professional, formal, empathetic'"
            )

        prompt_parts = [
            f"OUTPUT TO EVALUATE:\n{output}\n",
            f"DESIRED TONE:\n{criteria}\n"
        ]

        if reference:
            prompt_parts.append(
                f"REFERENCE EXAMPLE (showing correct tone):\n{reference}\n"
            )

        prompt_parts.append(
            "Evaluate how well the output matches the desired tone. "
            "Provide detailed analysis and actionable feedback."
        )

        return "\n".join(prompt_parts)
```

### Step 5: Specify Response Type

Tell the evaluator which Pydantic model to use:

```python
    def _get_response_type(self) -> Type[BaseModel]:
        return ToneResponse
```

### Step 6: Implement Score Computation

Extract a Score object from the LLM's structured response:

```python
    async def _compute_score(self, response: BaseModel) -> Score:
        tone_response = cast(ToneResponse, response)

        return Score(
            name=self.name,
            value=tone_response.score,
            confidence=tone_response.confidence,
            explanation=tone_response.explanation,
            metadata={
                "detected_tone": tone_response.detected_tone,
                "tone_issues": tone_response.tone_issues,
                "issue_count": len(tone_response.tone_issues)
            }
        )
```

## Complete Example: ToneEvaluator

Here's the full implementation:

```python
"""Tone evaluator for assessing writing style and voice."""

from typing import Optional, Type, cast
from pydantic import BaseModel, Field
from arbiter_ai.evaluators.base import BasePydanticEvaluator
from arbiter_ai.core.models import Score

__all__ = ["ToneEvaluator", "ToneResponse"]


class ToneResponse(BaseModel):
    """Structured response for tone evaluation."""

    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How well the tone matches criteria"
    )
    confidence: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Confidence in this assessment"
    )
    explanation: str = Field(
        ..., description="Why this score was assigned"
    )
    detected_tone: str = Field(
        ..., description="The tone detected in the output"
    )
    tone_issues: list[str] = Field(
        default_factory=list,
        description="Specific tone problems found"
    )


class ToneEvaluator(BasePydanticEvaluator):
    """Evaluates whether output matches the desired tone.

    Example:
        >>> from arbiter_ai import LLMManager
        >>>
        >>> # With model parameter (client created automatically)
        >>> evaluator = ToneEvaluator(model="gpt-4o-mini")
        >>> score = await evaluator.evaluate(
        ...     output="Hey! This is awesome!!!",
        ...     criteria="Professional business tone"
        ... )
        >>> print(f"Score: {score.value:.2f}")
        >>> print(f"Issues: {score.metadata['tone_issues']}")
    """

    @property
    def name(self) -> str:
        return "tone"

    def _get_system_prompt(self) -> str:
        return """You are an expert at evaluating writing tone and style.

Your task is to assess whether text matches a desired tone. Consider:
- Formality level (formal, casual, neutral)
- Language choice (professional, colloquial, technical)
- Emotional tone (enthusiastic, serious, empathetic)
- Audience appropriateness
- Cultural sensitivity

Provide:
- A score from 0.0 (completely wrong tone) to 1.0 (perfect tone match)
- Your confidence in this assessment
- Clear explanation of why the score was assigned
- The tone you detected
- Specific tone issues if the score is low

Be precise and constructive in your evaluation."""

    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str],
        criteria: Optional[str]
    ) -> str:
        if not criteria:
            raise ValueError(
                "ToneEvaluator requires criteria specifying desired tone. "
                "Example: criteria='Professional, formal, empathetic'"
            )

        prompt_parts = [
            f"OUTPUT TO EVALUATE:\n{output}\n",
            f"DESIRED TONE:\n{criteria}\n"
        ]

        if reference:
            prompt_parts.append(
                f"REFERENCE EXAMPLE (showing correct tone):\n{reference}\n"
            )

        prompt_parts.append(
            "Evaluate how well the output matches the desired tone. "
            "Provide detailed analysis and actionable feedback."
        )

        return "\n".join(prompt_parts)

    def _get_response_type(self) -> Type[BaseModel]:
        return ToneResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        tone_response = cast(ToneResponse, response)

        return Score(
            name=self.name,
            value=tone_response.score,
            confidence=tone_response.confidence,
            explanation=tone_response.explanation,
            metadata={
                "detected_tone": tone_response.detected_tone,
                "tone_issues": tone_response.tone_issues,
                "issue_count": len(tone_response.tone_issues)
            }
        )
```

### Usage Example

```python
from arbiter_ai import evaluate

# Simple evaluation
result = await evaluate(
    output="Hey! This is totally awesome!!!",
    criteria="Professional business tone, formal language",
    evaluators=["tone"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Detected tone: {result.scores[0].metadata['detected_tone']}")
print(f"Issues: {result.scores[0].metadata['tone_issues']}")

# Cost tracking
print(f"Cost: ${await result.total_llm_cost():.6f}")
```

## Best Practices

### 1. Clear System Prompts

Make your system prompt specific about what to evaluate:

```python
# GOOD: Specific guidance
"You are an expert at evaluating medical accuracy in patient communications.
Consider: medical correctness, HIPAA compliance, patient-appropriate language."

# BAD: Vague guidance
"You evaluate medical stuff."
```

### 2. Structured Response Models

Use descriptive Pydantic models with field constraints:

```python
# GOOD: Clear constraints and descriptions
class MedicalResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Medical accuracy score")
    hipaa_compliant: bool = Field(..., description="Whether HIPAA compliant")
    medical_errors: list[str] = Field(default_factory=list)

# BAD: No constraints or descriptions
class MedicalResponse(BaseModel):
    score: float
    compliant: bool
    errors: list
```

### 3. Comprehensive Metadata

Include useful metadata in Score objects:

```python
# GOOD: Rich metadata for debugging
return Score(
    name=self.name,
    value=response.score,
    confidence=response.confidence,
    explanation=response.explanation,
    metadata={
        "criteria_met": response.criteria_met,
        "criteria_not_met": response.criteria_not_met,
        "risk_level": response.risk_level,
        "specific_issues": response.issues
    }
)

# BAD: Minimal metadata
return Score(name=self.name, value=response.score)
```

### 4. Validation in User Prompt

Validate required parameters early:

```python
def _get_user_prompt(self, output: str, reference: Optional[str], criteria: Optional[str]) -> str:
    # GOOD: Validate requirements
    if not criteria:
        raise ValueError(
            "MedicalEvaluator requires criteria. "
            "Example: criteria='Medical accuracy, HIPAA compliance'"
        )

    # Build prompt...
```

### 5. Helpful Error Messages

Provide actionable error messages:

```python
# GOOD: Shows how to fix
raise ValueError(
    "ToneEvaluator requires criteria specifying desired tone. "
    "Example: criteria='Professional, formal, empathetic'"
)

# BAD: Unhelpful
raise ValueError("Missing criteria")
```

## Testing Your Evaluator

### Unit Test Template

```python
import pytest
from arbiter_ai.core.llm_client import LLMManager
from your_module import ToneEvaluator

@pytest.mark.asyncio
async def test_tone_evaluator_basic():
    """Test basic tone evaluation."""
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = ToneEvaluator(llm_client=client)

    score = await evaluator.evaluate(
        output="Hey! This is awesome!!!",
        criteria="Professional business tone"
    )

    assert 0.0 <= score.value <= 1.0
    assert score.name == "tone"
    assert score.confidence > 0.0
    assert "detected_tone" in score.metadata


@pytest.mark.asyncio
async def test_tone_evaluator_requires_criteria():
    """Test that criteria is required."""
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = ToneEvaluator(llm_client=client)

    with pytest.raises(ValueError, match="requires criteria"):
        await evaluator.evaluate(output="Some text")


@pytest.mark.asyncio
async def test_tone_evaluator_with_reference():
    """Test evaluation with reference text."""
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = ToneEvaluator(llm_client=client)

    score = await evaluator.evaluate(
        output="Hey! Check this out!",
        reference="We are pleased to announce...",
        criteria="Professional business tone"
    )

    assert score.value < 0.8  # Casual output should score low
```

### Mock Testing (for CI/CD)

```python
@pytest.mark.asyncio
async def test_tone_evaluator_mocked(mocker):
    """Test evaluator with mocked LLM calls."""
    # Mock the LLM response
    mock_response = ToneResponse(
        score=0.3,
        confidence=0.9,
        explanation="Output is too casual",
        detected_tone="casual, enthusiastic",
        tone_issues=["Excessive exclamation marks", "Informal greeting"]
    )

    mocker.patch(
        "arbiter_ai.core.llm_client.LLMClient.create_agent"
    ).return_value.run.return_value.output = mock_response

    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = ToneEvaluator(llm_client=client)

    score = await evaluator.evaluate(
        output="Hey! This is awesome!!!",
        criteria="Professional business tone"
    )

    assert score.value == 0.3
    assert len(score.metadata["tone_issues"]) == 2
```

## Advanced Patterns

### Multi-Criteria Evaluation

Add a method for evaluating multiple aspects:

```python
async def evaluate_multi(
    self,
    output: str,
    criteria: dict[str, str]
) -> list[Score]:
    """Evaluate multiple tone aspects separately.

    Example:
        >>> scores = await evaluator.evaluate_multi(
        ...     output="Product description",
        ...     criteria={
        ...         "professionalism": "Business-appropriate language",
        ...         "enthusiasm": "Engaging and positive",
        ...         "clarity": "Clear and concise"
        ...     }
        ... )
    """
    # Implementation similar to CustomCriteriaEvaluator.evaluate_multi()
```

### Backend Support

Add alternative evaluation backends (like SemanticEvaluator does with FAISS):

```python
def __init__(
    self,
    llm_client: LLMClient,
    backend: Literal["llm", "rule_based"] = "llm"
):
    super().__init__(llm_client)
    self.backend = backend

    if backend == "rule_based":
        self._rule_engine = ToneRuleEngine()

async def evaluate(self, output: str, ...) -> Score:
    if self.backend == "rule_based":
        # Use rule-based evaluation (fast, deterministic)
        return await self._rule_engine.evaluate(output, criteria)

    # Use LLM evaluation (rich explanations)
    return await super().evaluate(output, reference, criteria)
```

### Caching for Identical Evaluations

```python
from functools import lru_cache
import hashlib

def _cache_key(self, output: str, criteria: str) -> str:
    """Generate cache key for evaluation."""
    content = f"{output}::{criteria}::{self.name}"
    return hashlib.sha256(content.encode()).hexdigest()

async def evaluate(self, output: str, ...) -> Score:
    # Check cache first
    cache_key = self._cache_key(output, criteria or "")
    if cache_key in self._cache:
        return self._cache[cache_key]

    # Evaluate and cache
    score = await super().evaluate(output, reference, criteria)
    self._cache[cache_key] = score
    return score
```

## Exporting Your Evaluator

### Step 1: Add to `arbiter_ai/evaluators/__init__.py`

```python
from .tone import ToneEvaluator

__all__ = [
    # ... existing evaluators
    "ToneEvaluator",
]
```

### Step 2: Add to `arbiter_ai/__init__.py`

```python
from .evaluators import ToneEvaluator

__all__ = [
    # ... existing exports
    "ToneEvaluator",
]
```

### Step 3: Register for String-Based Access (Optional)

To use `evaluators=["tone"]` in `evaluate()`, register it:

```python
# In arbiter_ai/core/registry.py or similar
from arbiter_ai.evaluators import ToneEvaluator

EVALUATOR_REGISTRY = {
    "semantic": SemanticEvaluator,
    "custom_criteria": CustomCriteriaEvaluator,
    "tone": ToneEvaluator,  # Add here
}
```

## Common Pitfalls

1. **Not Validating Criteria**: Always check required parameters in `_get_user_prompt()`
2. **Vague System Prompts**: Be specific about evaluation criteria and approach
3. **Missing Field Constraints**: Use Pydantic Field with `ge`, `le`, descriptions
4. **Forgetting Tests**: Always write unit tests with >80% coverage
5. **No Examples**: Add usage examples in docstrings
6. **Missing Metadata**: Include useful metadata in Score objects for debugging

## Next Steps

1. Create your evaluator following this guide
2. Write comprehensive unit tests (>80% coverage)
3. Create an example file in `examples/`
4. Update exports in `__init__.py` files
5. Test with `make test` before committing

## Questions?

- Check existing evaluators in `arbiter_ai/evaluators/` for reference
- See `arbiter_ai/evaluators/base.py` for BasePydanticEvaluator details
- Review `examples/` for usage patterns
