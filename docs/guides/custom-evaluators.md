# Building Custom Evaluators

Complete guide to creating your own evaluators using Arbiter's template method pattern.

## Overview

Arbiter makes it easy to create custom evaluators for domain-specific evaluation needs. All evaluators extend `BasePydanticEvaluator` and implement 4 methods.

**Benefits:**
- Automatic LLM interaction tracking
- Structured outputs with type safety
- Consistent error handling
- Provider-agnostic design
- Easy to test and maintain

## Quick Example

```python
from typing import Type, Optional, cast
from pydantic import BaseModel, Field
from arbiter.evaluators.base import BasePydanticEvaluator
from arbiter.core.models import Score

# 1. Define your response model
class SentimentResponse(BaseModel):
    """Structured response for sentiment analysis."""
    score: float = Field(ge=0.0, le=1.0, description="Sentiment score (0=negative, 1=positive)")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    sentiment: str = Field(description="positive, negative, or neutral")
    explanation: str

# 2. Implement your evaluator
class SentimentEvaluator(BasePydanticEvaluator):
    """Evaluates sentiment of text."""

    @property
    def name(self) -> str:
        return "sentiment"

    def _get_system_prompt(self) -> str:
        return """You are an expert sentiment analyst.
Analyze the sentiment of the given text and provide:
1. A score from 0 (very negative) to 1 (very positive)
2. The overall sentiment (positive, negative, neutral)
3. Your confidence level
4. A brief explanation"""

    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None
    ) -> str:
        return f"Analyze the sentiment of this text:\n\n{output}"

    def _get_response_type(self) -> Type[BaseModel]:
        return SentimentResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(SentimentResponse, response)
        return Score(
            name=self.name,
            value=resp.score,
            confidence=resp.confidence,
            explanation=resp.explanation,
            metadata={"sentiment": resp.sentiment}
        )
```

## Using Your Custom Evaluator

### Option 1: Register and Use with evaluate()

```python
from arbiter import evaluate, register_evaluator

# Register your evaluator
register_evaluator("sentiment", SentimentEvaluator)

# Use it like any built-in evaluator
result = await evaluate(
    output="I absolutely love this product! It exceeded all my expectations.",
    evaluators=["sentiment"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Sentiment: {result.scores[0].metadata['sentiment']}")
```

### Option 2: Use Directly

```python
from arbiter.core.llm_client import LLMManager

# Get LLM client
client = await LLMManager.get_client(model="gpt-4o-mini")

# Create and use evaluator directly
evaluator = SentimentEvaluator(client)
score = await evaluator.evaluate(
    output="I absolutely love this product!"
)

print(f"Sentiment: {score.metadata['sentiment']}")
print(f"Score: {score.value:.2f}")
```

## The Four Required Methods

### 1. `_get_system_prompt()` - Expert Role

Defines the LLM's role and instructions.

**Best Practices:**
- Be specific about the role (e.g., "expert sentiment analyst")
- List what to provide (numbered is good)
- Keep concise but clear

**Example:**
```python
def _get_system_prompt(self) -> str:
    return """You are an expert code quality analyzer.
Evaluate code quality based on:
1. Readability and clarity
2. Maintainability and structure
3. Best practices adherence
4. Performance considerations"""
```

### 2. `_get_user_prompt()` - Task Specification

Formats the actual evaluation task.

**Parameters:**
- `output`: The text to evaluate (required)
- `reference`: Optional reference text for comparison
- `criteria`: Optional evaluation criteria

**Best Practices:**
- Format clearly with sections
- Include context if needed
- Use the parameters appropriately

**Example:**
```python
def _get_user_prompt(
    self,
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None
) -> str:
    prompt = f"Evaluate this code:\n\n```\n{output}\n```\n\n"

    if criteria:
        prompt += f"Focus on these aspects: {criteria}\n\n"

    if reference:
        prompt += f"Compare against this reference:\n```\n{reference}\n```"

    return prompt
```

### 3. `_get_response_type()` - Structure Definition

Returns the Pydantic model for structured responses.

**Best Practices:**
- Use clear field names
- Add descriptions for clarity
- Use Field() for validation
- Include confidence field (0-1)

**Example:**
```python
class CodeQualityResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0)
    maintainability: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    explanation: str

def _get_response_type(self) -> Type[BaseModel]:
    return CodeQualityResponse
```

### 4. `_compute_score()` - Extract Score

Converts the LLM response to an Arbiter Score.

**Best Practices:**
- Use `cast()` for type safety
- Extract score value
- Include confidence
- Add metadata for context

**Example:**
```python
async def _compute_score(self, response: BaseModel) -> Score:
    resp = cast(CodeQualityResponse, response)
    return Score(
        name=self.name,
        value=resp.score,
        confidence=resp.confidence,
        explanation=resp.explanation,
        metadata={
            "readability": resp.readability,
            "maintainability": resp.maintainability,
            "issues": resp.issues,
            "strengths": resp.strengths
        }
    )
```

## Complete Example: Code Quality Evaluator

```python
from typing import Type, Optional, cast
from pydantic import BaseModel, Field
from arbiter.evaluators.base import BasePydanticEvaluator
from arbiter.core.models import Score

class CodeQualityResponse(BaseModel):
    """Structured response for code quality evaluation."""
    score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0, description="How readable is the code")
    maintainability: float = Field(ge=0.0, le=1.0, description="How maintainable")
    issues: list[str] = Field(default_factory=list, description="Problems found")
    strengths: list[str] = Field(default_factory=list, description="Good practices")
    explanation: str = Field(description="Overall assessment")

class CodeQualityEvaluator(BasePydanticEvaluator):
    """Evaluates code quality across multiple dimensions."""

    @property
    def name(self) -> str:
        return "code_quality"

    def _get_system_prompt(self) -> str:
        return """You are an expert software engineer and code reviewer.
Evaluate code quality based on:
1. Readability (clear naming, structure, comments)
2. Maintainability (modularity, coupling, cohesion)
3. Best practices (error handling, type safety, patterns)
4. Performance (efficiency, resource usage)

Provide a score from 0-1 for overall quality and specific aspects."""

    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None
    ) -> str:
        prompt = f"Evaluate this code:\n\n```python\n{output}\n```\n\n"

        if criteria:
            prompt += f"Pay special attention to: {criteria}\n\n"

        if reference:
            prompt += f"Compare against this reference implementation:\n```python\n{reference}\n```\n\n"

        prompt += "Provide specific issues found, strengths, and an overall assessment."
        return prompt

    def _get_response_type(self) -> Type[BaseModel]:
        return CodeQualityResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(CodeQualityResponse, response)
        return Score(
            name=self.name,
            value=resp.score,
            confidence=resp.confidence,
            explanation=resp.explanation,
            metadata={
                "readability": resp.readability,
                "maintainability": resp.maintainability,
                "issues": resp.issues,
                "strengths": resp.strengths
            }
        )
```

### Using the Code Quality Evaluator

```python
from arbiter import register_evaluator, evaluate

# Register the custom evaluator
register_evaluator("code_quality", CodeQualityEvaluator)

# Evaluate code
code = '''
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item["price"] * item["quantity"]
    return total
'''

result = await evaluate(
    output=code,
    criteria="readability, error handling, type safety",
    evaluators=["code_quality"],
    model="gpt-4o-mini"
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Readability: {result.scores[0].metadata['readability']:.2f}")
print(f"Issues found: {len(result.scores[0].metadata['issues'])}")
for issue in result.scores[0].metadata['issues']:
    print(f"  - {issue}")
```

## Testing Your Evaluator

Always write tests for custom evaluators:

```python
import pytest
from arbiter.core.llm_client import LLMManager

@pytest.mark.asyncio
async def test_code_quality_evaluator():
    """Test code quality evaluator."""
    # Create evaluator
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = CodeQualityEvaluator(client)

    # Good code should score high
    good_code = '''
def calculate_total(items: list[dict]) -> float:
    """Calculate total price for items."""
    return sum(item["price"] * item["quantity"] for item in items)
'''

    score = await evaluator.evaluate(output=good_code)

    assert 0.0 <= score.value <= 1.0
    assert score.confidence > 0.0
    assert "readability" in score.metadata
    assert "maintainability" in score.metadata

@pytest.mark.asyncio
async def test_code_quality_with_issues():
    """Test code quality catches issues."""
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = CodeQualityEvaluator(client)

    # Code with issues
    bad_code = '''
def calc(x):
    t = 0
    for i in x:
        t = t + i["p"] * i["q"]  # What if keys don't exist?
    return t
'''

    score = await evaluator.evaluate(output=bad_code)

    # Should find issues
    assert len(score.metadata["issues"]) > 0
    # Score should be lower
    assert score.value < 0.8
```

## Advanced Patterns

### Multi-Criteria Evaluation

```python
class MultiCriteriaResponse(BaseModel):
    """Response with multiple scores."""
    scores: dict[str, float] = Field(description="Score per criterion")
    confidence: float = Field(default=0.85)
    explanations: dict[str, str] = Field(description="Explanation per criterion")
    overall_explanation: str

async def _compute_score(self, response: BaseModel) -> Score:
    """Can return a single aggregate or multiple scores."""
    resp = cast(MultiCriteriaResponse, response)

    # Option 1: Return average
    avg_score = sum(resp.scores.values()) / len(resp.scores)

    # Option 2: Return multiple Score objects (advanced)
    # Would need to modify evaluate() API to handle this

    return Score(
        name=self.name,
        value=avg_score,
        confidence=resp.confidence,
        explanation=resp.overall_explanation,
        metadata={
            "scores": resp.scores,
            "explanations": resp.explanations
        }
    )
```

### Using External APIs

```python
import httpx

class HybridToxicityEvaluator(BasePydanticEvaluator):
    """Combines Perspective API with LLM judgment."""

    def __init__(self, client, perspective_key: str):
        super().__init__(client)
        self.perspective_key = perspective_key

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None,
        threshold: float = 0.5
    ) -> Score:
        # Call Perspective API first
        async with httpx.AsyncClient() as http_client:
            perspective_score = await self._call_perspective_api(
                http_client, output
            )

        # Then use LLM for nuanced analysis
        llm_score = await super().evaluate(output, reference, criteria, threshold)

        # Combine scores
        final_score = (perspective_score + llm_score.value) / 2

        return Score(
            name=self.name,
            value=final_score,
            confidence=llm_score.confidence,
            explanation=llm_score.explanation,
            metadata={
                "perspective_score": perspective_score,
                "llm_score": llm_score.value,
                **llm_score.metadata
            }
        )

    async def _call_perspective_api(
        self, client: httpx.AsyncClient, text: str
    ) -> float:
        """Call Google Perspective API for toxicity score."""
        # Implementation details...
        pass
```

## Best Practices

### 1. Clear Naming
```python
# Good
class FactualAccuracyEvaluator
class MedicalComplianceEvaluator
class BrandVoiceEvaluator

# Bad (too vague)
class MyEvaluator
class Evaluator1
class CustomEval
```

### 2. Descriptive Prompts
```python
# Good - specific instructions
"""You are an expert medical content reviewer.
Evaluate if the text:
1. Is medically accurate
2. Complies with HIPAA guidelines
3. Uses appropriate terminology
4. Has correct dosage information"""

# Bad - too vague
"""Check if this is good medical content."""
```

### 3. Structured Responses
```python
# Good - clear structure
class Response(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    medical_accuracy: float
    hipaa_compliant: bool
    issues_found: list[str]
    explanation: str

# Bad - unstructured
class Response(BaseModel):
    result: str  # LLM will return free text
```

### 4. Meaningful Metadata
```python
# Good - useful debugging info
metadata={
    "claims_verified": 5,
    "claims_false": 0,
    "claims_uncertain": 1,
    "sources_cited": ["pubmed:12345"],
    "model_version": "gpt-4o-mini"
}

# Bad - not helpful
metadata={"done": True}
```

### 5. Error Handling
```python
async def _compute_score(self, response: BaseModel) -> Score:
    try:
        resp = cast(MyResponse, response)

        # Validate score is in range
        score_value = max(0.0, min(1.0, resp.score))

        return Score(
            name=self.name,
            value=score_value,
            confidence=resp.confidence,
            explanation=resp.explanation
        )
    except (AttributeError, ValueError) as e:
        # Log error and return low-confidence score
        logger.error(f"Error computing score: {e}")
        return Score(
            name=self.name,
            value=0.0,
            confidence=0.0,
            explanation=f"Error: {str(e)}"
        )
```

## Common Patterns

### Reference-Based Evaluation
```python
def _get_user_prompt(self, output: str, reference: Optional[str] = None, criteria: Optional[str] = None) -> str:
    if not reference:
        raise ValueError("Reference is required for this evaluator")

    return f"""Compare the output to the reference:

Reference: {reference}

Output: {output}

Evaluate how well the output matches the reference."""
```

### Criteria-Based Evaluation
```python
def _get_user_prompt(self, output: str, reference: Optional[str] = None, criteria: Optional[str] = None) -> str:
    criteria_list = criteria or "accuracy, clarity, completeness"

    return f"""Evaluate this output against these criteria:

Criteria: {criteria_list}

Output: {output}"""
```

### Standalone Evaluation
```python
def _get_user_prompt(self, output: str, reference: Optional[str] = None, criteria: Optional[str] = None) -> str:
    return f"""Evaluate this output independently:

{output}

Assess overall quality, coherence, and usefulness."""
```

## Next Steps

- Check out [examples/evaluator_registry_example.py](../../examples/evaluator_registry_example.py)
- Read the [API documentation](../api/evaluators.md)
- See built-in evaluators for reference:
  - [SemanticEvaluator](../../arbiter/evaluators/semantic.py)
  - [CustomCriteriaEvaluator](../../arbiter/evaluators/custom_criteria.py)
  - [PairwiseComparisonEvaluator](../../arbiter/evaluators/pairwise.py)

## Getting Help

- Open an [issue](https://github.com/evanvolgas/arbiter/issues) for bugs
- Start a [discussion](https://github.com/evanvolgas/arbiter/discussions) for questions
- Check the [troubleshooting guide](troubleshooting.md)
