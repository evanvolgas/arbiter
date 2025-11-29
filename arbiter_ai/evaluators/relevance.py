"""Relevance evaluator for query-output alignment assessment.

This evaluator assesses whether LLM outputs are relevant to the given query
or context by analyzing if the output addresses the query requirements,
identifying missing points, and detecting off-topic content.

## When to Use:

- Ensuring LLM outputs stay on-topic
- Validating query-output alignment
- Detecting irrelevant or off-topic content
- Assessing completeness of responses
- Quality control for question-answering systems

## Example:

    >>> evaluator = RelevanceEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output="Python was created by Guido van Rossum in 1991",
    ...     reference="What programming language was created by Guido van Rossum and when?"
    ... )
    >>> print(f"Relevance score: {score.value:.2f}")
    >>> print(f"Missing points: {score.metadata.get('missing_points', [])}")
    Relevance score: 1.0
    Missing points: []
"""

from typing import List, Optional, Type, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.models import Score
from .base import BasePydanticEvaluator

__all__ = ["RelevanceEvaluator", "RelevanceResponse"]


class RelevanceResponse(BaseModel):
    """Structured response for relevance evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0=completely irrelevant, 1=fully relevant and complete)",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this relevance assessment",
    )
    explanation: str = Field(
        ..., description="Explanation of the relevance assessment and reasoning"
    )
    addressed_points: List[str] = Field(
        default_factory=list,
        description="Query aspects that were addressed in the output",
    )
    missing_points: List[str] = Field(
        default_factory=list,
        description="Query aspects that were not addressed or incompletely covered",
    )
    irrelevant_content: List[str] = Field(
        default_factory=list,
        description="Content in the output that is off-topic or not relevant to the query",
    )

    @field_validator("explanation")
    @classmethod
    def validate_explanation_quality(cls, v: str) -> str:
        """Ensure explanation is meaningful and not empty."""
        if len(v.strip()) < 1:
            raise ValueError("Explanation cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_relevance_analysis(self) -> "RelevanceResponse":
        """Ensure points are identified for non-trivial relevance assessments."""
        total_points = (
            len(self.addressed_points)
            + len(self.missing_points)
            + len(self.irrelevant_content)
        )
        if self.confidence > 0.7 and total_points == 0 and self.score not in (0.0, 1.0):
            raise ValueError(
                "High confidence scores (>0.9) require at least one point to be identified"
            )
        return self


class RelevanceEvaluator(BasePydanticEvaluator):
    """Evaluates query-output relevance and alignment.

    This evaluator analyzes whether LLM outputs are relevant to the given
    query or context by identifying addressed points, missing information,
    and irrelevant content. It's critical for ensuring outputs stay on-topic
    and actually answer the question asked.

    The evaluator:
    - Analyzes the query to identify what should be addressed
    - Checks if the output addresses each query aspect
    - Identifies missing points that weren't covered
    - Detects irrelevant or off-topic content
    - Provides a score based on coverage and relevance

    Modes:
    - With reference (query): Assess relevance to the query/question
    - With criteria: Focus on specific relevance aspects

    Example:
        >>> # Create evaluator
        >>> from arbiter import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = RelevanceEvaluator(client)
        >>>
        >>> # Assess query-output relevance
        >>> score = await evaluator.evaluate(
        ...     output="The Eiffel Tower is 300 meters tall and located in Paris",
        ...     reference="How tall is the Eiffel Tower?"
        ... )
        >>>
        >>> print(f"Relevance: {score.value:.2f}")
        >>> print(f"Missing points: {score.metadata['missing_points']}")
        >>> print(f"Irrelevant content: {score.metadata['irrelevant_content']}")
        >>>
        >>> # Check LLM interactions
        >>> interactions = evaluator.get_interactions()
        >>> print(f"Made {len(interactions)} LLM calls")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "relevance"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining relevance evaluation approach."""
        return """You are an expert in assessing query-output relevance and alignment.

Your task is to evaluate whether an output is relevant to a given query by:

1. ANALYZING THE QUERY: Identify what the query is asking for
   - Main question or topic
   - Specific aspects or sub-questions
   - Expected information to be covered

2. EVALUATING THE OUTPUT: Check if it addresses the query
   - Does it answer the main question?
   - Does it cover all required aspects?
   - Is the information on-topic?

3. CATEGORIZING CONTENT:
   - Addressed points: Query aspects that were covered in the output
   - Missing points: Query aspects that were not addressed or incompletely covered
   - Irrelevant content: Information in the output that is off-topic or not needed

4. SCORING: Calculate relevance score based on:
   score = (addressed_points / (addressed_points + missing_points)) * (1 - irrelevance_penalty)
   - All query aspects addressed, no irrelevant content = 1.0
   - Some aspects missing = proportional score
   - Significant irrelevant content = score reduction
   - Completely off-topic = 0.0

Important Guidelines:
- Focus on query alignment: Does the output answer what was asked?
- Completeness matters: Missing key points should reduce the score
- Penalize irrelevance: Off-topic content dilutes the response quality
- Context awareness: Consider what a reasonable answer should include
- Partial credit: Give credit for partially addressing points
- Direct vs indirect: Both can be relevant if they answer the query

Provide:
- A relevance score from 0.0 (completely irrelevant) to 1.0 (fully relevant and complete)
- Your confidence in this assessment
- Clear lists of addressed points, missing points, and irrelevant content
- Detailed explanation of your reasoning"""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if not reference and not criteria:
            return """Error: Relevance evaluation requires a query or criteria.

Please provide:
- A query/question in the 'reference' parameter to evaluate relevance, OR
- Specific relevance criteria in the 'criteria' parameter

Example:
    reference="What is the capital of France?"
    criteria="Focus on answering the core question directly"
"""

        if reference and criteria:
            # Evaluate relevance to query with specific criteria
            return f"""Evaluate the relevance of this OUTPUT to the QUERY, focusing on: {criteria}

QUERY (what should be addressed):
{reference}

OUTPUT (to evaluate):
{output}

Your task:
1. Analyze the QUERY to understand what should be addressed
2. Evaluate if the OUTPUT addresses the query requirements
3. Identify:
   - Addressed points: What query aspects were covered
   - Missing points: What query aspects were not covered or incomplete
   - Irrelevant content: Any off-topic information

4. Pay special attention to the criteria: {criteria}
5. Calculate relevance score based on query coverage and focus

Provide your relevance assessment with clear categorization."""

        elif reference:
            # Standard query-output relevance
            return f"""Evaluate the relevance of this OUTPUT to the QUERY.

QUERY (what should be addressed):
{reference}

OUTPUT (to evaluate):
{output}

Your task:
1. Analyze the QUERY to identify what should be addressed
   - Main question or topic
   - Specific aspects or details requested
   - Expected information

2. Evaluate the OUTPUT for relevance:
   - Does it answer the main query?
   - Does it cover all required aspects?
   - Is the information on-topic?

3. Categorize content:
   - Addressed points: Query aspects that were covered
   - Missing points: Query aspects not addressed or incomplete
   - Irrelevant content: Off-topic information

4. Calculate relevance score based on query coverage and focus

Provide your assessment with clear categorization of points."""

        else:
            # Criteria-only mode
            return f"""Evaluate the relevance of this OUTPUT based on these criteria: {criteria}

OUTPUT (to evaluate):
{output}

Your task:
1. Interpret the relevance criteria
2. Assess if the output meets the specified relevance requirements
3. Identify:
   - Addressed points: What criteria aspects were met
   - Missing points: What criteria aspects were not met
   - Irrelevant content: Content that doesn't align with criteria

4. Calculate relevance score based on criteria alignment

Provide your relevance assessment."""

    def _get_response_type(self) -> Type[BaseModel]:
        """Use relevance response model."""
        return RelevanceResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from relevance response."""
        relevance_response = cast(RelevanceResponse, response)

        # Build detailed explanation
        explanation_parts = [relevance_response.explanation]

        if (
            hasattr(relevance_response, "addressed_points")
            and relevance_response.addressed_points
        ):
            explanation_parts.append(
                "\n\nAddressed Points (Covered):\n- "
                + "\n- ".join(relevance_response.addressed_points)
            )

        if (
            hasattr(relevance_response, "missing_points")
            and relevance_response.missing_points
        ):
            explanation_parts.append(
                "\n\nMissing Points (Not Addressed):\n- "
                + "\n- ".join(relevance_response.missing_points)
            )

        if (
            hasattr(relevance_response, "irrelevant_content")
            and relevance_response.irrelevant_content
        ):
            explanation_parts.append(
                "\n\nIrrelevant Content (Off-Topic):\n- "
                + "\n- ".join(relevance_response.irrelevant_content)
            )

        full_explanation = "".join(explanation_parts)

        return Score(
            name=self.name,
            value=relevance_response.score,
            confidence=relevance_response.confidence,
            explanation=full_explanation,
            metadata={
                "addressed_points": relevance_response.addressed_points,
                "missing_points": relevance_response.missing_points,
                "irrelevant_content": relevance_response.irrelevant_content,
                "addressed_count": len(
                    getattr(relevance_response, "addressed_points", [])
                ),
                "missing_count": len(getattr(relevance_response, "missing_points", [])),
                "irrelevant_count": len(
                    getattr(relevance_response, "irrelevant_content", [])
                ),
                "total_points": len(getattr(relevance_response, "addressed_points", []))
                + len(getattr(relevance_response, "missing_points", [])),
            },
        )
