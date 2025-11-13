"""Core data models for Arbiter evaluation framework.

This module defines the primary data structures used throughout Arbiter:
- EvaluationResult: Complete result of an evaluation
- Score: Individual metric score
- Metric: Metadata about a computed metric
- LLMInteraction: Track individual LLM API calls for transparency
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

__all__ = [
    "Score",
    "Metric",
    "LLMInteraction",
    "EvaluationResult",
    "ComparisonResult",
]


class Score(BaseModel):
    """Individual evaluation score for a specific metric.

    Represents a single numeric score with metadata about how it was
    computed and what it represents.

    Example:
        >>> score = Score(
        ...     name="semantic_similarity",
        ...     value=0.92,
        ...     confidence=0.95,
        ...     explanation="High semantic overlap between output and reference"
        ... )
    """

    name: str = Field(..., description="Name of the metric (e.g., 'factuality')")
    value: float = Field(..., ge=0.0, le=1.0, description="Score value between 0 and 1")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in this score"
    )
    explanation: Optional[str] = Field(None, description="Human-readable explanation of the score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the score"
    )


class LLMInteraction(BaseModel):
    """Record of a single LLM API call during evaluation.

    Tracks all LLM interactions for complete observability and debugging.
    Similar to Sifaka's Generation tracking but focused on evaluation context.

    This provides:
    - Complete audit trail of LLM usage
    - Token and cost tracking
    - Debugging capabilities
    - Transparency in how evaluations were computed

    Example:
        >>> interaction = LLMInteraction(
        ...     prompt="Evaluate the factuality of this statement...",
        ...     response="Score: 0.85. The statement is mostly accurate...",
        ...     model="gpt-4o",
        ...     tokens_used=150,
        ...     latency=1.2,
        ...     purpose="factuality_scoring"
        ... )
    """

    prompt: str = Field(..., description="The prompt sent to the LLM")
    response: str = Field(..., description="The LLM's response")
    model: str = Field(..., description="Model used for this call")
    tokens_used: int = Field(default=0, description="Tokens consumed in this call")
    latency: float = Field(..., description="Time taken for this call (seconds)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this call was made"
    )
    purpose: str = Field(
        ...,
        description="Purpose of this call (e.g., 'scoring', 'semantic_comparison', 'factuality_check')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context about this call"
    )


class Metric(BaseModel):
    """Metadata about a computed metric.

    Provides information about how a metric was computed, including
    the model used, processing time, and any relevant context.
    """

    name: str = Field(..., description="Name of the metric")
    evaluator: str = Field(..., description="Name of the evaluator that computed it")
    model: Optional[str] = Field(None, description="LLM model used (if applicable)")
    processing_time: float = Field(..., description="Time taken to compute (seconds)")
    tokens_used: int = Field(default=0, description="Tokens consumed (if applicable)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvaluationResult(BaseModel):
    """Complete result of an evaluation operation.

    Contains all scores, metadata, and audit trail information from
    evaluating an LLM output against reference or criteria.

    This is the primary result object returned by all evaluation
    operations in Arbiter.

    Example:
        >>> result = EvaluationResult(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     scores=[
        ...         Score(name="semantic_similarity", value=0.95),
        ...         Score(name="factuality", value=1.0),
        ...     ],
        ...     overall_score=0.975,
        ...     passed=True,
        ...     partial=False,
        ...     errors={}
        ... )
        >>>
        >>> # Partial result example (some evaluators failed)
        >>> result = EvaluationResult(
        ...     output="text",
        ...     scores=[Score(name="semantic", value=0.8)],
        ...     overall_score=0.8,
        ...     passed=True,
        ...     partial=True,
        ...     errors={"factuality": "API timeout"}
        ... )
    """

    # Input data
    output: str = Field(..., description="The LLM output that was evaluated")
    reference: Optional[str] = Field(
        None, description="Reference text used for comparison (if applicable)"
    )
    criteria: Optional[str] = Field(
        None, description="Evaluation criteria used (if reference-free)"
    )

    # Results
    scores: List[Score] = Field(
        default_factory=list,
        description="Individual metric scores from successful evaluators only. Failed evaluators are excluded.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate score calculated as the average of successful evaluator scores only. "
        "Failed evaluators are excluded from the calculation. If partial=True, this represents "
        "the average of only the successful evaluations, not all requested evaluators.",
    )
    passed: bool = Field(..., description="Whether evaluation passed quality threshold")

    # Error handling
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Errors encountered during evaluation, keyed by evaluator name",
    )
    partial: bool = Field(
        default=False,
        description="Whether this is a partial result (some evaluators failed)",
    )

    # Metadata
    metrics: List[Metric] = Field(
        default_factory=list, description="Metadata about computed metrics"
    )
    evaluator_names: List[str] = Field(default_factory=list, description="Names of evaluators used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When evaluation completed"
    )

    # LLM interaction tracking (like Sifaka's generations)
    interactions: List[LLMInteraction] = Field(
        default_factory=list,
        description="All LLM API calls made during evaluation for full transparency",
    )

    # Audit trail
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and context"
    )

    def get_score(self, name: str) -> Optional[Score]:
        """Get score by metric name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Score object if found, None otherwise
        """
        for score in self.scores:
            if score.name == name:
                return score
        return None

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric metadata by name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Metric object if found, None otherwise
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def get_interactions_by_purpose(self, purpose: str) -> List[LLMInteraction]:
        """Get all LLM interactions for a specific purpose.

        Args:
            purpose: Purpose to filter by (e.g., 'scoring', 'semantic_comparison')

        Returns:
            List of interactions matching the purpose

        Example:
            >>> # Get all scoring-related LLM calls
            >>> scoring_calls = result.get_interactions_by_purpose("scoring")
            >>> total_tokens = sum(i.tokens_used for i in scoring_calls)
        """
        return [i for i in self.interactions if i.purpose == purpose]

    def total_llm_cost(self, cost_per_1k_tokens: float = 0.01) -> float:
        """Estimate total LLM cost based on token usage.

        Args:
            cost_per_1k_tokens: Cost per 1000 tokens (default: $0.01)

        Returns:
            Estimated cost in dollars

        Example:
            >>> cost = result.total_llm_cost(cost_per_1k_tokens=0.03)
            >>> print(f"Evaluation cost: ${cost:.4f}")
        """
        total_tokens = sum(i.tokens_used for i in self.interactions)
        return (total_tokens / 1000) * cost_per_1k_tokens


class ComparisonResult(BaseModel):
    """Result of comparing two LLM outputs.

    Used for pairwise comparison evaluation where two outputs are compared
    against each other to determine which is better, or if they're equivalent.

    This is different from EvaluationResult which evaluates a single output
    against reference or criteria. ComparisonResult compares two outputs
    directly against each other.

    Example:
        >>> comparison = ComparisonResult(
        ...     output_a="GPT-4 response",
        ...     output_b="Claude response",
        ...     winner="output_a",
        ...     confidence=0.85,
        ...     reasoning="Output A is more accurate and complete",
        ...     aspect_scores={
        ...         "accuracy": {"output_a": 0.9, "output_b": 0.8},
        ...         "clarity": {"output_a": 0.85, "output_b": 0.9},
        ...     }
        ... )
        >>> print(f"Winner: {comparison.winner}")
    """

    # Input data
    output_a: str = Field(..., description="First output being compared")
    output_b: str = Field(..., description="Second output being compared")
    reference: Optional[str] = Field(
        None, description="Optional reference context (e.g., user question)"
    )
    criteria: Optional[str] = Field(None, description="Optional criteria used for comparison")

    # Comparison results
    winner: Literal["output_a", "output_b", "tie"] = Field(
        ...,
        description="Which output is better, or 'tie' if equivalent",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the comparison decision",
    )
    reasoning: str = Field(..., description="Detailed explanation of why this winner was chosen")
    aspect_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Scores for each aspect/criterion, mapping aspect name to scores for output_a and output_b",
    )

    # Metadata
    total_tokens: int = Field(default=0, description="Total tokens used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When comparison completed"
    )

    # LLM interaction tracking
    interactions: List[LLMInteraction] = Field(
        default_factory=list,
        description="All LLM API calls made during comparison for full transparency",
    )

    # Audit trail
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and context"
    )

    def get_aspect_score(
        self, aspect: str, output: Literal["output_a", "output_b"]
    ) -> Optional[float]:
        """Get score for a specific aspect and output.

        Args:
            aspect: Name of the aspect (e.g., 'accuracy', 'clarity')
            output: Which output to get score for ('output_a' or 'output_b')

        Returns:
            Score value if found, None otherwise

        Example:
            >>> accuracy_a = comparison.get_aspect_score("accuracy", "output_a")
            >>> print(f"Output A accuracy: {accuracy_a}")
        """
        if aspect in self.aspect_scores:
            return self.aspect_scores[aspect].get(output)
        return None

    def total_llm_cost(self, cost_per_1k_tokens: float = 0.01) -> float:
        """Estimate total LLM cost based on token usage.

        Args:
            cost_per_1k_tokens: Cost per 1000 tokens (default: $0.01)

        Returns:
            Estimated cost in dollars
        """
        total_tokens = sum(i.tokens_used for i in self.interactions)
        return (total_tokens / 1000) * cost_per_1k_tokens
