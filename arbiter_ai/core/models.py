"""Core data models for Arbiter evaluation framework.

This module defines the primary data structures used throughout Arbiter:
- EvaluationResult: Complete result of an evaluation
- Score: Individual metric score
- Metric: Metadata about a computed metric
- LLMInteraction: Track individual LLM API calls for transparency
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

__all__ = [
    "Score",
    "Metric",
    "LLMInteraction",
    "EvaluationResult",
    "ComparisonResult",
    "BatchEvaluationResult",
]


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Replacement for deprecated datetime.utcnow() which returns naive datetime.
    Uses timezone.utc for Python 3.10+ compatibility.
    """
    return datetime.now(timezone.utc)


def _get_interaction_cost(interaction: "LLMInteraction") -> float:
    """Calculate cost for a single LLM interaction.

    Helper function to avoid code duplication across cost calculation methods.
    Uses cached cost if available, otherwise calculates from token counts.

    Args:
        interaction: LLMInteraction with token data and optional cached cost

    Returns:
        Cost in USD for this interaction
    """
    # Use cached cost if available
    if interaction.cost is not None:
        return interaction.cost

    # Calculate on-the-fly using cost calculator
    from arbiter_ai.core.cost_calculator import get_cost_calculator

    calc = get_cost_calculator()
    return calc.calculate_cost(
        model=interaction.model,
        input_tokens=interaction.input_tokens,
        output_tokens=interaction.output_tokens,
        cached_tokens=interaction.cached_tokens,
    )


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

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    name: str = Field(..., description="Name of the metric (e.g., 'factuality')")
    value: float = Field(..., ge=0.0, le=1.0, description="Score value between 0 and 1")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in this score"
    )
    explanation: Optional[str] = Field(
        None, description="Human-readable explanation of the score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the score"
    )


class LLMInteraction(BaseModel):
    """Record of a single LLM API call during evaluation.

    Tracks all LLM interactions for complete observability and debugging.
    Provides automatic transparency into evaluation processes.

    This provides:
    - Complete audit trail of LLM usage
    - Detailed token and cost tracking (input/output/cached)
    - Debugging capabilities
    - Transparency in how evaluations were computed

    Example:
        >>> interaction = LLMInteraction(
        ...     prompt="Evaluate the factuality of this statement...",
        ...     response="Score: 0.85. The statement is mostly accurate...",
        ...     model="gpt-4o",
        ...     input_tokens=120,
        ...     output_tokens=30,
        ...     latency=1.2,
        ...     purpose="factuality_scoring",
        ...     cost=0.000045
        ... )
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    prompt: str = Field(..., description="The prompt sent to the LLM")
    response: str = Field(..., description="The LLM's response")
    model: str = Field(..., description="Model used for this call")

    # Token tracking (new detailed fields)
    input_tokens: int = Field(default=0, ge=0, description="Input tokens consumed")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    cached_tokens: int = Field(
        default=0,
        ge=0,
        description="Cached input tokens (e.g., Anthropic prompt caching)",
    )

    # Backward compatibility
    tokens_used: int = Field(
        default=0, ge=0, description="Total tokens (for backward compatibility)"
    )

    # Cost tracking
    cost: Optional[float] = Field(
        None, ge=0, description="Actual cost in USD (calculated using llm-prices data)"
    )

    latency: float = Field(..., ge=0, description="Time taken for this call (seconds)")
    timestamp: datetime = Field(
        default_factory=_utc_now, description="When this call was made"
    )
    purpose: str = Field(
        ...,
        description="Purpose of this call (e.g., 'scoring', 'semantic_comparison', 'factuality_check')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context about this call"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens across input and output.

        Returns:
            Sum of input_tokens and output_tokens

        Example:
            >>> interaction.total_tokens
            150
        """
        return self.input_tokens + self.output_tokens


class Metric(BaseModel):
    """Metadata about a computed metric.

    Provides information about how a metric was computed, including
    the model used, processing time, and any relevant context.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    name: str = Field(..., description="Name of the metric")
    evaluator: str = Field(..., description="Name of the evaluator that computed it")
    model: Optional[str] = Field(None, description="LLM model used (if applicable)")
    processing_time: float = Field(..., description="Time taken to compute (seconds)")
    tokens_used: int = Field(default=0, description="Tokens consumed (if applicable)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


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

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

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
    evaluator_names: List[str] = Field(
        default_factory=list, description="Names of evaluators used"
    )
    total_tokens: int = Field(default=0, description="Total tokens used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=_utc_now, description="When evaluation completed"
    )

    # LLM interaction tracking for complete transparency
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

    async def total_llm_cost(self, use_actual_pricing: bool = True) -> float:
        """Calculate total LLM cost with accurate pricing.

        Uses llm-prices.com data for accurate cost calculation. Falls back
        to conservative estimates if pricing data unavailable.

        Args:
            use_actual_pricing: If True, use llm-prices data; if False, use simple estimation

        Returns:
            Total cost in USD

        Example:
            >>> result = await evaluate(output, reference)
            >>> cost = await result.total_llm_cost()
            >>> print(f"Evaluation cost: ${cost:.6f}")
        """
        if not use_actual_pricing:
            # Simple fallback: average $0.02 per 1K tokens
            total_tokens = sum(i.total_tokens for i in self.interactions)
            return (total_tokens / 1000) * 0.02

        # Use cost calculator for accurate pricing
        from arbiter_ai.core.cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        await calc.ensure_loaded()

        total = 0.0
        for interaction in self.interactions:
            total += _get_interaction_cost(interaction)

        return total

    async def cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown by evaluator and model.

        Returns:
            Dictionary with cost breakdowns including:
            - total: Total cost in USD
            - by_evaluator: Cost grouped by evaluator purpose
            - by_model: Cost grouped by model
            - token_breakdown: Token usage details

        Example:
            >>> breakdown = await result.cost_breakdown()
            >>> print(breakdown)
            {
                "total": 0.0105,
                "by_evaluator": {
                    "semantic": 0.0045,
                    "factuality": 0.0060
                },
                "by_model": {
                    "gpt-4o-mini": 0.0105
                },
                "token_breakdown": {
                    "input_tokens": 2000,
                    "output_tokens": 500,
                    "cached_tokens": 100
                }
            }
        """
        from arbiter_ai.core.cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        await calc.ensure_loaded()

        by_evaluator: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        total = 0.0
        total_input = 0
        total_output = 0
        total_cached = 0

        for interaction in self.interactions:
            # Calculate cost using helper
            cost = _get_interaction_cost(interaction)

            # Aggregate by purpose (evaluator)
            purpose = interaction.purpose
            by_evaluator[purpose] = by_evaluator.get(purpose, 0.0) + cost

            # Aggregate by model
            model = interaction.model
            by_model[model] = by_model.get(model, 0.0) + cost

            # Track tokens
            total_input += interaction.input_tokens
            total_output += interaction.output_tokens
            total_cached += interaction.cached_tokens

            total += cost

        return {
            "total": round(total, 6),
            "by_evaluator": {k: round(v, 6) for k, v in by_evaluator.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "token_breakdown": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cached_tokens": total_cached,
                "total_tokens": total_input + total_output,
            },
        }


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

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Input data
    output_a: str = Field(..., description="First output being compared")
    output_b: str = Field(..., description="Second output being compared")
    reference: Optional[str] = Field(
        None, description="Optional reference context (e.g., user question)"
    )
    criteria: Optional[str] = Field(
        None, description="Optional criteria used for comparison"
    )

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
    reasoning: str = Field(
        ..., description="Detailed explanation of why this winner was chosen"
    )
    aspect_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Scores for each aspect/criterion, mapping aspect name to scores for output_a and output_b",
    )

    # Metadata
    total_tokens: int = Field(default=0, description="Total tokens used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=_utc_now, description="When comparison completed"
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

    async def total_llm_cost(self, use_actual_pricing: bool = True) -> float:
        """Calculate total LLM cost with accurate pricing.

        Uses llm-prices.com data for accurate cost calculation. Falls back
        to conservative estimates if pricing data unavailable.

        Args:
            use_actual_pricing: If True, use llm-prices data; if False, use simple estimation

        Returns:
            Total cost in USD

        Example:
            >>> result = await compare(output_a, output_b)
            >>> cost = await result.total_llm_cost()
            >>> print(f"Comparison cost: ${cost:.6f}")
        """
        if not use_actual_pricing:
            # Simple fallback: average $0.02 per 1K tokens
            total_tokens = sum(i.total_tokens for i in self.interactions)
            return (total_tokens / 1000) * 0.02

        # Use cost calculator for accurate pricing
        from arbiter_ai.core.cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        await calc.ensure_loaded()

        total = 0.0
        for interaction in self.interactions:
            total += _get_interaction_cost(interaction)

        return total


class BatchEvaluationResult(BaseModel):
    """Result of batch evaluation operation.

    Contains results for all items, error tracking, and aggregate statistics.
    Provides efficient batch processing while maintaining individual result fidelity.

    Example:
        >>> batch_result = await batch_evaluate(
        ...     items=[
        ...         {"output": "Paris is the capital of France", "reference": "Paris is France's capital"},
        ...         {"output": "Tokyo is the capital of Japan", "reference": "Tokyo is Japan's capital"},
        ...         {"output": "Invalid output"},  # This might fail
        ...     ],
        ...     evaluators=["semantic"],
        ...     model="gpt-4o-mini"
        ... )
        >>> print(f"Success rate: {batch_result.successful_items}/{batch_result.total_items}")
        >>> print(f"Total cost: ${await batch_result.total_llm_cost():.4f}")
        >>>
        >>> # Access individual results
        >>> for i, result in enumerate(batch_result.results):
        ...     if result:  # Check if evaluation succeeded
        ...         print(f"Item {i}: score = {result.overall_score:.2f}")
        ...     else:
        ...         error = next(e for e in batch_result.errors if e['index'] == i)
        ...         print(f"Item {i}: failed - {error['error']}")
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Results (None for failed items, preserving order)
    results: List[Optional[EvaluationResult]] = Field(
        default_factory=list,
        description="Evaluation results in original order. None for failed items.",
    )

    # Error tracking
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors that occurred, each with 'index', 'item', and 'error' keys",
    )

    # Statistics
    total_items: int = Field(..., description="Total number of items in batch")
    successful_items: int = Field(..., description="Number of successful evaluations")
    failed_items: int = Field(..., description="Number of failed evaluations")

    # Timing and tokens
    processing_time: float = Field(..., description="Total processing time in seconds")
    total_tokens: int = Field(
        default=0, description="Total tokens across all evaluations"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When batch completed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and context"
    )

    def get_result(self, index: int) -> Optional[EvaluationResult]:
        """Get result by original item index.

        Args:
            index: Zero-based index of the item in the original batch

        Returns:
            EvaluationResult if successful, None if failed or out of range

        Example:
            >>> result = batch_result.get_result(0)
            >>> if result:
            ...     print(f"Score: {result.overall_score}")
        """
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def get_error(self, index: int) -> Optional[Dict[str, Any]]:
        """Get error information for a failed item.

        Args:
            index: Zero-based index of the item in the original batch

        Returns:
            Error dict if item failed, None if successful or not found

        Example:
            >>> error = batch_result.get_error(2)
            >>> if error:
            ...     print(f"Failed: {error['error']}")
        """
        for error in self.errors:
            if error["index"] == index:
                return error
        return None

    async def total_llm_cost(self, use_actual_pricing: bool = True) -> float:
        """Calculate total LLM cost across all successful evaluations.

        Uses llm-prices.com data for accurate cost calculation.

        Args:
            use_actual_pricing: If True, use llm-prices data; if False, use simple estimation

        Returns:
            Total cost in USD across all evaluations

        Example:
            >>> batch_result = await batch_evaluate(items=[...])
            >>> total_cost = await batch_result.total_llm_cost()
            >>> avg_cost_per_item = total_cost / batch_result.total_items
            >>> print(f"Average cost per item: ${avg_cost_per_item:.4f}")
        """
        # Collect costs from all successful results
        successful_results = [r for r in self.results if r is not None]
        if not successful_results:
            return 0.0

        # Calculate cost for each result in parallel
        import asyncio

        costs = await asyncio.gather(
            *[
                r.total_llm_cost(use_actual_pricing=use_actual_pricing)
                for r in successful_results
            ]
        )
        return sum(costs)

    async def cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown across all successful evaluations.

        Returns:
            Dictionary with aggregate cost breakdowns including:
            - total: Total cost across all items
            - per_item_average: Average cost per item
            - by_evaluator: Aggregated cost by evaluator
            - by_model: Aggregated cost by model
            - success_rate: Ratio of successful to total items

        Example:
            >>> breakdown = await batch_result.cost_breakdown()
            >>> print(f"Total: ${breakdown['total']:.4f}")
            >>> print(f"Per item: ${breakdown['per_item_average']:.4f}")
            >>> print(f"Success rate: {breakdown['success_rate']:.1%}")
        """
        successful_results = [r for r in self.results if r is not None]
        if not successful_results:
            return {
                "total": 0.0,
                "per_item_average": 0.0,
                "by_evaluator": {},
                "by_model": {},
                "success_rate": 0.0,
            }

        # Get breakdowns from each result
        import asyncio

        breakdowns = await asyncio.gather(
            *[r.cost_breakdown() for r in successful_results]
        )

        # Aggregate costs by evaluator and model
        by_evaluator: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        total = 0.0

        for breakdown in breakdowns:
            total += breakdown["total"]
            for evaluator, cost in breakdown["by_evaluator"].items():
                by_evaluator[evaluator] = by_evaluator.get(evaluator, 0.0) + cost
            for model, cost in breakdown["by_model"].items():
                by_model[model] = by_model.get(model, 0.0) + cost

        return {
            "total": round(total, 6),
            "per_item_average": (
                round(total / self.total_items, 6) if self.total_items > 0 else 0.0
            ),
            "by_evaluator": {k: round(v, 6) for k, v in by_evaluator.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "success_rate": (
                self.successful_items / self.total_items
                if self.total_items > 0
                else 0.0
            ),
        }
