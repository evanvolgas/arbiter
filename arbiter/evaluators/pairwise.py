"""Pairwise comparison evaluator for comparing two LLM outputs.

This evaluator compares two outputs against each other to determine which
is better, making it ideal for A/B testing, model comparison, and output selection.

## When to Use:

- A/B testing different prompts or models
- Comparing model outputs side-by-side
- Selecting the best output from multiple candidates
- Evaluating relative quality differences
- Aspect-level comparison (accuracy, clarity, completeness, etc.)

## Example:

    >>> evaluator = PairwiseComparisonEvaluator(llm_client)
    >>> comparison = await evaluator.compare(
    ...     output_a="GPT-4 response",
    ...     output_b="Claude response",
    ...     criteria="accuracy, clarity, completeness"
    ... )
    >>> print(f"Winner: {comparison.winner}")
    >>> print(f"Confidence: {comparison.confidence:.2f}")
"""

import logging
import time
from typing import Dict, List, Literal, Optional, Type, cast

from pydantic import BaseModel, Field

from ..core.models import ComparisonResult, LLMInteraction, Score
from .base import BasePydanticEvaluator

logger = logging.getLogger(__name__)

__all__ = [
    "PairwiseComparisonEvaluator",
    "PairwiseResponse",
    "AspectComparison",
]


class AspectComparison(BaseModel):
    """Comparison result for a single aspect/criterion."""

    aspect: str = Field(..., description="Name of the aspect being compared")
    output_a_score: float = Field(
        ..., ge=0.0, le=1.0, description="Score for output_a on this aspect"
    )
    output_b_score: float = Field(
        ..., ge=0.0, le=1.0, description="Score for output_b on this aspect"
    )
    reasoning: str = Field(
        ..., description="Explanation of the comparison for this aspect"
    )


class PairwiseResponse(BaseModel):
    """Structured response for pairwise comparison evaluation."""

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
    aspect_comparisons: List[AspectComparison] = Field(
        default_factory=list,
        description="Detailed comparison for each aspect/criterion",
    )


class PairwiseComparisonEvaluator(BasePydanticEvaluator):
    """Compares two LLM outputs to determine which is better.

    This evaluator supports two usage patterns:
    1. compare() - Explicit A/B comparison of two outputs
    2. evaluate() - Compare output against reference (standard evaluator API)

    The evaluator:
    - Compares outputs aspect-by-aspect (if criteria provided)
    - Determines overall winner or tie
    - Provides detailed reasoning for the decision
    - Returns aspect-level scores for transparency

    Usage Pattern 1 - compare() for A/B testing:
        >>> from arbiter import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = PairwiseComparisonEvaluator(client)
        >>>
        >>> comparison = await evaluator.compare(
        ...     output_a="GPT-4 response: Paris is the capital of France, founded in 3rd century BC.",
        ...     output_b="Claude response: The capital of France is Paris, established around 250 BC.",
        ...     criteria="accuracy, clarity, completeness",
        ...     reference="What is the capital of France?"
        ... )
        >>>
        >>> print(f"Winner: {comparison.winner}")
        >>> print(f"Confidence: {comparison.confidence:.2f}")
        >>> print(f"Reasoning: {comparison.reasoning}")
        >>>
        >>> # Check aspect scores
        >>> for aspect in comparison.aspect_scores:
        ...     print(f"{aspect}: A={comparison.aspect_scores[aspect]['output_a']:.2f}, "
        ...           f"B={comparison.aspect_scores[aspect]['output_b']:.2f}")

    Usage Pattern 2 - evaluate() for standard evaluation API:
        >>> # Compare output against reference (standard evaluator interface)
        >>> score = await evaluator.evaluate(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     criteria="accuracy, completeness"
        ... )
        >>>
        >>> print(f"Score: {score.value:.2f}")  # High if output better, low if reference better
        >>> print(f"Winner: {score.metadata['winner']}")  # output_a, output_b, or tie
        >>> print(f"Confidence: {score.confidence:.2f}")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "pairwise_comparison"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining pairwise comparison approach."""
        return """You are an expert evaluator specializing in comparing two texts to determine which is better.

Your task is to compare two outputs and determine:
1. Which output is superior overall (or if they're equivalent - a tie)
2. Your confidence in this decision
3. Detailed reasoning explaining your choice
4. Aspect-by-aspect comparison if criteria are provided

When comparing, consider:
- Accuracy and factual correctness
- Clarity and coherence
- Completeness and detail
- Relevance to the task
- Overall quality and usefulness

Be fair, analytical, and thorough. If outputs are genuinely equivalent, declare a tie.
Provide clear reasoning that helps understand your decision."""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for pairwise comparison.

        When evaluate() is called with a single output, this compares the output
        against the reference text (treating them as output_a and output_b).

        Args:
            output: The output to evaluate (treated as output_a)
            reference: Reference text to compare against (treated as output_b)
            criteria: Optional evaluation criteria

        Returns:
            Formatted comparison prompt

        Raises:
            ValueError: If reference is not provided
        """
        if not reference:
            raise ValueError(
                "PairwiseComparisonEvaluator.evaluate() requires a reference text. "
                "The output is compared against the reference. "
                "Use compare() method for explicit output_a vs output_b comparison."
            )

        # Build comparison prompt (output vs reference)
        prompt_parts = [
            "Compare these two outputs and determine which is better:",
            "",
            "OUTPUT A:",
            output,
            "",
            "OUTPUT B (REFERENCE):",
            reference,
        ]

        if criteria:
            prompt_parts.extend(
                [
                    "",
                    "EVALUATION CRITERIA:",
                    criteria,
                    "",
                    "Compare the outputs on each criterion and provide:",
                    "1. A score (0.0-1.0) for each output on each criterion",
                    "2. Reasoning for each comparison",
                    "3. An overall winner (output_a, output_b, or tie)",
                    "4. Overall confidence and reasoning",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "",
                    "Compare the outputs overall and determine:",
                    "1. Which output is better (output_a, output_b, or tie)",
                    "2. Your confidence in this decision",
                    "3. Detailed reasoning explaining your choice",
                ]
            )

        return "\n".join(prompt_parts)

    def _get_response_type(self) -> Type[BaseModel]:
        """Use pairwise response model."""
        return PairwiseResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Compute score from pairwise comparison response.

        For evaluate() calls, this converts the pairwise comparison result
        (output vs reference) into a single score. If output_a wins, score is high.
        If output_b (reference) wins, score is low. Ties get medium score.

        Args:
            response: PairwiseResponse from LLM

        Returns:
            Score representing how well output compares to reference
        """
        pairwise_response = cast(PairwiseResponse, response)

        # Convert winner to a score:
        # - output_a wins (output better than reference) → high score (0.8-1.0)
        # - tie → medium score (0.5)
        # - output_b wins (reference better than output) → low score (0.0-0.2)
        if pairwise_response.winner == "output_a":
            # Output is better than reference
            base_score = 0.9
        elif pairwise_response.winner == "tie":
            # Output is equivalent to reference
            base_score = 0.5
        else:  # output_b wins
            # Reference is better than output
            base_score = 0.1

        # Modulate by confidence (low confidence → move toward 0.5)
        confidence = pairwise_response.confidence
        final_score = base_score * confidence + 0.5 * (1 - confidence)

        # Build explanation
        explanation_parts = [pairwise_response.reasoning]

        if pairwise_response.aspect_comparisons:
            explanation_parts.append("\n\nAspect Comparisons:")
            for aspect in pairwise_response.aspect_comparisons:
                explanation_parts.append(
                    f"- {aspect.aspect}: Output={aspect.output_a_score:.2f}, "
                    f"Reference={aspect.output_b_score:.2f} ({aspect.reasoning})"
                )

        return Score(
            name=self.name,
            value=final_score,
            confidence=confidence,
            explanation="\n".join(explanation_parts),
            metadata={
                "winner": pairwise_response.winner,
                "aspect_count": len(pairwise_response.aspect_comparisons),
            },
        )

    async def compare(
        self,
        output_a: str,
        output_b: str,
        criteria: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> ComparisonResult:
        """Compare two outputs and determine which is better.

        This is the main entry point for pairwise comparison. It compares
        two outputs against each other (and optionally against criteria or reference)
        to determine a winner.

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            criteria: Optional criteria for comparison (e.g., "accuracy, clarity, completeness")
            reference: Optional reference context (e.g., user question or ground truth)

        Returns:
            ComparisonResult with winner, confidence, reasoning, and aspect scores

        Raises:
            ValueError: If outputs are empty
            EvaluatorError: If comparison fails

        Example:
            >>> comparison = await evaluator.compare(
            ...     output_a="GPT-4 response",
            ...     output_b="Claude response",
            ...     criteria="accuracy, clarity"
            ... )
            >>> print(f"Winner: {comparison.winner}")
        """
        start_time = time.time()

        # Input validation
        if not output_a or not output_a.strip():
            raise ValueError("output_a cannot be empty")
        if not output_b or not output_b.strip():
            raise ValueError("output_b cannot be empty")

        try:
            # Ensure client is initialized
            await self._ensure_client()

            # Build comparison prompt
            prompt_parts = [
                "Compare these two outputs and determine which is better:",
                "",
                "OUTPUT A:",
                output_a,
                "",
                "OUTPUT B:",
                output_b,
            ]

            if reference:
                prompt_parts.extend(["", "REFERENCE CONTEXT:", reference])

            if criteria:
                prompt_parts.extend(
                    [
                        "",
                        "EVALUATION CRITERIA:",
                        criteria,
                        "",
                        "Compare the outputs on each criterion and provide:",
                        "1. A score (0.0-1.0) for each output on each criterion",
                        "2. Reasoning for each comparison",
                        "3. An overall winner (output_a, output_b, or tie)",
                        "4. Overall confidence and reasoning",
                    ]
                )
            else:
                prompt_parts.extend(
                    [
                        "",
                        "Compare the outputs overall and determine:",
                        "1. Which output is better (output_a, output_b, or tie)",
                        "2. Your confidence in this decision",
                        "3. Detailed reasoning explaining your choice",
                    ]
                )

            user_prompt = "\n".join(prompt_parts)

            # Create PydanticAI agent with structured output
            system_prompt = self._get_system_prompt()
            response_type = self._get_response_type()
            agent = self.llm_client.create_agent(system_prompt, response_type)

            # Run comparison
            result = await agent.run(user_prompt)
            pairwise_response = cast(PairwiseResponse, result.output)

            # Extract detailed token usage
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0
            tokens_used = 0  # Backward compatibility

            try:
                if hasattr(result, "usage"):
                    usage = result.usage()
                    if usage:
                        # PydanticAI usage object structure
                        input_tokens = getattr(usage, "request_tokens", 0)
                        output_tokens = getattr(usage, "response_tokens", 0)
                        tokens_used = getattr(usage, "total_tokens", 0)

                        # Some providers support cached tokens
                        if hasattr(usage, "cached_tokens"):
                            cached_tokens = getattr(usage, "cached_tokens", 0)
            except Exception:
                pass

            # Calculate cost using cost calculator
            cost = None
            try:
                from arbiter.core.cost_calculator import get_cost_calculator

                calc = get_cost_calculator()
                await calc.ensure_loaded()

                cost = calc.calculate_cost(
                    model=self.llm_client.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )
            except Exception as e:
                # If cost calculation fails, continue without cost
                logger.warning(
                    f"Cost calculation failed for model {self.llm_client.model}: {e}"
                )
                pass

            # Build aspect_scores dictionary from aspect_comparisons
            aspect_scores: Dict[str, Dict[str, float]] = {}
            for aspect_comp in pairwise_response.aspect_comparisons:
                aspect_scores[aspect_comp.aspect] = {
                    "output_a": aspect_comp.output_a_score,
                    "output_b": aspect_comp.output_b_score,
                }

            # Record interaction
            latency = time.time() - start_time
            interaction = LLMInteraction(
                prompt=user_prompt,
                response=(
                    pairwise_response.model_dump_json()
                    if hasattr(pairwise_response, "model_dump_json")
                    else str(pairwise_response)
                ),
                model=self.llm_client.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                tokens_used=tokens_used
                or (input_tokens + output_tokens),  # Backward compat
                cost=cost,
                latency=latency,
                purpose=f"{self.name}_comparison",
                metadata={
                    "evaluator": self.name,
                    "system_prompt": system_prompt,
                    "has_reference": reference is not None,
                    "has_criteria": criteria is not None,
                    "winner": pairwise_response.winner,
                },
            )
            self.interactions.append(interaction)

            # Create ComparisonResult
            processing_time = time.time() - start_time
            return ComparisonResult(
                output_a=output_a,
                output_b=output_b,
                reference=reference,
                criteria=criteria,
                winner=pairwise_response.winner,
                confidence=pairwise_response.confidence,
                reasoning=pairwise_response.reasoning,
                aspect_scores=aspect_scores,
                total_tokens=tokens_used,
                processing_time=processing_time,
                interactions=self.interactions.copy(),
                metadata={
                    "evaluator": self.name,
                    "aspect_count": len(aspect_scores),
                },
            )

        except Exception as e:
            from ..core.exceptions import EvaluatorError

            raise EvaluatorError(
                f"Pairwise comparison failed in {self.name}",
                details={"error": str(e), "evaluator": self.name},
            ) from e
