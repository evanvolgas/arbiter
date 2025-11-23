"""Custom criteria evaluator for domain-specific evaluation.

This evaluator allows you to evaluate LLM outputs against custom criteria,
making it suitable for domain-specific quality assessments like medical
accuracy, legal compliance, brand voice, etc.

## When to Use:

- Domain-specific quality checks (medical accuracy, legal compliance)
- Brand voice and tone evaluation
- Style guide adherence
- Multi-aspect evaluation (accuracy, tone, completeness)
- Reference-free evaluation (no ground truth needed)

## Example:

    >>> evaluator = CustomCriteriaEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output="Medical advice about diabetes",
    ...     criteria="Medical accuracy, HIPAA compliance, appropriate tone for patients"
    ... )
    >>> print(f"Score: {score.value:.2f}")
    >>> print(f"Criteria met: {score.metadata.get('criteria_met', [])}")

## Multi-Criteria Mode:

    >>> scores = await evaluator.evaluate_multi(
    ...     output="Product description",
    ...     criteria={
    ...         "accuracy": "Factually correct product information",
    ...         "persuasiveness": "Compelling call-to-action",
    ...         "brand_voice": "Matches company brand guidelines"
    ...     }
    ... )
    >>> for score in scores:
    ...     print(f"{score.name}: {score.value:.2f}")
"""

import time
from typing import Dict, List, Optional, Type, cast

from pydantic import BaseModel, Field

from ..core.models import LLMInteraction, Score
from .base import BasePydanticEvaluator

__all__ = ["CustomCriteriaEvaluator", "CustomCriteriaResponse", "MultiCriteriaResponse"]


class CustomCriteriaResponse(BaseModel):
    """Structured response for custom criteria evaluation."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall score indicating how well the output meets the criteria (0=doesn't meet, 1=fully meets)",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this evaluation",
    )
    explanation: str = Field(
        ..., description="Detailed explanation of the evaluation and reasoning"
    )
    criteria_met: List[str] = Field(
        default_factory=list,
        description="List of criteria that the output successfully meets",
    )
    criteria_not_met: List[str] = Field(
        default_factory=list,
        description="List of criteria that the output does not meet or partially meets",
    )


class MultiCriteriaResponse(BaseModel):
    """Structured response for multi-criteria evaluation."""

    criteria_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Score for each criterion (0-1). REQUIRED: Must include all criteria names as keys.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate score across all criteria",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this evaluation",
    )
    explanation: str = Field(..., description="Overall explanation of the evaluation")
    criteria_details: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Detailed assessment for each criterion with 'met' status and 'reasoning'",
    )


class CustomCriteriaEvaluator(BasePydanticEvaluator):
    """Evaluates outputs against custom criteria.

    This evaluator is highly flexible and can assess outputs based on
    any criteria you specify, making it ideal for domain-specific
    evaluation needs.

    The evaluator:
    - Supports single criteria (string) or multi-criteria (dict) evaluation
    - Provides detailed breakdown of which criteria are met/not met
    - Returns confidence scores and explanations
    - Works without reference text (criteria-only evaluation)

    Example:
        >>> # Single criteria
        >>> from arbiter import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = CustomCriteriaEvaluator(client)
        >>>
        >>> score = await evaluator.evaluate(
        ...     output="Medical advice about diabetes management",
        ...     criteria="Medical accuracy, HIPAA compliance, appropriate tone for patients"
        ... )
        >>>
        >>> print(f"Score: {score.value:.2f}")
        >>> print(f"Criteria met: {score.metadata.get('criteria_met', [])}")
        >>>
        >>> # Multi-criteria (returns multiple scores)
        >>> scores = await evaluator.evaluate_multi(
        ...     output="Product description",
        ...     criteria={
        ...         "accuracy": "Factually correct product information",
        ...         "persuasiveness": "Compelling call-to-action"
        ...     }
        ... )
        >>> for score in scores:
        ...     print(f"{score.name}: {score.value:.2f}")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "custom_criteria"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining custom criteria evaluation approach."""
        return """You are an expert evaluator specializing in assessing text quality against custom criteria.

Your task is to evaluate how well a given output meets specified criteria. You should:

1. Carefully analyze the output against each criterion
2. Determine which criteria are fully met, partially met, or not met
3. Provide a score from 0.0 (doesn't meet criteria) to 1.0 (fully meets criteria)
4. Explain your reasoning clearly
5. List which specific criteria were met and which were not

Be thorough, fair, and analytical in your evaluation. Consider:
- How well the output addresses each criterion
- Whether the output demonstrates the required qualities
- Any gaps or shortcomings relative to the criteria
- The overall quality and completeness of the output

Provide clear, actionable feedback that helps understand why the score was assigned."""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if not criteria:
            raise ValueError(
                "CustomCriteriaEvaluator requires criteria to be provided. "
                "Use criteria parameter to specify evaluation criteria."
            )

        prompt_parts = [f"OUTPUT TO EVALUATE:\n{output}\n"]

        if reference:
            prompt_parts.append(f"REFERENCE CONTEXT:\n{reference}\n")

        prompt_parts.append(f"EVALUATION CRITERIA:\n{criteria}\n")

        prompt_parts.append(
            """Evaluate how well the output meets the specified criteria. Provide:
1. An overall score (0.0 to 1.0)
2. Your confidence in this assessment
3. A detailed explanation of your reasoning
4. A list of criteria that are met
5. A list of criteria that are not met or only partially met"""
        )

        return "\n".join(prompt_parts)

    def _get_response_type(self) -> Type[BaseModel]:
        """Use custom criteria response model."""
        return CustomCriteriaResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from custom criteria response."""
        criteria_response = cast(CustomCriteriaResponse, response)

        return Score(
            name=self.name,  # Use evaluator's name property for consistency
            value=criteria_response.score,
            confidence=criteria_response.confidence,
            explanation=criteria_response.explanation,
            metadata={
                "criteria_met": criteria_response.criteria_met,
                "criteria_not_met": criteria_response.criteria_not_met,
                "criteria_met_count": len(criteria_response.criteria_met),
                "criteria_not_met_count": len(criteria_response.criteria_not_met),
            },
        )

    async def evaluate_multi(
        self,
        output: str,
        criteria: Dict[str, str],
        reference: Optional[str] = None,
    ) -> List[Score]:
        """Evaluate output against multiple criteria, returning one score per criterion.

        This method evaluates the output against each criterion separately,
        returning a list of Score objects (one per criterion) plus an overall score.

        Args:
            output: The text to evaluate
            criteria: Dictionary mapping criterion names to their descriptions
            reference: Optional reference text for context

        Returns:
            List of Score objects, one per criterion plus an overall score

        Example:
            >>> scores = await evaluator.evaluate_multi(
            ...     output="Product description",
            ...     criteria={
            ...         "accuracy": "Factually correct product information",
            ...         "persuasiveness": "Compelling call-to-action"
            ...     }
            ... )
            >>> for score in scores:
            ...     print(f"{score.name}: {score.value:.2f}")
        """
        if not criteria:
            raise ValueError("criteria dictionary cannot be empty")

        # Format criteria as a structured prompt
        criteria_list = []
        for name, description in criteria.items():
            criteria_list.append(f"- {name}: {description}")

        criteria_text = "\n".join(criteria_list)

        # Create a custom prompt for multi-criteria evaluation
        prompt_parts = [f"OUTPUT TO EVALUATE:\n{output}\n"]

        if reference:
            prompt_parts.append(f"REFERENCE CONTEXT:\n{reference}\n")

        prompt_parts.append(f"EVALUATION CRITERIA:\n{criteria_text}\n")

        prompt_parts.append(
            """Evaluate the output against EACH criterion separately.

IMPORTANT: You MUST provide a 'criteria_scores' dictionary with a score (0.0-1.0) for each criterion name.

For each criterion, analyze:
1. How well the output meets that specific criterion (score from 0.0 to 1.0)
2. Whether the criterion is met (yes/no/partial)
3. Detailed reasoning for the score

Then provide:
- criteria_scores: Dictionary mapping each criterion name to its score
- overall_score: Aggregate score across all criteria (0.0-1.0)
- confidence: Your confidence level (0.0-1.0)
- explanation: Overall summary
- criteria_details: Dictionary with 'met' status and 'reasoning' for each criterion

Example structure:
{
  "criteria_scores": {"accuracy": 0.9, "persuasiveness": 0.8, ...},
  "overall_score": 0.85,
  "confidence": 0.88,
  "explanation": "Overall assessment...",
  "criteria_details": {
    "accuracy": {"met": "yes", "reasoning": "Because..."},
    "persuasiveness": {"met": "partial", "reasoning": "Because..."}
  }
}"""
        )

        # Use multi-criteria response type
        start_time = time.time()
        await self._ensure_client()
        system_prompt = self._get_system_prompt()
        agent = self.llm_client.create_agent(system_prompt, MultiCriteriaResponse)

        # Run evaluation
        user_prompt = "\n".join(prompt_parts)
        result = await agent.run(user_prompt)
        multi_response = cast(MultiCriteriaResponse, result.output)

        # Extract detailed token usage from PydanticAI result
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        tokens_used = 0  # Backward compatibility

        try:
            if hasattr(result, "usage"):
                usage = result.usage()  # Call as function
                if usage:
                    # PydanticAI usage object structure
                    input_tokens = getattr(usage, "request_tokens", 0)
                    output_tokens = getattr(usage, "response_tokens", 0)
                    tokens_used = getattr(usage, "total_tokens", 0)

                    # Some providers support cached tokens
                    if hasattr(usage, "cached_tokens"):
                        cached_tokens = getattr(usage, "cached_tokens", 0)
        except Exception:
            # Fallback if usage() call fails or not available
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
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Cost calculation failed for model {self.llm_client.model}: {e}"
            )
            pass

        # Record interaction for transparency
        latency = time.time() - start_time
        interaction = LLMInteraction(
            prompt=user_prompt,
            response=(
                multi_response.model_dump_json()
                if hasattr(multi_response, "model_dump_json")
                else str(multi_response)
            ),
            model=self.llm_client.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            tokens_used=tokens_used
            or (input_tokens + output_tokens),  # Backward compat
            cost=cost,
            latency=latency,
            purpose=f"{self.name}_multi_evaluation",
            metadata={
                "evaluator": self.name,
                "system_prompt": system_prompt,
                "has_reference": reference is not None,
                "criteria_count": len(criteria),
                "is_multi_criteria": True,
            },
        )
        self.interactions.append(interaction)

        # Create individual scores for each criterion
        scores: List[Score] = []

        # If LLM didn't provide criteria_scores, create them from overall score
        if not multi_response.criteria_scores:
            # Fallback: assign overall score to all criteria
            for criterion_name in criteria.keys():
                multi_response.criteria_scores[criterion_name] = (
                    multi_response.overall_score
                )

        for criterion_name, criterion_score in multi_response.criteria_scores.items():
            criterion_detail = multi_response.criteria_details.get(criterion_name, {})
            met_status = criterion_detail.get("met", "unknown")
            reasoning = criterion_detail.get("reasoning", "")

            score = Score(
                name=f"custom_criteria_{criterion_name}",
                value=criterion_score,
                confidence=multi_response.confidence,
                explanation=reasoning or f"Evaluation of {criterion_name} criterion",
                metadata={
                    "criterion": criterion_name,
                    "met_status": met_status,
                    "is_multi_criteria": True,
                },
            )
            scores.append(score)

        # Add overall score
        overall_score = Score(
            name="custom_criteria_overall",
            value=multi_response.overall_score,
            confidence=multi_response.confidence,
            explanation=multi_response.explanation,
            metadata={
                "is_multi_criteria": True,
                "criterion_count": len(criteria),
            },
        )
        scores.append(overall_score)

        return scores
