"""Factuality evaluator for detecting hallucinations and verifying claims.

This evaluator assesses the factual accuracy of LLM outputs by identifying
claims and verifying them against reference text or general knowledge. It's
essential for catching hallucinations and ensuring trustworthy AI outputs.

## When to Use:

- Detecting hallucinations in LLM outputs
- Verifying factual claims against source documents (RAG validation)
- Fact-checking before publishing content
- Ensuring accuracy in critical domains (medical, legal, financial)
- Evaluating information retrieval quality

## Example:

    >>> evaluator = FactualityEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output="Paris is the capital of France, founded in 1985",
    ...     reference="Paris is the capital of France, founded in ancient times"
    ... )
    >>> print(f"Factuality score: {score.value:.2f}")
    >>> print(f"Non-factual claims: {score.metadata.get('non_factual_claims', [])}")
    Factuality score: 0.67
    Non-factual claims: ['Paris was founded in 1985']
"""

from typing import List, Optional, Type, cast

from pydantic import BaseModel, Field

from ..core.models import Score
from .base import BasePydanticEvaluator

__all__ = ["FactualityEvaluator", "FactualityResponse"]


class FactualityResponse(BaseModel):
    """Structured response for factuality evaluation."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factuality score (0=all claims are false, 1=all claims are factual)",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this factuality assessment",
    )
    explanation: str = Field(
        ..., description="Explanation of the factuality assessment and reasoning"
    )
    factual_claims: List[str] = Field(
        default_factory=list,
        description="Claims that are factually correct and verified",
    )
    non_factual_claims: List[str] = Field(
        default_factory=list,
        description="Claims that are factually incorrect, misleading, or hallucinated",
    )
    uncertain_claims: List[str] = Field(
        default_factory=list,
        description="Claims that cannot be verified as true or false with available information",
    )


class FactualityEvaluator(BasePydanticEvaluator):
    """Evaluates factual accuracy and detects hallucinations.

    This evaluator identifies factual claims in the output and verifies them
    against reference text or evaluates them for objective verifiability.
    It's critical for ensuring trustworthy AI outputs and catching hallucinations.

    The evaluator:
    - Extracts factual claims from the output
    - Verifies claims against reference text (if provided)
    - Categorizes claims as factual, non-factual, or uncertain
    - Provides a score based on the ratio of factual to total claims
    - Returns detailed explanations for each claim

    Modes:
    - With reference: Verify all claims against reference text
    - Standalone: Identify objectively verifiable claims and assess plausibility
    - With criteria: Focus on specific types of facts (dates, names, numbers, etc.)

    Example:
        >>> # Create evaluator
        >>> from arbiter import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = FactualityEvaluator(client)
        >>>
        >>> # Detect hallucinations by comparing to reference
        >>> score = await evaluator.evaluate(
        ...     output="The Eiffel Tower was built in 1889 and is 324 meters tall",
        ...     reference="The Eiffel Tower, built in 1889, stands 300 meters tall"
        ... )
        >>>
        >>> print(f"Factuality: {score.value:.2f}")
        >>> print(f"Non-factual: {score.metadata['non_factual_claims']}")
        >>>
        >>> # Standalone fact-checking
        >>> score = await evaluator.evaluate(
        ...     output="Water boils at 100°C at sea level under standard pressure"
        ... )
        >>> print(f"Factuality: {score.value:.2f}")
        >>>
        >>> # Check LLM interactions
        >>> interactions = evaluator.get_interactions()
        >>> print(f"Made {len(interactions)} LLM calls")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "factuality"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining factuality evaluation approach."""
        return """You are an expert fact-checker specializing in identifying and verifying factual claims.

Your task is to evaluate the factual accuracy of text by:

1. EXTRACTING CLAIMS: Identify all factual claims (statements that can be verified as true or false)
2. VERIFYING CLAIMS: Check each claim against reference text (if provided) or general knowledge
3. CATEGORIZING CLAIMS:
   - Factual: Claims that are correct and can be verified
   - Non-factual: Claims that are incorrect, misleading, or hallucinated
   - Uncertain: Claims that cannot be verified with available information

4. SCORING: Calculate factuality score based on the ratio:
   score = factual_claims / (factual_claims + non_factual_claims)
   - Uncertain claims do not affect the score (neutral)
   - No claims = score 1.0 (no misinformation)
   - All claims correct = score 1.0
   - All claims wrong = score 0.0

Important Guidelines:
- Be precise: A claim is only "factual" if it's completely accurate
- Detect hallucinations: Flag any invented facts, wrong dates, incorrect numbers
- Handle uncertainty: If you can't verify a claim, mark it as "uncertain", not factual
- Consider context: Some claims may be true in context but misleading
- Check specifics: Pay attention to dates, names, numbers, locations

Provide:
- A factuality score from 0.0 (all claims false) to 1.0 (all claims true)
- Your confidence in this assessment
- Clear categorization of each claim
- Detailed explanation of your reasoning"""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if reference:
            # Verify claims against reference text
            return f"""Evaluate the factual accuracy of the OUTPUT by verifying claims against the REFERENCE.

OUTPUT (to fact-check):
{output}

REFERENCE (source of truth):
{reference}

Your task:
1. Extract all factual claims from the OUTPUT
2. Verify each claim against the REFERENCE
3. Categorize each claim as:
   - Factual: Claim is supported by the reference
   - Non-factual: Claim contradicts or is not supported by the reference
   - Uncertain: Cannot determine from the reference alone

4. Calculate factuality score = factual / (factual + non_factual)

Provide your assessment with clear categorization of claims."""

        elif criteria:
            # Focus on specific types of facts
            return f"""Evaluate the factual accuracy of this text, focusing on: {criteria}

TEXT TO FACT-CHECK:
{output}

Your task:
1. Extract factual claims related to the specified criteria
2. Assess the accuracy and verifiability of each claim
3. Categorize claims as factual, non-factual, or uncertain
4. Pay special attention to the criteria areas

Provide your factuality assessment."""

        else:
            # Standalone fact-checking (assess plausibility and common knowledge)
            return f"""Evaluate the factual accuracy of this text using general knowledge and common sense.

TEXT TO FACT-CHECK:
{output}

Your task:
1. Extract all factual claims (statements that can be verified)
2. Assess each claim for:
   - Accuracy based on general knowledge
   - Plausibility and logical consistency
   - Common hallucination patterns (invented names, wrong dates, impossible claims)

3. Categorize each claim as:
   - Factual: Claim aligns with general knowledge and is verifiable
   - Non-factual: Claim is demonstrably false or implausible
   - Uncertain: Cannot verify without additional sources

4. Calculate factuality score

Provide your assessment with detailed reasoning."""

    def _get_response_type(self) -> Type[BaseModel]:
        """Use factuality response model."""
        return FactualityResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from factuality response."""
        factuality_response = cast(FactualityResponse, response)

        # Build detailed explanation
        explanation_parts = [factuality_response.explanation]

        if (
            hasattr(factuality_response, "factual_claims")
            and factuality_response.factual_claims
        ):
            explanation_parts.append(
                "\n\nFactual Claims (Verified):\n- "
                + "\n- ".join(factuality_response.factual_claims)
            )

        if (
            hasattr(factuality_response, "non_factual_claims")
            and factuality_response.non_factual_claims
        ):
            explanation_parts.append(
                "\n\nNon-Factual Claims (Incorrect/Hallucinated):\n- "
                + "\n- ".join(factuality_response.non_factual_claims)
            )

        if (
            hasattr(factuality_response, "uncertain_claims")
            and factuality_response.uncertain_claims
        ):
            explanation_parts.append(
                "\n\nUncertain Claims (Cannot Verify):\n- "
                + "\n- ".join(factuality_response.uncertain_claims)
            )

        full_explanation = "".join(explanation_parts)

        return Score(
            name=self.name,
            value=factuality_response.score,
            confidence=factuality_response.confidence,
            explanation=full_explanation,
            metadata={
                "factual_claims": factuality_response.factual_claims,
                "non_factual_claims": factuality_response.non_factual_claims,
                "uncertain_claims": factuality_response.uncertain_claims,
                "factual_count": len(
                    getattr(factuality_response, "factual_claims", [])
                ),
                "non_factual_count": len(
                    getattr(factuality_response, "non_factual_claims", [])
                ),
                "uncertain_count": len(
                    getattr(factuality_response, "uncertain_claims", [])
                ),
            },
        )
