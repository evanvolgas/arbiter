"""Semantic similarity evaluator with pluggable backends.

This evaluator assesses how semantically similar two texts are,
even if they use different wording. Supports two backends:

- **LLM backend (default)**: Rich explanations, slower, costs API tokens
- **FAISS backend (optional)**: Fast, free, deterministic (requires: pip install arbiter[scale])

## When to Use:

- Evaluating paraphrase quality
- Checking if outputs capture reference meaning
- Comparing different phrasings of the same concept
- Translation or summarization quality

## Backends:

**LLM (default)**:
- Performance: 1-3s, ~$0.001/comparison
- Provides detailed explanations
- Best for: Understanding why scores differ

**FAISS (optional)**:
- Performance: 50ms, $0/comparison
- No explanations (just scores)
- Best for: Batch processing, development/testing

## Example:

    >>> # LLM backend (default)
    >>> evaluator = SemanticEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris"
    ... )
    >>> print(f"Similarity: {score.value:.2f}")
    >>> print(f"Why: {score.explanation}")
    >>>
    >>> # FAISS backend (requires: pip install arbiter[scale])
    >>> evaluator = SemanticEvaluator(llm_client, backend="faiss")
    >>> score = await evaluator.evaluate(
    ...     output="Paris is the capital",
    ...     reference="The capital is Paris"
    ... )
    >>> print(f"Similarity: {score.value:.2f}")  # Fast, free, no explanation
"""

from typing import Literal, Optional, Type

from pydantic import BaseModel, Field

from ..core.llm_client import LLMClient
from ..core.models import Score
from .base import BasePydanticEvaluator
from .similarity_backends import (
    FAISSSimilarityBackend,
    LLMSimilarityBackend,
    SimilarityBackend,
)

__all__ = ["SemanticEvaluator", "SemanticResponse"]


class SemanticResponse(BaseModel):
    """Structured response for semantic similarity evaluation."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score (0=completely different, 1=identical meaning)",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this similarity assessment",
    )
    explanation: str = Field(
        ..., description="Explanation of why this similarity score was assigned"
    )
    key_differences: list[str] = Field(
        default_factory=list,
        description="Key semantic differences between the texts",
    )
    key_similarities: list[str] = Field(
        default_factory=list,
        description="Key semantic similarities between the texts",
    )


class SemanticEvaluator(BasePydanticEvaluator):
    """Evaluates semantic similarity with pluggable backends.

    Supports two computation backends:
    - **LLM (default)**: Rich explanations, slower (~2s), costs tokens (~$0.001)
    - **FAISS (optional)**: Fast (~50ms), free, deterministic, no explanations

    The evaluator:
    - Identifies core meaning in both texts
    - Compares semantic content, not just words
    - Provides detailed explanation (LLM) or score only (FAISS)
    - Returns confidence in the assessment

    Example:
        >>> # LLM backend (default) - Rich explanations
        >>> from arbiter import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = SemanticEvaluator(client)
        >>>
        >>> score = await evaluator.evaluate(
        ...     output="The quick brown fox jumps over the lazy dog",
        ...     reference="A fast brown fox leaps above a sleepy canine"
        ... )
        >>> print(f"Similarity: {score.value:.2f}")
        >>> print(f"Explanation: {score.explanation}")
        >>>
        >>> # FAISS backend (optional) - Fast and free
        >>> evaluator_fast = SemanticEvaluator(client, backend="faiss")
        >>> score_fast = await evaluator_fast.evaluate(
        ...     output="Paris is the capital",
        ...     reference="The capital is Paris"
        ... )
        >>> print(f"Similarity: {score_fast.value:.2f}")  # No explanation
    """

    def __init__(
        self,
        llm_client: LLMClient,
        backend: Literal["llm", "faiss"] = "llm",
    ):
        """Initialize semantic evaluator with specified backend.

        Args:
            llm_client: LLM client for evaluation (used by LLM backend)
            backend: Similarity computation backend
                    - "llm": LLM reasoning (default, rich explanations)
                    - "faiss": Vector embeddings (fast, free, requires arbiter[scale])

        Raises:
            ImportError: If backend="faiss" but sentence-transformers not installed
            ValueError: If backend is not "llm" or "faiss"

        Example:
            >>> # Default LLM backend
            >>> evaluator = SemanticEvaluator(llm_client)
            >>>
            >>> # Fast FAISS backend (requires: pip install arbiter[scale])
            >>> evaluator_fast = SemanticEvaluator(llm_client, backend="faiss")
        """
        super().__init__(llm_client)
        self.backend_type = backend

        # Initialize similarity backend
        if backend == "faiss":
            self._similarity_backend: SimilarityBackend = FAISSSimilarityBackend()
        elif backend == "llm":
            self._similarity_backend = LLMSimilarityBackend(llm_client)
        else:
            raise ValueError(f"Invalid backend: {backend}. Must be 'llm' or 'faiss'")

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None,
    ) -> Score:
        """Evaluate semantic similarity between output and reference.

        Uses the configured backend (LLM or FAISS) for similarity computation.
        FAISS backend only works when reference is provided.

        Args:
            output: The text to evaluate
            reference: Optional reference text for comparison
            criteria: Optional criteria for evaluation (LLM backend only)

        Returns:
            Score object with similarity score and metadata

        Raises:
            ValueError: If FAISS backend used without reference

        Example:
            >>> # With reference (works for both backends)
            >>> score = await evaluator.evaluate(
            ...     output="Paris is the capital",
            ...     reference="The capital is Paris"
            ... )
            >>>
            >>> # Without reference (LLM backend only)
            >>> score = await evaluator.evaluate(
            ...     output="This text is clear and coherent"
            ... )
        """
        # FAISS backend: Use direct similarity computation
        if self.backend_type == "faiss":
            if not reference:
                raise ValueError(
                    "FAISS backend requires a reference text for comparison. "
                    "Use LLM backend for reference-free evaluation."
                )

            # Compute similarity using FAISS backend
            result = await self._similarity_backend.compute_similarity(
                output, reference
            )

            return Score(
                name=self.name,
                value=result.score,
                confidence=result.confidence,
                explanation=result.explanation,
                metadata=result.metadata,
            )

        # LLM backend: Use template method pattern (default behavior)
        return await super().evaluate(output, reference, criteria)

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "semantic"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining semantic evaluation approach."""
        return """You are an expert at evaluating semantic similarity between texts.

Your task is to assess how similar two texts are in MEANING, not just in wording.

Consider:
- Core concepts and ideas conveyed
- Factual accuracy and information content
- Intent and purpose of the text
- Logical relationships between ideas

Ignore:
- Exact wording or phrasing
- Grammatical structure
- Writing style or tone (unless it changes meaning)

Provide:
- A similarity score from 0.0 (completely different meaning) to 1.0 (identical meaning)
- Your confidence in this assessment
- Clear explanation of your reasoning
- Key similarities and differences you identified

Be precise and analytical in your evaluation."""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if not reference:
            # If no reference, we can't do semantic comparison
            # Fall back to reference-free evaluation with criteria
            if criteria:
                return f"""Evaluate the semantic quality of this text based on the criteria: {criteria}

Text to evaluate:
{output}

Provide your semantic quality assessment."""
            else:
                # No reference and no criteria - evaluate general semantic coherence
                return f"""Evaluate the semantic coherence and clarity of this text:

{output}

Assess how well the text conveys clear meaning and logical ideas."""

        # Standard semantic similarity evaluation
        return f"""Compare the semantic similarity of these two texts:

OUTPUT (to evaluate):
{output}

REFERENCE (ground truth):
{reference}

Assess how similar they are in MEANING. Consider whether they convey the same information,
even if expressed differently. Provide a detailed analysis."""

    def _get_response_type(self) -> Type[BaseModel]:
        """Use custom semantic response model."""
        return SemanticResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from semantic response."""
        semantic_response = response  # Type hint: it's a SemanticResponse

        # Build detailed explanation
        explanation_parts = [semantic_response.explanation]

        if (
            hasattr(semantic_response, "key_similarities")
            and semantic_response.key_similarities
        ):
            explanation_parts.append(
                "\n\nKey Similarities:\n- "
                + "\n- ".join(semantic_response.key_similarities)
            )

        if (
            hasattr(semantic_response, "key_differences")
            and semantic_response.key_differences
        ):
            explanation_parts.append(
                "\n\nKey Differences:\n- "
                + "\n- ".join(semantic_response.key_differences)
            )

        full_explanation = "".join(explanation_parts)

        return Score(
            name=self.name,  # Use evaluator's name property for consistency
            value=semantic_response.score,
            confidence=semantic_response.confidence,
            explanation=full_explanation,
            metadata={
                "similarities_count": len(
                    getattr(semantic_response, "key_similarities", [])
                ),
                "differences_count": len(
                    getattr(semantic_response, "key_differences", [])
                ),
            },
        )
