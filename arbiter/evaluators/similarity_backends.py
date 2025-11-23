"""Similarity computation backends for SemanticEvaluator.

Provides pluggable backends for computing semantic similarity:
- LLMSimilarityBackend: Uses LLM reasoning (default, rich explanations)
- FAISSSimilarityBackend: Uses vector embeddings (fast, free, deterministic)

Example:
    >>> # LLM-based (default)
    >>> backend = LLMSimilarityBackend(llm_client)
    >>> score = await backend.compute_similarity("text1", "text2")
    >>>
    >>> # FAISS-based (requires: pip install arbiter[scale])
    >>> backend = FAISSSimilarityBackend()
    >>> score = await backend.compute_similarity("text1", "text2")
"""

from typing import Any, Dict, Protocol

import numpy as np
from pydantic import BaseModel, Field

from ..core.llm_client import LLMClient

__all__ = [
    "SimilarityBackend",
    "LLMSimilarityBackend",
    "FAISSSimilarityBackend",
    "SimilarityResult",
]


class SimilarityResult(BaseModel):
    """Result from similarity computation."""

    score: float = Field(
        ge=0.0, le=1.0, description="Similarity score (0=different, 1=identical)"
    )
    confidence: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Confidence in assessment"
    )
    explanation: str = Field(
        default="", description="Explanation of similarity (empty for FAISS)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Backend-specific metadata"
    )


class SimilarityBackend(Protocol):
    """Protocol for semantic similarity computation backends.

    All backends must implement compute_similarity() to return a score
    between 0.0 (completely different) and 1.0 (identical meaning).
    """

    async def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            SimilarityResult with score, confidence, explanation

        Raises:
            ValueError: If texts are empty
        """
        ...


class LLMSimilarityBackend:
    """LLM-based similarity computation (default backend).

    Uses LLM reasoning to assess semantic similarity. Provides rich
    explanations but slower and costs API tokens.

    Performance:
        - Latency: 1-3 seconds
        - Cost: ~$0.001 per comparison (gpt-4o-mini)
        - Explanation: Detailed reasoning included

    Example:
        >>> backend = LLMSimilarityBackend(llm_client)
        >>> result = await backend.compute_similarity(
        ...     "Paris is the capital",
        ...     "The capital is Paris"
        ... )
        >>> print(f"Score: {result.score:.2f}")
        >>> print(f"Why: {result.explanation}")
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize LLM backend.

        Args:
            llm_client: Configured LLM client for similarity evaluation
        """
        self.llm_client = llm_client

    async def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Compute similarity using LLM reasoning.

        Args:
            text1: Output text to evaluate
            text2: Reference text for comparison

        Returns:
            SimilarityResult with detailed explanation

        Raises:
            ValueError: If either text is empty
        """
        if not text1 or not text2:
            raise ValueError("Both texts must be non-empty for similarity comparison")

        # Define response model for structured LLM output
        class SemanticResponse(BaseModel):
            score: float = Field(ge=0.0, le=1.0)
            confidence: float = Field(default=0.85, ge=0.0, le=1.0)
            explanation: str
            key_differences: list[str] = Field(default_factory=list)
            key_similarities: list[str] = Field(default_factory=list)

        # System prompt for LLM
        system_prompt = """You are an expert at evaluating semantic similarity between texts.

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

        # User prompt with texts to compare
        user_prompt = f"""Compare the semantic similarity of these two texts:

OUTPUT (to evaluate):
{text1}

REFERENCE (ground truth):
{text2}

Assess how similar they are in MEANING. Consider whether they convey the same information,
even if expressed differently. Provide a detailed analysis."""

        # Call LLM via PydanticAI
        from pydantic_ai import Agent

        agent = Agent(
            model=self.llm_client.model,
            system_prompt=system_prompt,
            result_type=SemanticResponse,
        )

        # Run agent
        result = await agent.run(user_prompt)
        response = result.output

        # Build detailed explanation
        explanation_parts = [response.explanation]

        if response.key_similarities:
            explanation_parts.append(
                "\n\nKey Similarities:\n- " + "\n- ".join(response.key_similarities)
            )

        if response.key_differences:
            explanation_parts.append(
                "\n\nKey Differences:\n- " + "\n- ".join(response.key_differences)
            )

        full_explanation = "".join(explanation_parts)

        return SimilarityResult(
            score=response.score,
            confidence=response.confidence,
            explanation=full_explanation,
            metadata={
                "backend": "llm",
                "similarities_count": len(response.key_similarities),
                "differences_count": len(response.key_differences),
            },
        )


class FAISSSimilarityBackend:
    """FAISS-based similarity computation (optional fast backend).

    Uses sentence-transformers for local embeddings and numpy for cosine
    similarity. Requires: pip install arbiter[scale]

    Performance:
        - Latency: 50-100ms (20-60x faster than LLM)
        - Cost: $0 (runs locally, no API calls)
        - Explanation: None (deterministic number only)

    Trade-offs:
        + Fast, free, offline, deterministic
        - No explanations, less nuanced than LLM reasoning

    Example:
        >>> # Requires: pip install arbiter[scale]
        >>> backend = FAISSSimilarityBackend()
        >>> result = await backend.compute_similarity(
        ...     "Paris is the capital",
        ...     "The capital is Paris"
        ... )
        >>> print(f"Score: {result.score:.2f}")  # e.g., 0.94
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize FAISS backend with sentence-transformer model.

        Args:
            model_name: Sentence-transformers model to use for embeddings
                       Default: all-MiniLM-L6-v2 (fast, 384-dim, good quality)

        Raises:
            ImportError: If sentence-transformers not installed
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "FAISS backend requires sentence-transformers. "
                "Install with: pip install arbiter[scale]"
            )

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    async def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Compute similarity using vector embeddings.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            SimilarityResult with cosine similarity score

        Raises:
            ValueError: If either text is empty
        """
        if not text1 or not text2:
            raise ValueError("Both texts must be non-empty for similarity comparison")

        # Generate embeddings (runs locally, no API call)
        embeddings = self.model.encode([text1, text2])
        vec1, vec2 = embeddings[0], embeddings[1]

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        similarity = dot_product / norm_product

        # Ensure score is in valid range [0, 1]
        # Cosine similarity for text embeddings is typically [0, 1] (negative rare)
        # Use raw cosine as score: 0=different, 1=identical
        # Clamp to [0,1] to handle any edge cases
        score = float(max(0.0, min(1.0, similarity)))

        return SimilarityResult(
            score=score,
            confidence=0.95,  # FAISS is deterministic, high confidence
            explanation="Vector similarity computed using FAISS (no LLM explanation)",
            metadata={
                "backend": "faiss",
                "model": self.model_name,
                "embedding_dim": len(vec1),
                "raw_cosine": float(similarity),
            },
        )
