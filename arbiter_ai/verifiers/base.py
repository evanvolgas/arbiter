"""Base classes for external fact verification plugins.

This module provides the abstract base class for implementing external
verification plugins that can enhance factuality evaluation with real-world
data sources.

Verification plugins enable fact-checking against:
- Web search results (SearchVerifier)
- Source citations (CitationVerifier)
- Knowledge bases like Wikipedia (KnowledgeBaseVerifier)

Example:
    >>> from arbiter_ai.verifiers import SearchVerifier
    >>> verifier = SearchVerifier(api_key="tavily_key")
    >>> result = await verifier.verify(
    ...     claim="Paris is the capital of France",
    ...     context="Geography facts"
    ... )
    >>> print(f"Verified: {result.is_verified}, Confidence: {result.confidence}")
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

__all__ = ["FactualityVerifier", "VerificationResult"]


class VerificationResult(BaseModel):
    """Result from external fact verification.

    Attributes:
        is_verified: Whether the claim was verified as factual
        confidence: Confidence score for this verification (0.0-1.0)
        evidence: Supporting evidence found (e.g., search snippets, citations)
        explanation: Human-readable explanation of verification process
        source: Source of verification (e.g., "tavily_search", "wikipedia")
    """

    is_verified: bool = Field(
        ..., description="Whether the claim was verified as factual"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this verification (0.0-1.0)",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence found (e.g., search snippets, citations)",
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of verification process"
    )
    source: str = Field(
        ..., description="Source of verification (e.g., 'tavily_search', 'wikipedia')"
    )


class FactualityVerifier(ABC):
    """Abstract base class for external fact verification plugins.

    Verification plugins enhance factuality evaluation by checking claims
    against external data sources. Each plugin implements a specific
    verification strategy (web search, citations, knowledge bases).

    Subclasses must implement the verify() method to provide their
    verification logic.

    Example:
        >>> class MyVerifier(FactualityVerifier):
        ...     async def verify(self, claim: str, context: Optional[str] = None):
        ...         # Custom verification logic
        ...         return VerificationResult(
        ...             is_verified=True,
        ...             confidence=0.9,
        ...             evidence=["Supporting evidence"],
        ...             explanation="Verification reasoning",
        ...             source="my_verifier"
        ...         )
    """

    @abstractmethod
    async def verify(
        self, claim: str, context: Optional[str] = None
    ) -> VerificationResult:
        """Verify a factual claim against external source.

        Args:
            claim: The factual claim to verify
            context: Optional context to help with verification

        Returns:
            VerificationResult with verification outcome, confidence, and evidence

        Raises:
            Exception: If verification fails (network error, API error, etc.)
        """
        pass
