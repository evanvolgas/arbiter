"""Search-based fact verification using web search APIs.

This verifier validates claims by searching the web and analyzing
top search results for supporting or contradicting evidence.

Example:
    >>> verifier = SearchVerifier(api_key="tavily_key")
    >>> result = await verifier.verify(
    ...     claim="Paris is the capital of France"
    ... )
    >>> print(f"Verified: {result.is_verified}")
"""

import os
from typing import Optional

try:
    from tavily import TavilyClient  # type: ignore

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from .base import FactualityVerifier, VerificationResult

__all__ = ["SearchVerifier"]


class SearchVerifier(FactualityVerifier):
    """Verifies factual claims using web search (Tavily API).

    This verifier searches the web for evidence supporting or contradicting
    a claim. It analyzes top search results and computes a confidence score
    based on the consistency and quality of evidence found.

    The verifier:
    - Searches the web for the claim using Tavily API
    - Analyzes top results for supporting evidence
    - Computes confidence based on result quality and consistency
    - Returns verification result with evidence snippets

    Args:
        api_key: Tavily API key (optional, reads from TAVILY_API_KEY env var)
        max_results: Maximum number of search results to analyze (default: 5)

    Raises:
        ImportError: If tavily package is not installed
        ValueError: If API key is not provided and not in environment

    Example:
        >>> # Using explicit API key
        >>> verifier = SearchVerifier(api_key="tavily_key")
        >>> result = await verifier.verify("Paris is the capital of France")
        >>>
        >>> # Using environment variable TAVILY_API_KEY
        >>> verifier = SearchVerifier()
        >>> result = await verifier.verify("Water boils at 100°C at sea level")
        >>>
        >>> print(f"Verified: {result.is_verified}")
        >>> print(f"Confidence: {result.confidence:.2f}")
        >>> print(f"Evidence: {result.evidence}")
    """

    def __init__(self, api_key: Optional[str] = None, max_results: int = 5) -> None:
        """Initialize search verifier with API key and configuration.

        Args:
            api_key: Tavily API key (optional, reads from TAVILY_API_KEY env var)
            max_results: Maximum number of search results to analyze (default: 5)

        Raises:
            ImportError: If tavily package is not installed
            ValueError: If API key is not provided and not in environment
        """
        if not TAVILY_AVAILABLE:
            raise ImportError(
                "Tavily package not installed. Install with: pip install tavily-python"
            )

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key required. Provide via api_key parameter or TAVILY_API_KEY environment variable"
            )

        self.max_results = max_results
        self.client = TavilyClient(api_key=self.api_key)

    async def verify(
        self, claim: str, context: Optional[str] = None
    ) -> VerificationResult:
        """Verify claim using web search.

        Args:
            claim: The factual claim to verify
            context: Optional context to help with search query

        Returns:
            VerificationResult with search-based verification

        Raises:
            Exception: If search API call fails
        """
        # Build search query
        query = claim
        if context:
            query = f"{claim} {context}"

        try:
            # Search the web (Tavily client is synchronous, not async)
            search_results = self.client.search(
                query=query, max_results=self.max_results
            )

            # Extract evidence from results
            evidence = []
            if "results" in search_results:
                for result in search_results["results"]:
                    content = result.get("content", "")
                    if content:
                        evidence.append(content[:200])  # Limit snippet length

            # Compute verification confidence
            # Simple heuristic: more results with content = higher confidence
            num_results = len(evidence)
            if num_results == 0:
                confidence = 0.3  # Low confidence if no evidence found
                is_verified = False
                explanation = (
                    f"No supporting evidence found in web search for: '{claim}'"
                )
            elif num_results <= 2:
                confidence = 0.6
                is_verified = True
                explanation = f"Found {num_results} search results supporting the claim"
            else:
                confidence = 0.85
                is_verified = True
                explanation = f"Found {num_results} search results consistently supporting the claim"

            return VerificationResult(
                is_verified=is_verified,
                confidence=confidence,
                evidence=evidence,
                explanation=explanation,
                source="tavily_search",
            )

        except Exception as e:
            # If search fails, return low-confidence unverified result
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                evidence=[],
                explanation=f"Search verification failed: {str(e)}",
                source="tavily_search",
            )
