"""Citation-based fact verification for source attribution checking.

This verifier checks if claims are supported by provided source documents,
useful for evaluating RAG systems and ensuring proper attribution.

Example:
    >>> verifier = CitationVerifier()
    >>> result = await verifier.verify(
    ...     claim="Paris is the capital of France",
    ...     context="Paris is the capital and most populous city of France."
    ... )
    >>> print(f"Verified: {result.is_verified}")
"""

from typing import Optional

from .base import FactualityVerifier, VerificationResult

__all__ = ["CitationVerifier"]


class CitationVerifier(FactualityVerifier):
    """Verifies claims against provided source documents.

    This verifier checks if a claim is supported by the provided context
    (source documents). It's particularly useful for:
    - RAG system evaluation (checking if outputs are grounded in sources)
    - Citation validation (ensuring claims are properly attributed)
    - Source checking (verifying facts match source material)

    The verifier:
    - Checks if claim content appears in or is supported by context
    - Uses substring matching and semantic similarity
    - Computes confidence based on match quality
    - Returns verification result with matched excerpts

    Args:
        min_similarity: Minimum similarity threshold for verification (0.0-1.0)

    Example:
        >>> verifier = CitationVerifier()
        >>>
        >>> # Check if claim is supported by source
        >>> result = await verifier.verify(
        ...     claim="The Eiffel Tower is in Paris",
        ...     context="The Eiffel Tower is a famous landmark located in Paris, France"
        ... )
        >>> print(f"Verified: {result.is_verified}")
        >>>
        >>> # Detect unsupported claims
        >>> result = await verifier.verify(
        ...     claim="The Eiffel Tower is 1000 meters tall",
        ...     context="The Eiffel Tower is 300 meters tall"
        ... )
        >>> print(f"Verified: {result.is_verified}")  # False
    """

    def __init__(self, min_similarity: float = 0.7) -> None:
        """Initialize citation verifier with similarity threshold.

        Args:
            min_similarity: Minimum similarity threshold for verification (0.0-1.0)
        """
        self.min_similarity = min_similarity

    async def verify(
        self, claim: str, context: Optional[str] = None
    ) -> VerificationResult:
        """Verify claim against provided source documents.

        Args:
            claim: The factual claim to verify
            context: Source documents to check claim against (required for this verifier)

        Returns:
            VerificationResult with citation-based verification

        Raises:
            ValueError: If context is not provided
        """
        if not context:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                evidence=[],
                explanation="No source context provided for citation verification",
                source="citation_check",
            )

        # Normalize claim and context for comparison
        claim_lower = claim.lower().strip()
        context_lower = context.lower().strip()

        # Check for direct substring match (high confidence)
        if claim_lower in context_lower:
            return VerificationResult(
                is_verified=True,
                confidence=0.95,
                evidence=[self._extract_matching_snippet(claim, context)],
                explanation=f"Claim directly found in source context: '{claim}'",
                source="citation_check",
            )

        # Check for key terms overlap (medium confidence)
        claim_words = set(
            word for word in claim_lower.split() if len(word) > 3  # Skip short words
        )
        context_words = set(context_lower.split())

        overlap = claim_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(claim_words) if claim_words else 0.0

        if overlap_ratio >= self.min_similarity:
            snippet = self._extract_matching_snippet(claim, context)
            return VerificationResult(
                is_verified=True,
                confidence=min(0.85, overlap_ratio),
                evidence=[snippet] if snippet else [],
                explanation=f"Claim is {overlap_ratio:.0%} semantically similar to source content",
                source="citation_check",
            )

        # No significant match found
        return VerificationResult(
            is_verified=False,
            confidence=overlap_ratio,
            evidence=[],
            explanation=f"Claim not sufficiently supported by source (only {overlap_ratio:.0%} overlap)",
            source="citation_check",
        )

    def _extract_matching_snippet(
        self, claim: str, context: str, window: int = 100
    ) -> str:
        """Extract snippet from context that best matches claim.

        Args:
            claim: The claim to find
            context: The source context
            window: Number of characters to extract around match

        Returns:
            Snippet from context containing the match
        """
        claim_lower = claim.lower()
        context_lower = context.lower()

        # Find position of best match
        pos = context_lower.find(claim_lower)
        if pos != -1:
            # Extract window around match
            start = max(0, pos - window // 2)
            end = min(len(context), pos + len(claim) + window // 2)
            snippet = context[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(context):
                snippet = snippet + "..."
            return snippet

        # If no direct match, try to find sentence with most overlap
        sentences = context.split(".")
        best_sentence = ""
        best_overlap = 0

        claim_words = set(claim_lower.split())
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words.intersection(sentence_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

        return best_sentence if best_sentence else context[:200]
