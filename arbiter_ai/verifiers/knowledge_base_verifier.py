"""Knowledge base fact verification using Wikipedia API.

This verifier validates claims against structured knowledge bases
like Wikipedia for well-established facts.

Example:
    >>> verifier = KnowledgeBaseVerifier()
    >>> result = await verifier.verify(
    ...     claim="Paris is the capital of France"
    ... )
    >>> print(f"Verified: {result.is_verified}")
"""

import urllib.parse
import urllib.request
from json import loads
from typing import Optional

from .base import FactualityVerifier, VerificationResult

__all__ = ["KnowledgeBaseVerifier"]


class KnowledgeBaseVerifier(FactualityVerifier):
    """Verifies claims against Wikipedia knowledge base.

    This verifier checks factual claims against Wikipedia using the
    MediaWiki API. It's useful for verifying well-established facts
    about entities, dates, locations, and general knowledge.

    The verifier:
    - Extracts key entities from the claim
    - Searches Wikipedia for relevant articles
    - Checks if claim is supported by article content
    - Returns verification with Wikipedia excerpts as evidence

    Wikipedia API is free and requires no authentication, making this
    verifier easy to use without additional setup.

    Args:
        max_results: Maximum number of Wikipedia articles to check (default: 3)

    Example:
        >>> verifier = KnowledgeBaseVerifier()
        >>>
        >>> # Verify factual claims
        >>> result = await verifier.verify("Paris is the capital of France")
        >>> print(f"Verified: {result.is_verified}")
        >>>
        >>> # Check historical facts
        >>> result = await verifier.verify("World War II ended in 1945")
        >>> print(f"Evidence: {result.evidence}")
    """

    def __init__(self, max_results: int = 3) -> None:
        """Initialize knowledge base verifier.

        Args:
            max_results: Maximum number of Wikipedia articles to check (default: 3)
        """
        self.max_results = max_results
        self.api_base = "https://en.wikipedia.org/w/api.php"

    async def verify(
        self, claim: str, context: Optional[str] = None
    ) -> VerificationResult:
        """Verify claim against Wikipedia knowledge base.

        Args:
            claim: The factual claim to verify
            context: Optional context to help with search (not used currently)

        Returns:
            VerificationResult with Wikipedia-based verification

        Raises:
            Exception: If Wikipedia API call fails
        """
        try:
            # Search Wikipedia for relevant articles
            search_results = self._search_wikipedia(claim)

            if not search_results:
                return VerificationResult(
                    is_verified=False,
                    confidence=0.3,
                    evidence=[],
                    explanation=f"No relevant Wikipedia articles found for: '{claim}'",
                    source="wikipedia",
                )

            # Get content from top results
            evidence = []
            total_similarity = 0.0

            for page_title in search_results[: self.max_results]:
                content = self._get_wikipedia_content(page_title)
                if content:
                    # Check if claim appears in content
                    similarity = self._compute_similarity(claim, content)
                    total_similarity += similarity

                    if similarity > 0.5:
                        snippet = self._extract_snippet(claim, content)
                        evidence.append(f"{page_title}: {snippet}")

            # Compute verification confidence
            avg_similarity = total_similarity / len(search_results[: self.max_results])

            if avg_similarity >= 0.7:
                confidence = 0.9
                is_verified = True
                explanation = f"Claim strongly supported by Wikipedia articles: {', '.join(search_results[:self.max_results])}"
            elif avg_similarity >= 0.5:
                confidence = 0.7
                is_verified = True
                explanation = "Claim partially supported by Wikipedia articles"
            else:
                confidence = 0.4
                is_verified = False
                explanation = "Claim not well-supported by Wikipedia content"

            return VerificationResult(
                is_verified=is_verified,
                confidence=confidence,
                evidence=evidence,
                explanation=explanation,
                source="wikipedia",
            )

        except Exception as e:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                evidence=[],
                explanation=f"Wikipedia verification failed: {str(e)}",
                source="wikipedia",
            )

    def _search_wikipedia(self, query: str) -> list[str]:
        """Search Wikipedia for relevant articles.

        Args:
            query: Search query

        Returns:
            List of article titles
        """
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(self.max_results),
            "format": "json",
        }

        url = f"{self.api_base}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = loads(response.read().decode("utf-8"))
                if len(data) >= 2 and isinstance(data[1], list):
                    return [str(title) for title in data[1]]  # Article titles
                return []
        except Exception:
            return []

    def _get_wikipedia_content(self, title: str) -> str:
        """Get content of a Wikipedia article.

        Args:
            title: Article title

        Returns:
            Article content (first paragraph)
        """
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": title,
            "format": "json",
        }

        url = f"{self.api_base}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = loads(response.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    extract = page_data.get("extract", "")
                    if extract and isinstance(extract, str):
                        return str(extract)
                return ""
        except Exception:
            return ""

    def _compute_similarity(self, claim: str, content: str) -> float:
        """Compute simple word overlap similarity.

        Args:
            claim: The claim
            content: The Wikipedia content

        Returns:
            Similarity score (0.0-1.0)
        """
        claim_lower = claim.lower()
        content_lower = content.lower()

        # Direct substring match
        if claim_lower in content_lower:
            return 0.95

        # Word overlap
        claim_words = set(word for word in claim_lower.split() if len(word) > 3)
        content_words = set(content_lower.split())

        if not claim_words:
            return 0.0

        overlap = claim_words.intersection(content_words)
        return len(overlap) / len(claim_words)

    def _extract_snippet(self, claim: str, content: str, length: int = 200) -> str:
        """Extract relevant snippet from content.

        Args:
            claim: The claim
            content: The content
            length: Maximum snippet length

        Returns:
            Relevant snippet
        """
        claim_lower = claim.lower()

        # Try to find sentence containing claim words
        sentences = content.split(".")
        claim_words = set(claim_lower.split())

        best_sentence = ""
        best_overlap = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words.intersection(sentence_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

        if best_sentence:
            return best_sentence[:length]

        return content[:length]
