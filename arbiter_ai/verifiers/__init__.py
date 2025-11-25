"""External fact verification plugins for enhanced factuality evaluation.

This module provides plugins that enhance factuality evaluation by verifying
claims against external data sources:

- SearchVerifier: Web search validation using Tavily API
- CitationVerifier: Source attribution checking for RAG systems
- KnowledgeBaseVerifier: Structured knowledge validation using Wikipedia

Verification plugins are optional and can be used with FactualityEvaluator
to improve hallucination detection and fact-checking accuracy.

Example:
    >>> from arbiter_ai.verifiers import SearchVerifier, CitationVerifier
    >>> from arbiter import evaluate
    >>>
    >>> # Setup verification plugins
    >>> verifiers = [
    ...     SearchVerifier(api_key="tavily_key"),
    ...     CitationVerifier()
    ... ]
    >>>
    >>> # Use with factuality evaluator
    >>> result = await evaluate(
    ...     output="Paris is the capital of France",
    ...     evaluators=["factuality"],
    ...     factuality_verifiers=verifiers
    ... )
"""

from .base import FactualityVerifier, VerificationResult
from .citation_verifier import CitationVerifier
from .knowledge_base_verifier import KnowledgeBaseVerifier
from .search_verifier import SearchVerifier

__all__ = [
    "FactualityVerifier",
    "VerificationResult",
    "SearchVerifier",
    "CitationVerifier",
    "KnowledgeBaseVerifier",
]
