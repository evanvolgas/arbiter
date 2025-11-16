"""Evaluators for assessing LLM outputs.

This module provides evaluators for different aspects of LLM outputs:
- Semantic similarity between output and reference (LLM or FAISS backend)
- Custom criteria evaluation (domain-specific quality checks)
- Factuality checking and hallucination detection
- Pairwise comparison for A/B testing
- Consistency evaluation (planned)
- Relevance scoring (planned)

All evaluators inherit from BasePydanticEvaluator and automatically
track LLM interactions for full observability.

Similarity Backends:
- LLMSimilarityBackend: Default, rich explanations
- FAISSSimilarityBackend: Optional, fast and free (requires: pip install arbiter[scale])
"""

from .base import BasePydanticEvaluator, EvaluatorResponse
from .custom_criteria import (
    CustomCriteriaEvaluator,
    CustomCriteriaResponse,
    MultiCriteriaResponse,
)
from .factuality import FactualityEvaluator, FactualityResponse
from .pairwise import (
    AspectComparison,
    PairwiseComparisonEvaluator,
    PairwiseResponse,
)
from .semantic import SemanticEvaluator, SemanticResponse
from .similarity_backends import (
    SimilarityBackend,
    SimilarityResult,
    LLMSimilarityBackend,
    FAISSSimilarityBackend,
)

__all__ = [
    "BasePydanticEvaluator",
    "EvaluatorResponse",
    "SemanticEvaluator",
    "SemanticResponse",
    "CustomCriteriaEvaluator",
    "CustomCriteriaResponse",
    "MultiCriteriaResponse",
    "FactualityEvaluator",
    "FactualityResponse",
    "PairwiseComparisonEvaluator",
    "PairwiseResponse",
    "AspectComparison",
    "SimilarityBackend",
    "SimilarityResult",
    "LLMSimilarityBackend",
    "FAISSSimilarityBackend",
]
