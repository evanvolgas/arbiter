"""Core type definitions and enumerations for Arbiter.

This module defines the fundamental types used throughout the Arbiter
evaluation framework, including provider enumerations, evaluator types,
and storage types.
"""

from enum import Enum
from typing import Literal

__all__ = ["Provider", "MetricType", "StorageType", "EvaluatorName"]


class Provider(str, Enum):
    """Enumeration of supported LLM providers.

    Each provider represents a different LLM API service. The enum
    values match PydanticAI's provider naming convention for the
    <provider>:<model> format (e.g., "openai:gpt-4o").

    Attributes:
        OPENAI: OpenAI's GPT models (GPT-3.5, GPT-4, etc.)
        ANTHROPIC: Anthropic's Claude models (Claude 3 family)
        GOOGLE: Google's Gemini models (Gemini Pro, etc.)
        GROQ: Groq's fast inference service for open models
        MISTRAL: Mistral AI models
        COHERE: Cohere models
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"  # For Gemini models
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"


class MetricType(str, Enum):
    """Types of evaluation metrics.

    Defines the standard metrics that evaluators can compute.
    """

    SEMANTIC_SIMILARITY = "semantic_similarity"
    FACTUALITY = "factuality"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    CUSTOM = "custom"


class StorageType(str, Enum):
    """Types of storage backends.

    Defines the available storage backend options for persisting
    evaluation results.
    """

    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    CUSTOM = "custom"


# Type hint for evaluator names
# This will be dynamically updated as evaluators are registered
# For now, includes built-in evaluators
EvaluatorName = Literal["semantic", "custom_criteria"]
