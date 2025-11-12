"""Arbiter: Streaming LLM evaluation framework with semantic comparison.

Arbiter provides composable evaluation primitives for assessing LLM outputs
through multiple metrics including semantic similarity, factuality, consistency,
and custom criteria. Built on PydanticAI with optional streaming support.

## Quick Start:

    >>> from arbiter import evaluate, SemanticEvaluator
    >>> evaluator = SemanticEvaluator(model="gpt-4o")
    >>> score = await evaluator.score(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris",
    ...     criteria="factuality"
    ... )
    >>> print(score.value)

## Key Features:

- **Semantic Comparison**: Milvus-backed vector similarity for output comparison
- **Multiple Metrics**: Factuality, consistency, semantic similarity, and more
- **Composable**: Build custom evaluation pipelines with middleware
- **Streaming Ready**: Optional ByteWax integration for streaming data
- **Full Observability**: Complete audit trails and performance metrics

## Main Components:

- `evaluate()`: Primary async function for evaluation
- `SemanticEvaluator`: Evaluator with semantic comparison
- `EvaluationResult`: Contains scores, metrics, and metadata
- `Config`: Configuration object for models and behavior

For more information, see the documentation at:
https://docs.arbiter.ai/
"""

from dotenv import load_dotenv

# Core components
from .core import (
    ArbiterError,
    BaseEvaluator,
    CachingMiddleware,
    ConfigurationError,
    ConnectionMetrics,
    EvaluationResult,
    EvaluatorError,
    LLMClient,
    LLMManager,
    LoggingMiddleware,
    Metric,
    MetricsMiddleware,
    MetricType,
    Middleware,
    MiddlewarePipeline,
    ModelProviderError,
    PerformanceMetrics,
    PerformanceMonitor,
    PluginError,
    Provider,
    RateLimitingMiddleware,
    RetryConfig,
    Score,
    StorageBackend,
    StorageError,
    StorageType,
    TimeoutError,
    ValidationError,
    get_global_monitor,
    monitor,
)

# Load environment variables
load_dotenv()

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core Models
    "EvaluationResult",
    "Score",
    "Metric",
    # Interfaces
    "BaseEvaluator",
    "StorageBackend",
    # LLM Client
    "LLMClient",
    "LLMManager",
    "Provider",
    "ConnectionMetrics",
    # Middleware
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    # Monitoring
    "PerformanceMetrics",
    "PerformanceMonitor",
    "get_global_monitor",
    "monitor",
    # Configuration
    "RetryConfig",
    # Types
    "Provider",
    "MetricType",
    "StorageType",
    # Exceptions
    "ArbiterError",
    "ConfigurationError",
    "ModelProviderError",
    "EvaluatorError",
    "ValidationError",
    "StorageError",
    "PluginError",
    "TimeoutError",
]
