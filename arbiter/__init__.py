"""Arbiter: Production-grade LLM evaluation framework with complete observability.

Arbiter provides simple APIs, complete observability, and provider-agnostic
infrastructure for evaluating LLM outputs. Built on PydanticAI with automatic
interaction tracking, multiple evaluators, and extensible architecture.

**Key Features:**
- Simple API: Evaluate with 3 lines of code
- Multiple Evaluators: Semantic similarity, custom criteria, pairwise comparison
- Complete Observability: Automatic LLM interaction tracking (unique differentiator)
- Provider-Agnostic: Works with OpenAI, Anthropic, Google, Groq, Mistral, Cohere
- Production-Ready: Middleware, error handling, partial results, registry system

## Quick Start:

    >>> from arbiter import evaluate
    >>>
    >>> # Simple evaluation with automatic client management
    >>> result = await evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris",
    ...     evaluators=["semantic"],
    ...     model="gpt-4o"
    ... )
    >>> print(f"Score: {result.overall_score}")
    >>>
    >>> # Or use evaluator directly for more control
    >>> from arbiter import SemanticEvaluator, LLMManager
    >>> client = await LLMManager.get_client(model="gpt-4o")
    >>> evaluator = SemanticEvaluator(client)
    >>> score = await evaluator.evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris"
    ... )
    >>> print(f"Score: {score.value}")

## Current Evaluators:

- **SemanticEvaluator**: LLM-based semantic similarity evaluation
- **CustomCriteriaEvaluator**: Domain-specific criteria evaluation (single & multi-criteria)
- **FactualityEvaluator**: Hallucination detection and fact verification
- **PairwiseComparisonEvaluator**: A/B testing and model comparison

## Main Components:

- `evaluate()`: Primary async function for evaluation (supports multiple evaluators)
- `compare()`: Pairwise comparison function for A/B testing
- `SemanticEvaluator`, `CustomCriteriaEvaluator`, `PairwiseComparisonEvaluator`: Built-in evaluators
- `EvaluationResult`, `ComparisonResult`: Result models with complete interaction tracking
- `register_evaluator()`: Extend with custom evaluators via registry system

For more information, see the documentation at:
https://docs.arbiter.ai/
"""

from dotenv import load_dotenv

# Core API
from .api import compare, evaluate

# Core components
from .core import (
    AVAILABLE_EVALUATORS,
    ArbiterError,
    BaseEvaluator,
    CachingMiddleware,
    ComparisonResult,
    ConfigurationError,
    ConnectionMetrics,
    CostCalculator,
    EvaluationResult,
    EvaluatorError,
    EvaluatorName,
    LLMClient,
    LLMInteraction,
    LLMManager,
    LoggingMiddleware,
    Metric,
    MetricsMiddleware,
    MetricType,
    Middleware,
    MiddlewarePipeline,
    ModelPricing,
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
    get_available_evaluators,
    get_cost_calculator,
    get_evaluator_class,
    get_global_monitor,
    monitor,
    register_evaluator,
    validate_evaluator_name,
)

# Evaluators
from .evaluators import (
    BasePydanticEvaluator,
    CustomCriteriaEvaluator,
    FactualityEvaluator,
    PairwiseComparisonEvaluator,
    SemanticEvaluator,
)

# Load environment variables
load_dotenv()

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Main API
    "evaluate",
    "compare",
    # Evaluators
    "SemanticEvaluator",
    "CustomCriteriaEvaluator",
    "FactualityEvaluator",
    "PairwiseComparisonEvaluator",
    "BasePydanticEvaluator",
    # Core Models
    "EvaluationResult",
    "ComparisonResult",
    "Score",
    "Metric",
    "LLMInteraction",
    # Interfaces
    "BaseEvaluator",
    "StorageBackend",
    # LLM Client
    "LLMClient",
    "LLMManager",
    "Provider",
    "ConnectionMetrics",
    # Cost Tracking
    "CostCalculator",
    "ModelPricing",
    "get_cost_calculator",
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
    "EvaluatorName",
    # Registry
    "AVAILABLE_EVALUATORS",
    "register_evaluator",
    "get_evaluator_class",
    "get_available_evaluators",
    "validate_evaluator_name",
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
