"""Core infrastructure for Arbiter evaluation framework.

This module contains the fundamental components:
- Configuration management
- LLM client pooling
- Middleware system
- Monitoring and metrics
- Plugin infrastructure
- Exception handling
- Retry logic
- Circuit breaker pattern
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .cost_calculator import CostCalculator, ModelPricing, get_cost_calculator
from .exceptions import (
    ArbiterError,
    CircuitBreakerOpenError,
    ConfigurationError,
    EvaluatorError,
    ModelProviderError,
    PluginError,
    StorageError,
    TimeoutError,
    ValidationError,
)
from .interfaces import BaseEvaluator, StorageBackend
from .llm_client import LLMClient, LLMManager, LLMResponse, Provider
from .llm_client_pool import ConnectionMetrics, LLMClientPool, PoolConfig
from .middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewarePipeline,
    RateLimitingMiddleware,
    monitor as monitor_context,
)
from .models import ComparisonResult, EvaluationResult, LLMInteraction, Metric, Score
from .monitoring import PerformanceMetrics, PerformanceMonitor, get_global_monitor, monitor
from .retry import RETRY_PERSISTENT, RETRY_QUICK, RETRY_STANDARD, RetryConfig, with_retry
from .registry import (
    AVAILABLE_EVALUATORS,
    get_available_evaluators,
    get_evaluator_class,
    register_evaluator,
    validate_evaluator_name,
)
from .type_defs import MiddlewareContext
from .types import EvaluatorName, MetricType, Provider, StorageType

__all__ = [
    # Exceptions
    "ArbiterError",
    "ConfigurationError",
    "ModelProviderError",
    "EvaluatorError",
    "ValidationError",
    "StorageError",
    "PluginError",
    "TimeoutError",
    "CircuitBreakerOpenError",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    # Cost Calculator
    "CostCalculator",
    "ModelPricing",
    "get_cost_calculator",
    # Models
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
    "LLMResponse",
    "Provider",
    "LLMClientPool",
    "PoolConfig",
    "ConnectionMetrics",
    # Middleware
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    "monitor_context",
    # Monitoring
    "PerformanceMetrics",
    "PerformanceMonitor",
    "get_global_monitor",
    "monitor",
    # Retry
    "RetryConfig",
    "with_retry",
    "RETRY_QUICK",
    "RETRY_STANDARD",
    "RETRY_PERSISTENT",
    # Types
    "Provider",
    "MetricType",
    "StorageType",
    "EvaluatorName",
    "MiddlewareContext",
    # Registry
    "AVAILABLE_EVALUATORS",
    "register_evaluator",
    "get_evaluator_class",
    "get_available_evaluators",
    "validate_evaluator_name",
]
