"""Core infrastructure for Arbiter evaluation framework.

This module contains the fundamental components:
- Configuration management
- LLM client pooling
- Middleware system
- Monitoring and metrics
- Plugin infrastructure
- Exception handling
- Retry logic
"""

from .exceptions import (
    ArbiterError,
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
from .models import EvaluationResult, Metric, Score
from .monitoring import PerformanceMetrics, PerformanceMonitor, get_global_monitor, monitor
from .retry import RETRY_PERSISTENT, RETRY_QUICK, RETRY_STANDARD, RetryConfig, with_retry
from .type_defs import MiddlewareContext
from .types import MetricType, Provider, StorageType

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
    # Models
    "EvaluationResult",
    "Score",
    "Metric",
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
    "MiddlewareContext",
]
