"""Exception hierarchy for Arbiter evaluation framework.

This module defines all custom exceptions used throughout Arbiter.
All exceptions inherit from ArbiterError for easy catching.

## Exception Hierarchy:

```
ArbiterError (base)
├── ConfigurationError - Invalid configuration
├── ModelProviderError - LLM API failures
├── EvaluatorError - Evaluator execution failures
├── StorageError - Storage backend failures
├── PluginError - Plugin loading/execution failures
├── ValidationError - Input validation failures
├── TimeoutError - Operation timeout
└── CircuitBreakerOpenError - Circuit breaker blocking requests
```

## Usage:

    >>> from arbiter_ai.core.exceptions import ModelProviderError
    >>>
    >>> try:
    ...     result = await evaluate(output, reference)
    ... except ModelProviderError as e:
    ...     print(f"LLM API failed: {e}")
    ... except ArbiterError as e:
    ...     print(f"Arbiter error: {e}")
"""

from typing import Any, Dict, Optional

__all__ = [
    "ArbiterError",
    "ConfigurationError",
    "ModelProviderError",
    "EvaluatorError",
    "StorageError",
    "PluginError",
    "ValidationError",
    "TimeoutError",
    "CircuitBreakerOpenError",
]


class ArbiterError(Exception):
    """Base exception for all Arbiter errors.

    All custom exceptions in Arbiter inherit from this class.
    This allows catching all Arbiter-specific errors with a single
    except clause.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize error with message and optional details.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context like
                error codes, failed values, etc.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(ArbiterError):
    """Raised when configuration is invalid.

    Examples:
        - Invalid model name
        - Out-of-range temperature value
        - Missing required API keys
        - Incompatible configuration options

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid model name",
        ...     details={"model": "invalid-model", "valid_models": ["gpt-4", "gpt-3.5"]}
        ... )
    """


class ModelProviderError(ArbiterError):
    """Raised when LLM API calls fail.

    This exception covers all LLM provider errors including:
        - API authentication failures
        - Rate limiting
        - API errors (500, 503, etc.)
        - Network connectivity issues
        - Invalid requests

    This is a retryable error type - operations will automatically
    retry when this exception is raised.

    Example:
        >>> raise ModelProviderError(
        ...     "OpenAI API rate limit exceeded",
        ...     details={"status_code": 429, "retry_after": 60}
        ... )
    """


class EvaluatorError(ArbiterError):
    """Raised when evaluator execution fails.

    Examples:
        - Evaluator initialization failure
        - Invalid evaluator configuration
        - Evaluator processing errors
        - Score calculation failures

    Example:
        >>> raise EvaluatorError(
        ...     "Semantic evaluator failed",
        ...     details={"evaluator": "semantic", "reason": "embeddings unavailable"}
        ... )
    """


class StorageError(ArbiterError):
    """Raised when storage operations fail.

    Examples:
        - File system errors
        - Database connection failures
        - Serialization errors
        - Storage quota exceeded

    Example:
        >>> raise StorageError(
        ...     "Failed to save evaluation result",
        ...     details={"backend": "redis", "reason": "connection timeout"}
        ... )
    """


class PluginError(ArbiterError):
    """Raised when plugin operations fail.

    Examples:
        - Plugin not found
        - Plugin loading failure
        - Invalid plugin interface
        - Plugin execution error

    Example:
        >>> raise PluginError(
        ...     "Failed to load storage plugin",
        ...     details={"plugin": "s3-storage", "reason": "missing dependencies"}
        ... )
    """


class ValidationError(ArbiterError):
    """Raised when input validation fails.

    Examples:
        - Empty output text
        - Invalid reference format
        - Out-of-range parameters
        - Missing required fields

    Example:
        >>> raise ValidationError(
        ...     "Output text cannot be empty",
        ...     details={"field": "output", "value": ""}
        ... )
    """


class TimeoutError(ArbiterError):
    """Raised when operations exceed time limits.

    This is a retryable error type - operations may automatically
    retry when this exception is raised, depending on retry configuration.

    Examples:
        - LLM API call timeout
        - Evaluation timeout
        - Storage operation timeout

    Example:
        >>> raise TimeoutError(
        ...     "Evaluation timed out",
        ...     details={"timeout_seconds": 30, "elapsed_seconds": 35}
        ... )
    """


class CircuitBreakerOpenError(ArbiterError):
    """Raised when circuit breaker is open and blocking requests.

    This exception indicates that too many failures have occurred and
    the circuit breaker has entered the OPEN state to prevent cascading
    failures. Requests are temporarily blocked until the circuit enters
    HALF_OPEN state for recovery testing.

    This is a temporary error - clients should wait and retry after
    the circuit breaker timeout period.

    Examples:
        - LLM provider experiencing outage
        - Multiple consecutive API failures
        - Provider rate limiting across multiple requests

    Example:
        >>> raise CircuitBreakerOpenError(
        ...     "Circuit breaker is open",
        ...     details={
        ...         "failure_count": 5,
        ...         "retry_after": 60,
        ...         "last_failure": "API timeout"
        ...     }
        ... )
    """
