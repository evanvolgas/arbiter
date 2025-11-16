"""Middleware system for adding cross-cutting functionality to Arbiter.

This module provides a flexible middleware pipeline that allows you to
add functionality like logging, metrics, caching, and rate limiting
without modifying core Arbiter code.

## Middleware Pattern:

Middleware components form a chain where each component can:
1. Process the request before passing it on
2. Modify the request or context
3. Handle the response after processing
4. Short-circuit the chain if needed

## Built-in Middleware:

- **LoggingMiddleware**: Logs all evaluation operations
- **MetricsMiddleware**: Collects performance metrics
- **CachingMiddleware**: Caches evaluation results
- **RateLimitingMiddleware**: Limits request rate

## Usage:

    >>> from arbiter import evaluate, MiddlewarePipeline
    >>> from arbiter.core.middleware import LoggingMiddleware, MetricsMiddleware
    >>>
    >>> # Create middleware pipeline
    >>> pipeline = MiddlewarePipeline([
    ...     LoggingMiddleware(log_level="DEBUG"),
    ...     MetricsMiddleware(),
    ... ])
    >>>
    >>> # Use with evaluate()
    >>> result = await evaluate(
    ...     output="...",
    ...     reference="...",
    ...     middleware=pipeline
    ... )
    >>>
    >>> # Access metrics
    >>> metrics = pipeline.get_middleware(MetricsMiddleware)
    >>> print(metrics.get_metrics())

## Custom Middleware:

    >>> class MyMiddleware(Middleware):
    ...     async def process(self, output, reference, next_handler, context):
    ...         # Pre-processing
    ...         print(f"Evaluating: {output[:50]}...")
    ...
    ...         # Call next in chain
    ...         result = await next_handler(output, reference)
    ...
    ...         # Post-processing
    ...         print(f"Completed with score: {result.overall_score}")
    ...
    ...         return result
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, cast

from .models import ComparisonResult, EvaluationResult
from .type_defs import MiddlewareContext

logger = logging.getLogger(__name__)

__all__ = [
    "Middleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    "MiddlewarePipeline",
    "monitor",
]


class Middleware(ABC):
    """Abstract base class for all middleware components.

    Middleware allows you to intercept and modify the evaluation process
    without changing core Arbiter logic. Each middleware can inspect or
    modify the request, add to the context, and process the response.

    The middleware pattern enables:
    - Logging and debugging
    - Performance monitoring
    - Caching and optimization
    - Security and rate limiting
    - Request/response transformation

    Example:
        >>> class TimingMiddleware(Middleware):
        ...     async def process(self, output, reference, next_handler, context):
        ...         start = time.time()
        ...         result = await next_handler(output, reference)
        ...         elapsed = time.time() - start
        ...         print(f"Evaluation took {elapsed:.2f} seconds")
        ...         return result
    """

    @abstractmethod
    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> EvaluationResult:
        """Process the request through this middleware.

        This is the main method that each middleware must implement. It
        receives the request, can perform pre-processing, must call the
        next handler, and can perform post-processing.

        Args:
            output: The LLM output to be evaluated. Middleware can modify
                this before passing it to the next handler.
            reference: Reference text for comparison (may be None for
                reference-free evaluation).
            next_handler: Async callable representing the next middleware
                in the chain or the final evaluate handler. Must be called
                to continue processing.
            context: Mutable dictionary shared between all middleware in
                the pipeline. Used to pass data between middleware components.
                Common keys include 'evaluators', 'metrics', 'config'.

        Returns:
            EvaluationResult from the evaluation process. Middleware can
            inspect or modify this before returning.

        Raises:
            Any exception from the evaluation process. Middleware can
            catch and handle exceptions or let them propagate.

        Example:
            >>> async def process(self, output, reference, next_handler, context):
            ...     # Pre-processing
            ...     context['start_time'] = time.time()
            ...
            ...     # Must call next handler
            ...     result = await next_handler(output, reference)
            ...
            ...     # Post-processing
            ...     context['duration'] = time.time() - context['start_time']
            ...
            ...     return result
        """


class LoggingMiddleware(Middleware):
    """Middleware that logs all evaluation operations.

    Provides detailed logging of the evaluation process including:
    - Output and reference text (truncated)
    - Configuration (evaluators, metrics)
    - Processing time
    - Results (scores, pass/fail status)
    - Errors with full context

    Useful for debugging, monitoring, and understanding how
    evaluations are performed.

    Example:
        >>> # Basic usage
        >>> middleware = LoggingMiddleware()
        >>>
        >>> # With custom log level
        >>> middleware = LoggingMiddleware(log_level="DEBUG")
        >>>
        >>> # In pipeline
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(LoggingMiddleware())
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize logging middleware with specified level.

        Args:
            log_level: Logging level as string. Valid values are:
                - "DEBUG": Detailed information for debugging
                - "INFO": General informational messages (default)
                - "WARNING": Warning messages
                - "ERROR": Error messages only
                Case-insensitive.
        """
        self.log_level = getattr(logging, log_level.upper())

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> EvaluationResult:
        """Log the evaluation process."""
        start_time = time.time()

        logger.log(self.log_level, f"Starting evaluation for output: {output[:100]}...")
        if reference:
            logger.log(self.log_level, f"Reference: {reference[:100]}...")

        metrics_list = context.get("metrics", [])
        logger.log(
            self.log_level,
            f"Context: evaluators={context.get('evaluators', [])}, "
            f"metrics={len(metrics_list) if metrics_list else 0}",
        )

        try:
            result = await next_handler(output, reference)

            elapsed = time.time() - start_time
            logger.log(self.log_level, f"Evaluation completed in {elapsed:.2f}s")

            # Handle both EvaluationResult and ComparisonResult
            if hasattr(result, "overall_score"):
                # EvaluationResult
                logger.log(
                    self.log_level,
                    f"Result: overall_score={result.overall_score:.3f}, "
                    f"passed={result.passed}, "
                    f"num_scores={len(result.scores)}",
                )
            else:
                # ComparisonResult
                logger.log(
                    self.log_level,
                    f"Result: winner={result.winner}, "
                    f"confidence={result.confidence:.3f}",
                )

            return cast(EvaluationResult, result)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Evaluation failed after {elapsed:.2f}s: {type(e).__name__}: {e!s}"
            )
            raise


class MetricsMiddleware(Middleware):
    """Middleware that collects detailed metrics about evaluations.

    Tracks comprehensive metrics including:
    - Total requests and success rate
    - Processing time statistics
    - Score distributions
    - Token usage and costs
    - Error rates and types

    Metrics are accumulated across all requests and can be retrieved
    for analysis or monitoring dashboards.

    Example:
        >>> metrics_mw = MetricsMiddleware()
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(metrics_mw)
        >>>
        >>> # Process some requests
        >>> for output, ref in test_cases:
        ...     await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Get metrics
        >>> stats = metrics_mw.get_metrics()
        >>> print(f"Average time: {stats['average_time']:.2f}s")
        >>> print(f"Average score: {stats['average_score']:.3f}")
    """

    def __init__(self) -> None:
        """Initialize metrics collection with zero counters.

        Creates a metrics dictionary that tracks:
        - total_requests: Number of evaluate() calls
        - total_time: Cumulative processing time
        - average_score: Running average of overall scores
        - errors: Count of failed requests
        - llm_calls: Total LLM API calls made
        - tokens_used: Total tokens consumed
        """
        self.metrics = {
            "total_requests": 0,
            "total_time": 0.0,
            "average_score": 0.0,
            "errors": 0,
            "llm_calls": 0,
            "tokens_used": 0,
            "passed_count": 0,
        }

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> EvaluationResult:
        """Collect metrics about the evaluation."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        try:
            # Track LLM calls via context
            initial_llm_calls = context.get("llm_calls", 0)

            result = await next_handler(output, reference)

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["total_time"] += elapsed

            # Update average score
            total = self.metrics["total_requests"]
            old_avg = self.metrics["average_score"]
            # Handle both EvaluationResult (has overall_score) and ComparisonResult (has confidence)
            score_value = getattr(result, "overall_score", None) or getattr(result, "confidence", None) or 0.0
            self.metrics["average_score"] = (
                old_avg * (total - 1) + float(score_value)
            ) / total

            # Track pass/fail (ComparisonResult doesn't have 'passed' attribute)
            if hasattr(result, "passed") and result.passed:
                self.metrics["passed_count"] += 1

            # Track LLM calls
            final_llm_calls = context.get("llm_calls", 0)
            if isinstance(final_llm_calls, (int, float)) and isinstance(
                initial_llm_calls, (int, float)
            ):
                self.metrics["llm_calls"] += int(final_llm_calls - initial_llm_calls)

            # Track tokens
            self.metrics["tokens_used"] += result.total_tokens

            return cast(EvaluationResult, result)

        except Exception:
            self.metrics["errors"] += 1
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with calculated averages."""
        metrics = self.metrics.copy()

        # Calculate averages
        if metrics["total_requests"] > 0:
            metrics["avg_time_per_request"] = (
                metrics["total_time"] / metrics["total_requests"]
            )
            metrics["avg_llm_calls_per_request"] = (
                metrics["llm_calls"] / metrics["total_requests"]
            )
            metrics["pass_rate"] = metrics["passed_count"] / metrics["total_requests"]

        return metrics


class CachingMiddleware(Middleware):
    """Caches evaluation results for identical inputs.

    Provides significant performance improvements when evaluating the
    same output/reference pairs multiple times. Uses an LRU-style cache
    with configurable maximum size.

    Example:
        >>> cache = CachingMiddleware(max_size=200)
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(cache)
        >>>
        >>> # First call - cache miss
        >>> result1 = await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Second call - cache hit (instant)
        >>> result2 = await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Check cache stats
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(self, max_size: int = 100):
        """Initialize caching middleware.

        Args:
            max_size: Maximum number of cached results. When exceeded,
                oldest entries are removed (FIFO policy).
        """
        self.cache: Dict[str, EvaluationResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _get_cache_key(
        self, output: str, reference: Optional[str], context: MiddlewareContext
    ) -> str:
        """Generate cache key from inputs and context."""
        evaluators_list = context.get("evaluators", [])
        evaluators = ",".join(sorted(str(e) for e in evaluators_list))
        metrics_list = context.get("metrics", [])
        metrics = ",".join(sorted(str(m) for m in metrics_list))
        config_key = (
            f"{context.get('model', 'default')}_{context.get('temperature', 0.7)}"
        )

        ref_hash = hash(reference) if reference else 0
        return f"{hash(output)}_{ref_hash}_{evaluators}_{metrics}_{config_key}"

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> EvaluationResult:
        """Check cache before processing."""
        cache_key = self._get_cache_key(output, reference, context)

        # Check cache
        if cache_key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for key: {cache_key}")
            return self.cache[cache_key]

        # Cache miss
        self.misses += 1
        result = cast(EvaluationResult, await next_handler(output, reference))

        # Store in cache
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size, and max_size
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }


class RateLimitingMiddleware(Middleware):
    """Rate limits evaluation requests.

    Prevents excessive API usage by enforcing a maximum number of
    requests per minute. Useful for staying within API quotas and
    preventing accidental DoS of LLM services.

    Example:
        >>> rate_limiter = RateLimitingMiddleware(max_requests_per_minute=30)
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(rate_limiter)
        >>>
        >>> # Will raise error if rate exceeded
        >>> try:
        ...     for i in range(100):
        ...         await evaluate(output, ref, middleware=pipeline)
        ... except RuntimeError as e:
        ...     print(f"Rate limit hit: {e}")
    """

    def __init__(self, max_requests_per_minute: int = 60):
        """Initialize rate limiting.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.requests: List[float] = []

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> EvaluationResult:
        """Check rate limit before processing."""
        now = time.time()

        # Remove old requests (older than 60 seconds)
        self.requests = [t for t in self.requests if now - t < 60]

        # Check rate limit
        if len(self.requests) >= self.max_requests:
            wait_time = 60 - (now - self.requests[0])
            raise RuntimeError(
                f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
            )

        # Add current request
        self.requests.append(now)

        return cast(EvaluationResult, await next_handler(output, reference))


class MiddlewarePipeline:
    """Manages the middleware pipeline.

    Orchestrates multiple middleware components into a single processing
    chain. Middleware are executed in the order they are added.

    Example:
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(LoggingMiddleware())
        >>> pipeline.add(MetricsMiddleware())
        >>> pipeline.add(CachingMiddleware())
        >>>
        >>> # Use with evaluation
        >>> result = await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Get specific middleware
        >>> metrics = pipeline.get_middleware(MetricsMiddleware)
        >>> print(metrics.get_metrics())
    """

    def __init__(self, middleware: Optional[List[Middleware]] = None) -> None:
        """Initialize pipeline with optional middleware list.

        Args:
            middleware: Optional list of middleware to add initially
        """
        self.middleware: List[Middleware] = middleware or []

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add middleware to the pipeline.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for method chaining
        """
        self.middleware.append(middleware)
        return self

    def get_middleware(self, middleware_type: type) -> Optional[Middleware]:
        """Get middleware instance by type.

        Args:
            middleware_type: Class type of middleware to retrieve

        Returns:
            First middleware instance of the given type, or None
        """
        for mw in self.middleware:
            if isinstance(mw, middleware_type):
                return mw
        return None

    async def execute(
        self,
        output: str,
        reference: Optional[str],
        final_handler: Callable[[str, Optional[str]], Any],
        context: Optional[MiddlewareContext] = None,
    ) -> EvaluationResult:
        """Execute the middleware pipeline.

        Args:
            output: Output text to evaluate
            reference: Reference text (may be None)
            final_handler: The actual evaluation function
            context: Shared context between middleware

        Returns:
            EvaluationResult from the pipeline
        """
        if context is None:
            context = {}

        # Build the chain
        async def chain(
            index: int, current_output: str, current_reference: Optional[str]
        ) -> EvaluationResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final handler
                return cast(
                    EvaluationResult,
                    await final_handler(current_output, current_reference),
                )

            # Call current middleware
            current = self.middleware[index]
            return await current.process(
                current_output,
                current_reference,
                lambda o, r: chain(index + 1, o, r),
                context,
            )

        return await chain(0, output, reference)

    async def execute_comparison(
        self,
        output_a: str,
        output_b: str,
        criteria: Optional[str],
        reference: Optional[str],
        final_handler: Callable[
            [str, str, Optional[str], Optional[str]], Any
        ],
        context: Optional[MiddlewareContext] = None,
    ) -> ComparisonResult:
        """Execute middleware pipeline for pairwise comparison.

        This method adapts the pairwise comparison signature to work with
        the existing middleware infrastructure. Middleware can check the
        context for `is_pairwise_comparison=True` to detect and handle
        pairwise operations specially if needed.

        The adapter works by:
        1. Packaging both outputs into context for middleware access
        2. Passing a formatted string to middleware for logging/tracking
        3. Calling the final pairwise comparison handler
        4. Returning the ComparisonResult

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            criteria: Optional comparison criteria
            reference: Optional reference context
            final_handler: The actual comparison function
            context: Shared context between middleware

        Returns:
            ComparisonResult from the pipeline

        Example:
            >>> pipeline = MiddlewarePipeline([
            ...     LoggingMiddleware(),
            ...     MetricsMiddleware()
            ... ])
            >>> result = await pipeline.execute_comparison(
            ...     output_a="First output",
            ...     output_b="Second output",
            ...     criteria="accuracy, clarity",
            ...     reference="Reference text",
            ...     final_handler=compare_impl
            ... )
        """
        if context is None:
            context = {}

        # Mark this as a pairwise comparison for middleware
        context["is_pairwise_comparison"] = True
        context["pairwise_data"] = {
            "output_a": output_a,
            "output_b": output_b,
            "criteria": criteria,
        }

        # Create formatted output for middleware logging
        # This allows existing middleware to work without modification
        formatted_output = (
            f"PAIRWISE COMPARISON:\n"
            f"Output A: {output_a[:100]}...\n"
            f"Output B: {output_b[:100]}..."
        )

        # Build the chain - adapter pattern
        async def chain(
            index: int, current_output: str, current_reference: Optional[str]
        ) -> Any:
            if index >= len(self.middleware):
                # End of middleware chain, call final pairwise handler
                # Use original outputs, not formatted version
                return await final_handler(output_a, output_b, criteria, reference)

            # Call current middleware with formatted output
            current = self.middleware[index]

            # Middleware processes the formatted output but we preserve pairwise data
            result = await current.process(
                current_output,
                current_reference,
                lambda o, r: chain(index + 1, o, r),
                context,
            )

            # For pairwise, middleware returns EvaluationResult but we need ComparisonResult
            # The final_handler will return the correct type
            return result

        # Execute the chain - the formatted output is for middleware visibility only
        result = await chain(0, formatted_output, reference)

        # Return the ComparisonResult from final_handler
        return cast(ComparisonResult, result)


@asynccontextmanager
async def monitor(
    include_logging: bool = True, include_metrics: bool = True, log_level: str = "INFO"
) -> AsyncIterator[Dict[str, Any]]:
    """Context manager for monitoring evaluations.

    Convenient helper for creating a pipeline with logging and metrics
    middleware. Automatically logs final metrics when done.

    Args:
        include_logging: Whether to include logging middleware
        include_metrics: Whether to include metrics middleware
        log_level: Logging level (if logging enabled)

    Yields:
        Dictionary with 'pipeline' and 'metrics' keys

    Example:
        >>> async with monitor() as ctx:
        ...     pipeline = ctx['pipeline']
        ...     for output, ref in test_cases:
        ...         await evaluate(output, ref, middleware=pipeline)
        ...
        ...     # Metrics automatically logged at end
        ...     metrics = ctx['metrics']
        ...     if metrics:
        ...         print(metrics.get_metrics())
    """
    pipeline = MiddlewarePipeline()
    metrics_middleware = None

    if include_logging:
        pipeline.add(LoggingMiddleware(log_level))

    if include_metrics:
        metrics_middleware = MetricsMiddleware()
        pipeline.add(metrics_middleware)

    data = {"pipeline": pipeline, "metrics": metrics_middleware}

    yield data

    # After completion, log final metrics
    if metrics_middleware:
        final_metrics = metrics_middleware.get_metrics()
        logger.info(f"Session metrics: {final_metrics}")
