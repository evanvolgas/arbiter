"""Performance monitoring and observability for Arbiter evaluation operations.

This module provides comprehensive monitoring capabilities for tracking
the performance and behavior of evaluation operations. It collects
detailed metrics about LLM calls, evaluator execution, and overall
processing time.

## Key Features:

- **Detailed Metrics**: Track timing, token usage, and call counts
- **Performance Analysis**: Identify bottlenecks and optimization opportunities
- **Error Tracking**: Capture and categorize failures for debugging
- **Integration**: Optional Logfire integration for production monitoring
- **Context Managers**: Easy instrumentation of operations

## Usage:

    >>> # Basic monitoring
    >>> monitor = PerformanceMonitor()
    >>> async with monitor.track_operation("evaluate") as metrics:
    ...     result = await evaluate(output, reference)
    >>> print(metrics.to_dict())

    >>> # Global monitoring
    >>> from arbiter import get_global_monitor
    >>> monitor = get_global_monitor()
    >>>
    >>> # Context manager for specific operations
    >>> async with monitor_context("batch_evaluation"):
    ...     await process_batch(outputs, references)

## Metrics Collected:

- **Timing**: Total duration, LLM time, evaluator time
- **LLM Usage**: Call count, tokens used, tokens per second
- **Evaluators**: Which evaluators ran, how long they took
- **Scores**: Score distributions and averages
- **Errors**: Detailed error information for debugging

## Integration with Logfire:

Set the LOGFIRE_TOKEN environment variable to enable production monitoring:

    export LOGFIRE_TOKEN=your_token_here

This will send detailed traces and metrics to Logfire for analysis.
"""

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import logfire

# Configure logfire if token is available
if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

__all__ = ["PerformanceMetrics", "PerformanceMonitor", "get_global_monitor", "monitor"]


@dataclass
class PerformanceMetrics:
    """Comprehensive metrics for a single evaluation operation.

    This dataclass collects all performance-related data during the
    execution of an evaluate() call. It tracks timing, resource usage,
    and quality metrics to provide full observability.

    Example:
        >>> metrics = PerformanceMetrics(start_time=time.time())
        >>> # ... perform operations ...
        >>> metrics.llm_calls += 1
        >>> metrics.tokens_used += 150
        >>> metrics.end_time = time.time()
        >>> metrics.finalize()
        >>> print(f"Total time: {metrics.total_duration:.2f}s")

    Attributes:
        start_time: Unix timestamp when operation began
        end_time: Unix timestamp when operation completed
        total_duration: Total seconds elapsed
        llm_calls: Number of LLM API calls made
        llm_time: Total seconds spent in LLM calls
        tokens_used: Total tokens consumed
        tokens_per_second: Token generation rate
        evaluator_calls: Number of evaluator executions
        evaluator_time: Total seconds in evaluator execution
        evaluators_used: List of evaluator names called
        scores_computed: Number of scores calculated
        average_score: Average score across metrics
        errors: List of error details
    """

    # Timing metrics
    start_time: float
    end_time: float = 0.0
    total_duration: float = 0.0

    # LLM metrics
    llm_calls: int = 0
    llm_time: float = 0.0
    tokens_used: int = 0
    tokens_per_second: float = 0.0

    # Evaluator metrics
    evaluator_calls: int = 0
    evaluator_time: float = 0.0
    evaluators_used: List[str] = field(default_factory=list)

    # Score metrics
    scores_computed: int = 0
    average_score: float = 0.0
    score_distribution: Dict[str, float] = field(default_factory=dict)

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def finalize(self) -> None:
        """Calculate derived metrics after operation completes."""
        self.total_duration = self.end_time - self.start_time

        if self.tokens_used > 0 and self.llm_time > 0:
            self.tokens_per_second = self.tokens_used / self.llm_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "timing": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration": round(self.total_duration, 3),
                "llm_time": round(self.llm_time, 3),
                "evaluator_time": round(self.evaluator_time, 3),
            },
            "llm": {
                "calls": self.llm_calls,
                "tokens_used": self.tokens_used,
                "tokens_per_second": round(self.tokens_per_second, 1),
            },
            "evaluators": {
                "calls": self.evaluator_calls,
                "evaluators_used": list(set(self.evaluators_used)),
                "avg_time_per_call": (
                    round(self.evaluator_time / self.evaluator_calls, 3)
                    if self.evaluator_calls > 0
                    else 0
                ),
            },
            "scores": {
                "count": self.scores_computed,
                "average": round(self.average_score, 3),
                "distribution": self.score_distribution,
            },
            "errors": {"count": len(self.errors), "details": self.errors},
        }


class PerformanceMonitor:
    """Monitor for tracking evaluation performance.

    Provides context managers and tracking methods for monitoring
    evaluation operations.

    Example:
        >>> monitor = PerformanceMonitor()
        >>>
        >>> async with monitor.track_operation("evaluate") as metrics:
        ...     result = await evaluate(output, reference)
        ...     metrics.scores_computed = len(result.scores)
        >>>
        >>> print(monitor.get_summary())
    """

    def __init__(self) -> None:
        """Initialize monitor with empty metrics."""
        self.operations: List[PerformanceMetrics] = []
        self._current_metrics: Optional[PerformanceMetrics] = None

    @asynccontextmanager
    async def track_operation(
        self, operation_name: str
    ) -> AsyncIterator[PerformanceMetrics]:
        """Track a single operation.

        Args:
            operation_name: Name of the operation being tracked

        Yields:
            PerformanceMetrics object to populate during operation

        Example:
            >>> async with monitor.track_operation("evaluate") as metrics:
            ...     result = await evaluate(output, reference)
            ...     metrics.llm_calls = 1
            ...     metrics.tokens_used = result.total_tokens
        """
        metrics = PerformanceMetrics(start_time=time.time())
        self._current_metrics = metrics

        try:
            yield metrics
        finally:
            metrics.end_time = time.time()
            metrics.finalize()
            self.operations.append(metrics)
            self._current_metrics = None

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the currently active metrics object."""
        return self._current_metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all operations.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.operations:
            return {"total_operations": 0}

        total_time = sum(m.total_duration for m in self.operations)
        total_llm_calls = sum(m.llm_calls for m in self.operations)
        total_tokens = sum(m.tokens_used for m in self.operations)
        total_errors = sum(len(m.errors) for m in self.operations)

        return {
            "total_operations": len(self.operations),
            "total_time": round(total_time, 3),
            "total_llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "avg_time_per_operation": round(total_time / len(self.operations), 3),
            "avg_tokens_per_operation": (
                total_tokens // len(self.operations) if self.operations else 0
            ),
        }

    def reset(self) -> None:
        """Reset all tracked operations."""
        self.operations.clear()
        self._current_metrics = None


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


@asynccontextmanager
async def monitor(
    operation_name: str = "evaluation",
) -> AsyncIterator[PerformanceMetrics]:
    """Convenient context manager for monitoring operations.

    Args:
        operation_name: Name of the operation to track

    Yields:
        PerformanceMetrics object

    Example:
        >>> async with monitor("batch_evaluate") as metrics:
        ...     for output, ref in pairs:
        ...         result = await evaluate(output, ref)
        ...         metrics.scores_computed += len(result.scores)
    """
    global_monitor = get_global_monitor()
    async with global_monitor.track_operation(operation_name) as metrics:
        yield metrics
