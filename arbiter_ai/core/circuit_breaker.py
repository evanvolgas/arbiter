"""Circuit breaker pattern for LLM provider resilience.

The circuit breaker prevents cascading failures when LLM providers have
outages or degraded performance. It monitors failure rates and temporarily
blocks requests when failures exceed a threshold.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failures exceeded threshold, requests are blocked
- HALF_OPEN: Testing if provider has recovered

Example:
    >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
    >>> result = await breaker.call(llm_client.generate, prompt="Hello")
"""

import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar

from .exceptions import CircuitBreakerOpenError

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    The circuit breaker monitors failure rates and blocks requests when
    failures exceed a threshold. After a timeout period, it enters a
    half-open state to test if the service has recovered.

    Thread Safety:
        This implementation is suitable for asyncio applications where
        operations are serialized by the event loop. The circuit breaker
        uses simple attribute updates that are atomic at the Python level
        due to the GIL (Global Interpreter Lock).

        Under high concurrent load with multiple tasks calling simultaneously,
        the half_open_max_calls limit may be slightly exceeded (e.g., 2-3 calls
        instead of 1) due to race conditions in the check-then-increment pattern.
        This is acceptable for most use cases and keeps the implementation simple.

        For applications requiring strict concurrency guarantees, consider adding
        asyncio.Lock protection around state transitions.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds circuit stays open before entering half-open
        half_open_max_calls: Max calls allowed in half-open state
        state: Current circuit state
        failure_count: Number of consecutive failures
        last_failure_time: Timestamp of last failure
        half_open_calls: Number of calls made in half-open state

    Example:
        >>> # Basic usage
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
        >>> result = await breaker.call(some_async_function, arg1, arg2)
        >>>
        >>> # Check circuit state
        >>> if breaker.is_open:
        ...     print("Circuit is open, requests are blocked")
        >>> stats = breaker.get_stats()
        >>> print(f"Failures: {stats['failure_count']}")
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before entering half-open state
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by func
        """
        # Check if circuit should transition to half-open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open. Last failure: "
                    f"{time.time() - (self.last_failure_time or 0):.1f}s ago. "
                    f"Retry in {self._time_until_half_open():.1f}s."
                )

        # In half-open state, limit number of test calls
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    "Circuit breaker is half-open and max test calls reached"
                )
            self.half_open_calls += 1

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout

    def _time_until_half_open(self) -> float:
        """Calculate seconds until half-open state."""
        if self.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.timeout - elapsed)

    def _transition_to_half_open(self) -> None:
        """Transition circuit from open to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            # Success in half-open state, close the circuit
            self.state = CircuitState.CLOSED
            self.half_open_calls = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state, reopen the circuit
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
        elif self.failure_count >= self.failure_threshold:
            # Too many failures, open the circuit
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with state, failure count, and timing info
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "time_until_half_open": (
                self._time_until_half_open() if self.state == CircuitState.OPEN else 0.0
            ),
        }
