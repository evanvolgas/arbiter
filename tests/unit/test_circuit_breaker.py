"""Tests for circuit breaker pattern.

Tests cover:
- State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Failure threshold triggering
- Timeout and recovery logic
- Half-open state testing
- Manual reset
- Statistics tracking
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from arbiter.core.circuit_breaker import CircuitBreaker, CircuitState
from arbiter.core.exceptions import CircuitBreakerOpenError


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker with low threshold for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        timeout=1.0,  # 1 second for fast tests
        half_open_max_calls=1,
    )


@pytest.mark.asyncio
async def test_circuit_breaker_starts_closed(circuit_breaker):
    """Test that circuit breaker starts in CLOSED state."""
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.is_closed
    assert not circuit_breaker.is_open
    assert not circuit_breaker.is_half_open


@pytest.mark.asyncio
async def test_successful_call_passes_through(circuit_breaker):
    """Test that successful calls pass through circuit breaker."""

    async def successful_operation():
        return "success"

    result = await circuit_breaker.call(successful_operation)

    assert result == "success"
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures(circuit_breaker):
    """Test that circuit opens after reaching failure threshold."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Fail 3 times to reach threshold
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    # Circuit should now be open
    assert circuit_breaker.state == CircuitState.OPEN
    assert circuit_breaker.is_open
    assert circuit_breaker.failure_count == 3


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_requests_when_open(circuit_breaker):
    """Test that open circuit blocks all requests."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    assert circuit_breaker.is_open

    # Now requests should be blocked immediately
    async def operation_that_should_not_run():
        pytest.fail("This should not be called when circuit is open")
        return "unreachable"

    with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker is open"):
        await circuit_breaker.call(operation_that_should_not_run)


@pytest.mark.asyncio
async def test_circuit_transitions_to_half_open_after_timeout(circuit_breaker):
    """Test that circuit transitions to HALF_OPEN after timeout."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    assert circuit_breaker.is_open

    # Wait for timeout to expire
    await asyncio.sleep(1.1)

    # Next call should transition to half-open
    async def test_operation():
        return "testing recovery"

    result = await circuit_breaker.call(test_operation)

    # After successful call in half-open, circuit should close
    assert result == "testing recovery"
    assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_success_closes_circuit(circuit_breaker):
    """Test that successful call in HALF_OPEN closes the circuit."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Successful call in half-open should close circuit
    async def successful_operation():
        return "recovered"

    result = await circuit_breaker.call(successful_operation)

    assert result == "recovered"
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit(circuit_breaker):
    """Test that failure in HALF_OPEN reopens the circuit."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Failing call in half-open should reopen circuit
    with pytest.raises(ValueError):
        await circuit_breaker.call(failing_operation)

    assert circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_half_open_limits_test_calls(circuit_breaker):
    """Test that HALF_OPEN limits number of test calls."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    async def success_operation():
        return "success"

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # First call is allowed (transitions to half-open), let it succeed partially
    # by not raising an exception, but circuit should still be in half-open
    # Actually, we need to test the max_calls limit while still in HALF_OPEN
    # So let's start the first call but not complete it yet

    # Transition to HALF_OPEN by checking state after timeout
    assert circuit_breaker.state == CircuitState.OPEN
    # Manually transition to half-open to test the limit
    circuit_breaker._transition_to_half_open()

    # First call succeeds and increments counter
    result = await circuit_breaker.call(success_operation)
    assert result == "success"

    # Circuit should close after successful test
    # So let's open it again and test the half-open limit differently
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    await asyncio.sleep(1.1)
    circuit_breaker._transition_to_half_open()

    # Increment the half_open_calls manually to test the limit
    circuit_breaker.half_open_calls = circuit_breaker.half_open_max_calls

    # Now the next call should be blocked
    with pytest.raises(
        CircuitBreakerOpenError, match="half-open and max test calls reached"
    ):
        await circuit_breaker.call(success_operation)


@pytest.mark.asyncio
async def test_manual_reset(circuit_breaker):
    """Test manual reset of circuit breaker."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    assert circuit_breaker.is_open

    # Manual reset
    circuit_breaker.reset()

    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.success_count == 0
    assert circuit_breaker.last_failure_time is None


@pytest.mark.asyncio
async def test_success_count_tracking(circuit_breaker):
    """Test that successful calls are tracked."""

    async def successful_operation():
        return "success"

    # Make 5 successful calls
    for _ in range(5):
        await circuit_breaker.call(successful_operation)

    assert circuit_breaker.success_count == 5
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_get_stats(circuit_breaker):
    """Test circuit breaker statistics."""

    async def failing_operation():
        raise ValueError("Simulated failure")

    # Initial stats
    stats = circuit_breaker.get_stats()
    assert stats["state"] == "closed"
    assert stats["failure_count"] == 0
    assert stats["success_count"] == 0
    assert stats["last_failure_time"] is None

    # Fail to open circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

    # Stats after opening
    stats = circuit_breaker.get_stats()
    assert stats["state"] == "open"
    assert stats["failure_count"] == 3
    assert stats["last_failure_time"] is not None
    assert stats["time_until_half_open"] > 0


@pytest.mark.asyncio
async def test_circuit_breaker_with_arguments():
    """Test circuit breaker with function arguments."""

    breaker = CircuitBreaker(failure_threshold=2)

    async def operation_with_args(x: int, y: int, multiplier: int = 1) -> int:
        return (x + y) * multiplier

    # Test with positional and keyword arguments
    result = await breaker.call(operation_with_args, 5, 3, multiplier=2)
    assert result == 16


@pytest.mark.asyncio
async def test_circuit_breaker_preserves_exception_type():
    """Test that circuit breaker preserves original exception types."""

    breaker = CircuitBreaker(failure_threshold=1)

    class CustomError(Exception):
        pass

    async def failing_with_custom_error():
        raise CustomError("Custom error message")

    with pytest.raises(CustomError, match="Custom error message"):
        await breaker.call(failing_with_custom_error)


@pytest.mark.asyncio
async def test_concurrent_calls_with_circuit_breaker():
    """Test circuit breaker behavior with concurrent calls."""

    breaker = CircuitBreaker(failure_threshold=3)
    call_count = 0

    async def sometimes_failing_operation():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return f"success {call_count}"
        raise ValueError("Failed")

    # Run concurrent operations
    tasks = [breaker.call(sometimes_failing_operation) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # First 2 should succeed, next 3 should fail
    assert results[0] == "success 1"
    assert results[1] == "success 2"
    assert isinstance(results[2], ValueError)
    assert isinstance(results[3], ValueError)
    assert isinstance(results[4], ValueError)


@pytest.mark.asyncio
async def test_integration_with_llm_client_pattern():
    """Test circuit breaker integration pattern similar to LLMClient usage."""

    breaker = CircuitBreaker(failure_threshold=2, timeout=0.5)

    # Mock LLM client behavior
    mock_api_call = AsyncMock()

    # Simulate 2 failures, then success after timeout
    mock_api_call.side_effect = [
        ValueError("API Error"),
        ValueError("API Error"),
        "Success after recovery",
    ]

    # First two calls fail and open circuit
    with pytest.raises(ValueError):
        await breaker.call(mock_api_call)

    with pytest.raises(ValueError):
        await breaker.call(mock_api_call)

    assert breaker.is_open

    # Immediate retry is blocked
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(mock_api_call)

    # Wait for timeout
    await asyncio.sleep(0.6)

    # Now succeeds and closes circuit
    result = await breaker.call(mock_api_call)
    assert result == "Success after recovery"
    assert breaker.is_closed
