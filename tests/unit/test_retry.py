"""Unit tests for retry.py."""

import asyncio

import pytest

from arbiter_ai.core.exceptions import ModelProviderError, TimeoutError
from arbiter_ai.core.retry import (
    RETRY_PERSISTENT,
    RETRY_QUICK,
    RETRY_STANDARD,
    RetryConfig,
    with_retry,
)


class TestRetryConfig:
    """Test RetryConfig."""

    def test_initialization_defaults(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.delay == 1.0
        assert config.backoff == 2.0

    def test_initialization_custom(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(max_attempts=5, delay=2.0, backoff=1.5)

        assert config.max_attempts == 5
        assert config.delay == 2.0
        assert config.backoff == 1.5

    def test_validation_max_attempts_too_low(self):
        """Test that max_attempts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RetryConfig(max_attempts=0)

        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RetryConfig(max_attempts=-1)

    def test_validation_delay_zero_or_negative(self):
        """Test that delay <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="delay must be positive"):
            RetryConfig(delay=0)

        with pytest.raises(ValueError, match="delay must be positive"):
            RetryConfig(delay=-1.0)

    def test_validation_backoff_too_low(self):
        """Test that backoff < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="backoff must be at least 1.0"):
            RetryConfig(backoff=0.9)

        with pytest.raises(ValueError, match="backoff must be at least 1.0"):
            RetryConfig(backoff=0.0)


class TestWithRetry:
    """Test with_retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @with_retry()
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_default_config(self):
        """Test retry with default configuration."""
        call_count = 0

        @with_retry()
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ModelProviderError("API error")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_custom_config(self):
        """Test retry with custom configuration."""
        config = RetryConfig(max_attempts=5, delay=0.01, backoff=1.1)
        call_count = 0

        @with_retry(config)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ModelProviderError("API error")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_retry_timeout_error(self):
        """Test retry with TimeoutError."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=2, delay=0.01))
        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return "success"

        result = await timeout_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_asyncio_timeout_error(self):
        """Test retry with asyncio.TimeoutError."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=2, delay=0.01))
        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError("Asyncio timeout")
            return "success"

        result = await timeout_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_connection_error(self):
        """Test retry with ConnectionError."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=2, delay=0.01))
        async def connection_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "success"

        result = await connection_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test that error is raised after max attempts."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, delay=0.01))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ModelProviderError("Always fails")

        with pytest.raises(ModelProviderError, match="Always fails"):
            await always_fails()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raised_immediately(self):
        """Test that non-retryable errors are raised immediately."""
        call_count = 0

        @with_retry()
        async def validation_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Validation error")

        with pytest.raises(ValueError, match="Validation error"):
            await validation_error()

        # Should only be called once (no retry)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test that delay increases exponentially."""
        config = RetryConfig(max_attempts=3, delay=0.1, backoff=2.0)
        delays = []

        @with_retry(config)
        async def track_delays():
            if len(delays) < 2:
                delays.append(asyncio.get_event_loop().time())
                raise ModelProviderError("Error")
            return "success"

        await track_delays()

        # Calculate actual delays between attempts
        if len(delays) >= 2:
            actual_delay = delays[1] - delays[0]
            # First retry should wait ~0.1 seconds
            assert 0.08 < actual_delay < 0.15

    @pytest.mark.asyncio
    async def test_function_args_preserved(self):
        """Test that function arguments are preserved."""

        @with_retry()
        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await func_with_args("x", "y", c="z")

        assert result == "x-y-z"

    @pytest.mark.asyncio
    async def test_function_kwargs_preserved(self):
        """Test that function kwargs are preserved."""

        @with_retry()
        async def func_with_kwargs(**kwargs):
            return kwargs

        result = await func_with_kwargs(foo="bar", baz="qux")

        assert result == {"foo": "bar", "baz": "qux"}


class TestPresetConfigurations:
    """Test preset retry configurations."""

    def test_retry_quick(self):
        """Test RETRY_QUICK preset."""
        assert RETRY_QUICK.max_attempts == 2
        assert RETRY_QUICK.delay == 0.5
        assert RETRY_QUICK.backoff == 1.5

    def test_retry_standard(self):
        """Test RETRY_STANDARD preset."""
        assert RETRY_STANDARD.max_attempts == 3
        assert RETRY_STANDARD.delay == 1.0
        assert RETRY_STANDARD.backoff == 2.0

    def test_retry_persistent(self):
        """Test RETRY_PERSISTENT preset."""
        assert RETRY_PERSISTENT.max_attempts == 5
        assert RETRY_PERSISTENT.delay == 1.0
        assert RETRY_PERSISTENT.backoff == 2.0

    @pytest.mark.asyncio
    async def test_using_retry_quick_preset(self):
        """Test using RETRY_QUICK in decorator."""
        call_count = 0

        @with_retry(RETRY_QUICK)
        async def quick_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ModelProviderError("Error")
            return "success"

        result = await quick_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_using_retry_persistent_preset(self):
        """Test using RETRY_PERSISTENT in decorator."""
        call_count = 0

        @with_retry(RETRY_PERSISTENT)
        async def persistent_func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ModelProviderError("Error")
            return "success"

        result = await persistent_func()

        assert result == "success"
        assert call_count == 4
