"""Unit tests for middleware.py."""

import logging
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Callable, Optional

import pytest

from arbiter.core.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewarePipeline,
    RateLimitingMiddleware,
    monitor,
)
from arbiter.core.models import ComparisonResult, EvaluationResult, Score
from arbiter.core.type_defs import MiddlewareContext


@pytest.fixture
def eval_result():
    """Create a sample EvaluationResult."""
    return EvaluationResult(
        output="Test output text",
        reference="Test reference text",
        overall_score=0.85,
        passed=True,
        threshold=0.7,
        scores=[Score(name="semantic", value=0.85, confidence=0.9, explanation="Good")],
        metrics=[],
        total_tokens=100,
        processing_time=1.5,
        partial=False,
        errors={},
        interactions=[],
    )


@pytest.fixture
def comparison_result():
    """Create a sample ComparisonResult."""
    return ComparisonResult(
        output_a="Output A",
        output_b="Output B",
        winner="output_a",
        confidence=0.9,
        reasoning="A is better",
        aspect_scores={},
        total_tokens=100,
        processing_time=1.0,
        interactions=[],
    )


class TestLoggingMiddleware:
    """Test LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logging_with_reference(self, caplog, eval_result):
        """Test logging when reference is provided."""
        caplog.set_level(logging.INFO)

        middleware = LoggingMiddleware(log_level="INFO")

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        result = await middleware.process(
            output="Test output",
            reference="Test reference",
            next_handler=mock_handler,
            context={},
        )

        assert result == eval_result
        assert "Test output" in caplog.text
        assert "Test reference" in caplog.text

    @pytest.mark.asyncio
    async def test_logging_comparison_result(self, caplog, comparison_result):
        """Test logging for ComparisonResult (not EvaluationResult)."""
        caplog.set_level(logging.INFO)

        middleware = LoggingMiddleware(log_level="INFO")

        async def mock_handler(output: str, reference: Optional[str]) -> ComparisonResult:
            return comparison_result

        result = await middleware.process(
            output="Test output",
            reference=None,
            next_handler=mock_handler,
            context={},
        )

        assert result == comparison_result
        assert "winner=output_a" in caplog.text
        assert "confidence=0.900" in caplog.text

    @pytest.mark.asyncio
    async def test_logging_error_handling(self, caplog):
        """Test logging when evaluation fails."""
        caplog.set_level(logging.ERROR)

        middleware = LoggingMiddleware(log_level="INFO")

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await middleware.process(
                output="Test output",
                reference=None,
                next_handler=mock_handler,
                context={},
            )

        assert "Evaluation failed" in caplog.text
        assert "ValueError" in caplog.text


class TestMetricsMiddleware:
    """Test MetricsMiddleware."""

    @pytest.mark.asyncio
    async def test_metrics_tracks_passed_count(self, eval_result):
        """Test that passed_count is tracked for EvaluationResult."""
        middleware = MetricsMiddleware()

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        # Process 3 successful evaluations
        for _ in range(3):
            await middleware.process(
                output="Test",
                reference=None,
                next_handler=mock_handler,
                context={},
            )

        metrics = middleware.get_metrics()
        assert metrics["passed_count"] == 3
        assert metrics["total_requests"] == 3

    @pytest.mark.asyncio
    async def test_metrics_error_tracking(self, eval_result):
        """Test that errors are tracked."""
        middleware = MetricsMiddleware()

        # First successful call
        async def success_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        await middleware.process(
            output="Test",
            reference=None,
            next_handler=success_handler,
            context={},
        )

        # Second call fails
        async def error_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await middleware.process(
                output="Test",
                reference=None,
                next_handler=error_handler,
                context={},
            )

        metrics = middleware.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["errors"] == 1


class TestCachingMiddleware:
    """Test CachingMiddleware."""

    def test_initialization(self):
        """Test CachingMiddleware initialization."""
        cache = CachingMiddleware(max_size=50)
        assert cache.max_size == 50
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.cache) == 0

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, eval_result):
        """Test cache key generation with different inputs."""
        cache = CachingMiddleware()

        context1: MiddlewareContext = {
            "evaluators": ["semantic"],
            "metrics": [],
            "model": "gpt-4",
            "temperature": 0.7,
        }

        context2: MiddlewareContext = {
            "evaluators": ["semantic", "custom"],
            "metrics": [],
            "model": "gpt-4",
            "temperature": 0.7,
        }

        key1 = cache._get_cache_key("output1", "ref1", context1)
        key2 = cache._get_cache_key("output1", "ref1", context2)
        key3 = cache._get_cache_key("output2", "ref1", context1)

        # Different contexts should produce different keys
        assert key1 != key2
        # Different outputs should produce different keys
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_hit(self, eval_result):
        """Test cache hit behavior."""
        cache = CachingMiddleware()

        call_count = 0

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            nonlocal call_count
            call_count += 1
            return eval_result

        context: MiddlewareContext = {"evaluators": ["semantic"]}

        # First call - cache miss
        result1 = await cache.process(
            output="Test",
            reference="Ref",
            next_handler=mock_handler,
            context=context,
        )

        # Second call - cache hit
        result2 = await cache.process(
            output="Test",
            reference="Ref",
            next_handler=mock_handler,
            context=context,
        )

        # Handler should only be called once
        assert call_count == 1
        assert cache.hits == 1
        assert cache.misses == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_eviction(self, eval_result):
        """Test cache eviction when max_size is exceeded."""
        cache = CachingMiddleware(max_size=2)

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        context: MiddlewareContext = {}

        # Add 3 entries (exceeds max_size of 2)
        await cache.process("output1", "ref1", mock_handler, context)
        await cache.process("output2", "ref2", mock_handler, context)
        await cache.process("output3", "ref3", mock_handler, context)

        # Cache should only have 2 entries
        assert len(cache.cache) == 2

    def test_get_stats(self):
        """Test cache statistics."""
        cache = CachingMiddleware()
        cache.hits = 10
        cache.misses = 5

        stats = cache.get_stats()
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 10 / 15
        assert stats["size"] == 0
        assert stats["max_size"] == 100


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware."""

    def test_initialization(self):
        """Test RateLimitingMiddleware initialization."""
        limiter = RateLimitingMiddleware(max_requests_per_minute=30)
        assert limiter.max_requests == 30
        assert len(limiter.requests) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_allows_requests(self, eval_result):
        """Test that rate limiter allows requests under limit."""
        limiter = RateLimitingMiddleware(max_requests_per_minute=10)

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        # Should allow 10 requests
        for i in range(10):
            result = await limiter.process(
                output=f"Test {i}",
                reference=None,
                next_handler=mock_handler,
                context={},
            )
            assert result == eval_result

    @pytest.mark.asyncio
    async def test_rate_limiting_blocks_excess_requests(self, eval_result):
        """Test that rate limiter blocks requests over limit."""
        limiter = RateLimitingMiddleware(max_requests_per_minute=5)

        async def mock_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        # First 5 should succeed
        for i in range(5):
            await limiter.process(
                output=f"Test {i}",
                reference=None,
                next_handler=mock_handler,
                context={},
            )

        # 6th should fail
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await limiter.process(
                output="Test 6",
                reference=None,
                next_handler=mock_handler,
                context={},
            )


class TestMiddlewarePipeline:
    """Test MiddlewarePipeline."""

    def test_add_middleware(self):
        """Test adding middleware to pipeline."""
        pipeline = MiddlewarePipeline()
        logging_mw = LoggingMiddleware()
        metrics_mw = MetricsMiddleware()

        result = pipeline.add(logging_mw).add(metrics_mw)

        assert len(pipeline.middleware) == 2
        assert pipeline.middleware[0] == logging_mw
        assert pipeline.middleware[1] == metrics_mw
        assert result == pipeline  # Method chaining

    def test_get_middleware(self):
        """Test getting middleware by type."""
        pipeline = MiddlewarePipeline()
        logging_mw = LoggingMiddleware()
        metrics_mw = MetricsMiddleware()

        pipeline.add(logging_mw).add(metrics_mw)

        # Get existing middleware
        found_logging = pipeline.get_middleware(LoggingMiddleware)
        found_metrics = pipeline.get_middleware(MetricsMiddleware)

        assert found_logging == logging_mw
        assert found_metrics == metrics_mw

        # Get non-existent middleware
        found_cache = pipeline.get_middleware(CachingMiddleware)
        assert found_cache is None

    @pytest.mark.asyncio
    async def test_execute_pipeline(self, eval_result):
        """Test executing the middleware pipeline."""
        pipeline = MiddlewarePipeline()
        metrics_mw = MetricsMiddleware()
        pipeline.add(metrics_mw)

        async def final_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        result = await pipeline.execute(
            output="Test output",
            reference="Test reference",
            final_handler=final_handler,
            context=None,
        )

        assert result == eval_result
        assert metrics_mw.get_metrics()["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_execute_with_context(self, eval_result):
        """Test executing pipeline with custom context."""
        pipeline = MiddlewarePipeline()

        received_context = None

        class ContextCheckMiddleware(LoggingMiddleware):
            async def process(self, output, reference, next_handler, context):
                nonlocal received_context
                received_context = context
                return await next_handler(output, reference)

        pipeline.add(ContextCheckMiddleware())

        async def final_handler(output: str, reference: Optional[str]) -> EvaluationResult:
            return eval_result

        custom_context: MiddlewareContext = {"evaluators": ["semantic"], "custom_key": "custom_value"}

        await pipeline.execute(
            output="Test",
            reference=None,
            final_handler=final_handler,
            context=custom_context,
        )

        assert received_context == custom_context
        assert received_context["custom_key"] == "custom_value"


class TestMonitorContextManager:
    """Test monitor() context manager."""

    @pytest.mark.asyncio
    async def test_monitor_basic(self):
        """Test basic monitor context manager."""
        async with monitor() as ctx:
            assert "pipeline" in ctx
            assert "metrics" in ctx
            assert isinstance(ctx["pipeline"], MiddlewarePipeline)
            assert isinstance(ctx["metrics"], MetricsMiddleware)

    @pytest.mark.asyncio
    async def test_monitor_logging_only(self):
        """Test monitor with only logging."""
        async with monitor(include_metrics=False) as ctx:
            assert "pipeline" in ctx
            assert ctx["metrics"] is None

            # Pipeline should have logging middleware
            logging_mw = ctx["pipeline"].get_middleware(LoggingMiddleware)
            assert logging_mw is not None

    @pytest.mark.asyncio
    async def test_monitor_metrics_only(self):
        """Test monitor with only metrics."""
        async with monitor(include_logging=False) as ctx:
            assert "pipeline" in ctx
            assert ctx["metrics"] is not None

            # Pipeline should have metrics but not logging
            logging_mw = ctx["pipeline"].get_middleware(LoggingMiddleware)
            assert logging_mw is None

            metrics_mw = ctx["pipeline"].get_middleware(MetricsMiddleware)
            assert metrics_mw is not None

    @pytest.mark.asyncio
    async def test_monitor_custom_log_level(self):
        """Test monitor with custom log level."""
        async with monitor(log_level="DEBUG") as ctx:
            logging_mw = ctx["pipeline"].get_middleware(LoggingMiddleware)
            assert logging_mw is not None
            assert logging_mw.log_level == logging.DEBUG

    @pytest.mark.asyncio
    async def test_monitor_final_metrics_logged(self, caplog):
        """Test that final metrics are logged when monitor exits."""
        caplog.set_level(logging.INFO)

        async with monitor() as ctx:
            metrics_mw = ctx["metrics"]
            # Manually update metrics
            metrics_mw.metrics["total_requests"] = 5
            metrics_mw.metrics["total_time"] = 10.0

        # After exit, metrics should be logged
        assert "Session metrics" in caplog.text
