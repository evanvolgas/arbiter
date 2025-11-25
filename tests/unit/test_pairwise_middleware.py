"""Unit tests for pairwise comparison middleware integration."""

from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter_ai import compare
from arbiter_ai.core.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewarePipeline,
)
from arbiter_ai.core.models import ComparisonResult
from arbiter_ai.core.type_defs import MiddlewareContext
from arbiter_ai.evaluators.pairwise import AspectComparison, PairwiseResponse
from tests.conftest import MockAgentResult


class TestMiddleware(Middleware):
    """Test middleware that tracks calls."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_context: Optional[MiddlewareContext] = None
        self.last_output: Optional[str] = None
        self.last_reference: Optional[str] = None

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> Any:
        """Track the call and pass through."""
        self.call_count += 1
        self.last_context = context
        self.last_output = output
        self.last_reference = reference
        return await next_handler(output, reference)


class TestPairwiseMiddlewareIntegration:
    """Test suite for pairwise comparison middleware integration."""

    @pytest.mark.asyncio
    async def test_middleware_called_for_pairwise(self, mock_llm_client, mock_agent):
        """Test that middleware is called when using compare() with middleware."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A is better",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create middleware pipeline with test middleware
        test_middleware = TestMiddleware()
        pipeline = MiddlewarePipeline([test_middleware])

        # Execute comparison
        result = await compare(
            output_a="First output",
            output_b="Second output",
            criteria="accuracy",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify middleware was called
        assert test_middleware.call_count == 1
        assert result.winner == "output_a"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_pairwise_context_markers(self, mock_llm_client, mock_agent):
        """Test that middleware context is marked for pairwise comparison."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="tie",
            confidence=0.85,
            reasoning="Both outputs are equivalent",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create middleware pipeline with test middleware
        test_middleware = TestMiddleware()
        pipeline = MiddlewarePipeline([test_middleware])

        # Execute comparison
        await compare(
            output_a="Output A",
            output_b="Output B",
            criteria="quality",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify context markers
        assert test_middleware.last_context is not None
        assert test_middleware.last_context.get("is_pairwise_comparison") is True
        assert "pairwise_data" in test_middleware.last_context
        pairwise_data = test_middleware.last_context["pairwise_data"]
        assert pairwise_data["output_a"] == "Output A"
        assert pairwise_data["output_b"] == "Output B"
        assert pairwise_data["criteria"] == "quality"

    @pytest.mark.asyncio
    async def test_formatted_output_for_logging(self, mock_llm_client, mock_agent):
        """Test that middleware receives formatted output for logging."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="output_b",
            confidence=0.75,
            reasoning="Output B is superior",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create middleware pipeline with test middleware
        test_middleware = TestMiddleware()
        pipeline = MiddlewarePipeline([test_middleware])

        # Execute comparison
        await compare(
            output_a="First output text",
            output_b="Second output text",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify formatted output
        assert test_middleware.last_output is not None
        assert "PAIRWISE COMPARISON" in test_middleware.last_output
        assert "Output A:" in test_middleware.last_output
        assert "Output B:" in test_middleware.last_output

    @pytest.mark.asyncio
    async def test_logging_middleware_with_pairwise(
        self, mock_llm_client, mock_agent, caplog
    ):
        """Test that LoggingMiddleware works with pairwise comparison."""
        import logging

        caplog.set_level(logging.INFO)

        # Setup mock response
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.95,
            reasoning="Clear winner",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create pipeline with logging middleware
        pipeline = MiddlewarePipeline([LoggingMiddleware(log_level="INFO")])

        # Execute comparison
        await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify logging occurred
        assert "Starting evaluation" in caplog.text
        assert "PAIRWISE COMPARISON" in caplog.text

    @pytest.mark.asyncio
    async def test_metrics_middleware_with_pairwise(self, mock_llm_client, mock_agent):
        """Test that MetricsMiddleware tracks pairwise comparisons."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A wins",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create pipeline with metrics middleware
        metrics_middleware = MetricsMiddleware()
        pipeline = MiddlewarePipeline([metrics_middleware])

        # Execute multiple comparisons
        for i in range(3):
            await compare(
                output_a=f"Output A {i}",
                output_b=f"Output B {i}",
                llm_client=mock_llm_client,
                middleware=pipeline,
            )

        # Verify metrics
        metrics = metrics_middleware.get_metrics()
        assert metrics["total_requests"] == 3
        assert metrics["total_time"] > 0
        assert "avg_time_per_request" in metrics

    @pytest.mark.asyncio
    async def test_multiple_middleware_with_pairwise(self, mock_llm_client, mock_agent):
        """Test that multiple middleware work together for pairwise."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="tie",
            confidence=0.8,
            reasoning="Equal quality",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create pipeline with multiple middleware
        test_middleware = TestMiddleware()
        metrics_middleware = MetricsMiddleware()
        pipeline = MiddlewarePipeline(
            [
                LoggingMiddleware(),
                test_middleware,
                metrics_middleware,
            ]
        )

        # Execute comparison
        result = await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify all middleware were called
        assert test_middleware.call_count == 1
        assert metrics_middleware.get_metrics()["total_requests"] == 1
        assert result.winner == "tie"

    @pytest.mark.asyncio
    async def test_pairwise_without_middleware(self, mock_llm_client, mock_agent):
        """Test that pairwise comparison works without middleware (backward compatibility)."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="output_b",
            confidence=0.85,
            reasoning="Output B is better",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Execute comparison without middleware
        result = await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
        )

        # Verify result
        assert result.winner == "output_b"
        assert result.confidence == 0.85
        assert result.reasoning == "Output B is better"

    @pytest.mark.asyncio
    async def test_execute_comparison_directly(self, mock_llm_client):
        """Test MiddlewarePipeline.execute_comparison() method directly."""
        # Create test middleware
        test_middleware = TestMiddleware()
        pipeline = MiddlewarePipeline([test_middleware])

        # Mock final handler
        async def mock_final_handler(
            output_a: str,
            output_b: str,
            criteria: Optional[str],
            reference: Optional[str],
        ) -> ComparisonResult:
            return ComparisonResult(
                output_a=output_a,
                output_b=output_b,
                reference=reference,
                criteria=criteria,
                winner="output_a",
                confidence=0.9,
                reasoning="Test reasoning",
                aspect_scores={},
                total_tokens=100,
                processing_time=0.5,
                interactions=[],
            )

        # Execute comparison
        result = await pipeline.execute_comparison(
            output_a="Output A",
            output_b="Output B",
            criteria="test criteria",
            reference="test reference",
            final_handler=mock_final_handler,
        )

        # Verify middleware was called
        assert test_middleware.call_count == 1
        assert test_middleware.last_context is not None
        assert test_middleware.last_context.get("is_pairwise_comparison") is True

        # Verify result
        assert isinstance(result, ComparisonResult)
        assert result.winner == "output_a"
        assert result.output_a == "Output A"
        assert result.output_b == "Output B"

    @pytest.mark.asyncio
    async def test_pairwise_with_reference(self, mock_llm_client, mock_agent):
        """Test pairwise comparison with reference and middleware."""
        # Setup mock response
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.92,
            reasoning="Output A matches reference better",
            aspect_comparisons=[],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create middleware pipeline
        test_middleware = TestMiddleware()
        pipeline = MiddlewarePipeline([test_middleware])

        # Execute comparison with reference
        result = await compare(
            output_a="Output A",
            output_b="Output B",
            reference="Reference text",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify middleware received reference
        assert test_middleware.last_reference == "Reference text"
        assert result.reference == "Reference text"

    @pytest.mark.asyncio
    async def test_pairwise_with_criteria(self, mock_llm_client, mock_agent):
        """Test pairwise comparison with criteria and middleware."""
        # Setup mock response with aspect comparisons
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.88,
            reasoning="Output A scores higher on criteria",
            aspect_comparisons=[
                AspectComparison(
                    aspect="accuracy",
                    output_a_score=0.9,
                    output_b_score=0.8,
                    reasoning="A is more accurate",
                ),
                AspectComparison(
                    aspect="clarity",
                    output_a_score=0.85,
                    output_b_score=0.9,
                    reasoning="B is clearer",
                ),
            ],
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create middleware pipeline
        test_middleware = TestMiddleware()
        pipeline = MiddlewarePipeline([test_middleware])

        # Execute comparison with criteria
        result = await compare(
            output_a="Output A",
            output_b="Output B",
            criteria="accuracy, clarity",
            llm_client=mock_llm_client,
            middleware=pipeline,
        )

        # Verify criteria in context
        assert test_middleware.last_context is not None
        pairwise_data = test_middleware.last_context["pairwise_data"]
        assert pairwise_data["criteria"] == "accuracy, clarity"

        # Verify aspect scores
        assert "accuracy" in result.aspect_scores
        assert "clarity" in result.aspect_scores
        assert result.aspect_scores["accuracy"]["output_a"] == 0.9
        assert result.aspect_scores["clarity"]["output_b"] == 0.9
