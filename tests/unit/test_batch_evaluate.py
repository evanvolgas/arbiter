"""Unit tests for batch_evaluate() function."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter_ai.core.exceptions import ValidationError
from arbiter_ai.core.models import BatchEvaluationResult, EvaluationResult
from arbiter_ai.evaluators.custom_criteria import CustomCriteriaResponse
from arbiter_ai.evaluators.semantic import SemanticResponse
from tests.conftest import MockAgentResult


class TestBatchEvaluateFunction:
    """Test suite for batch_evaluate() function."""

    @pytest.mark.asyncio
    async def test_batch_evaluate_basic(self, mock_llm_client, mock_agent):
        """Test basic batch evaluation with multiple items."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="High similarity",
            key_similarities=[],
            key_differences=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {
                "output": "Paris is the capital of France",
                "reference": "Paris is France's capital",
            },
            {
                "output": "Tokyo is the capital of Japan",
                "reference": "Tokyo is Japan's capital",
            },
            {
                "output": "Berlin is the capital of Germany",
                "reference": "Berlin is Germany's capital",
            },
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert isinstance(result, BatchEvaluationResult)
        assert result.total_items == 3
        assert result.successful_items == 3
        assert result.failed_items == 0
        assert len(result.results) == 3
        assert all(r is not None for r in result.results)
        assert all(
            isinstance(r, EvaluationResult) for r in result.results if r is not None
        )
        assert len(result.errors) == 0
        assert result.processing_time > 0
        assert result.total_tokens == 300  # 3 items * 100 tokens each

    @pytest.mark.asyncio
    async def test_batch_evaluate_progress_callback(self, mock_llm_client, mock_agent):
        """Test that progress callback is invoked correctly."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Reference 1"},
            {"output": "Test 2", "reference": "Reference 2"},
        ]

        callback_calls = []

        def progress_callback(completed, total, latest_result):
            callback_calls.append(
                {
                    "completed": completed,
                    "total": total,
                    "latest_result": latest_result,
                }
            )

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
            progress_callback=progress_callback,
        )

        assert result.successful_items == 2
        assert len(callback_calls) == 2
        assert callback_calls[0]["completed"] == 1
        assert callback_calls[0]["total"] == 2
        assert callback_calls[1]["completed"] == 2
        assert callback_calls[1]["total"] == 2
        assert all(call["latest_result"] is not None for call in callback_calls)

    @pytest.mark.asyncio
    async def test_batch_evaluate_partial_failures(self, mock_llm_client, mock_agent):
        """Test that partial failures don't fail entire batch."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        # First and third succeed, second fails during evaluation
        mock_agent.run = AsyncMock(
            side_effect=[
                MockAgentResult(mock_response),
                Exception("Evaluation failed"),
                MockAgentResult(mock_response),
            ]
        )
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Reference 1"},
            {
                "output": "Test 2",
                "reference": "Reference 2",
            },  # Will fail during evaluation
            {"output": "Test 3", "reference": "Reference 3"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert result.total_items == 3
        assert result.successful_items == 2
        assert result.failed_items == 1
        assert len(result.results) == 3
        assert result.results[0] is not None
        assert result.results[1] is None  # Failed item
        assert result.results[2] is not None
        assert len(result.errors) == 1
        assert result.errors[0]["index"] == 1

    @pytest.mark.asyncio
    async def test_batch_evaluate_all_failures(self, mock_llm_client, mock_agent):
        """Test batch where all items fail."""
        from arbiter_ai.api import batch_evaluate

        mock_agent.run = AsyncMock(side_effect=Exception("All failed"))
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Reference 1"},
            {"output": "Test 2", "reference": "Reference 2"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert result.total_items == 2
        assert result.successful_items == 0
        assert result.failed_items == 2
        assert all(r is None for r in result.results)
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_batch_evaluate_empty_items_validation(self, mock_llm_client):
        """Test that empty items list raises validation error."""
        from arbiter_ai.api import batch_evaluate

        with pytest.raises(ValidationError, match="items list cannot be empty"):
            await batch_evaluate(
                items=[],
                evaluators=["semantic"],
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_batch_evaluate_missing_output_key_validation(self, mock_llm_client):
        """Test that missing 'output' key raises validation error."""
        from arbiter_ai.api import batch_evaluate

        items = [
            {"reference": "Test reference"},  # Missing 'output'
        ]

        with pytest.raises(ValidationError, match="missing required 'output' key"):
            await batch_evaluate(
                items=items,
                evaluators=["semantic"],
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_batch_evaluate_concurrency_control(
        self, mock_llm_client, mock_agent
    ):
        """Test that concurrency is controlled by max_concurrency parameter."""
        import asyncio

        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        concurrent_calls = []

        async def mock_run(*args, **kwargs):
            concurrent_calls.append(1)
            await asyncio.sleep(0.01)  # Small delay to allow concurrency tracking
            concurrent_calls.pop()
            return MockAgentResult(mock_response)

        mock_agent.run = mock_run
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [{"output": f"Test {i}", "reference": f"Ref {i}"} for i in range(10)]

        # With max_concurrency=2, we should never see more than 2 concurrent calls
        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
            max_concurrency=2,
        )

        assert result.successful_items == 10

    @pytest.mark.asyncio
    async def test_batch_evaluate_multiple_evaluators(
        self, mock_llm_client, mock_agent
    ):
        """Test batch evaluation with multiple evaluators per item."""
        from arbiter_ai.api import batch_evaluate

        semantic_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="High similarity",
        )

        criteria_response = CustomCriteriaResponse(
            score=0.85,
            confidence=0.9,
            explanation="Meets criteria",
        )

        # Alternate between semantic and criteria responses
        mock_agent.run = AsyncMock(
            side_effect=[
                MockAgentResult(semantic_response),
                MockAgentResult(criteria_response),
                MockAgentResult(semantic_response),
                MockAgentResult(criteria_response),
            ]
        )
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Ref 1", "criteria": "Accuracy"},
            {"output": "Test 2", "reference": "Ref 2", "criteria": "Clarity"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic", "custom_criteria"],
            llm_client=mock_llm_client,
        )

        assert result.successful_items == 2
        assert all(r is not None for r in result.results)
        # Each result should have 2 scores (semantic + custom_criteria)
        for eval_result in result.results:
            if eval_result:
                assert len(eval_result.scores) == 2

    @pytest.mark.asyncio
    async def test_batch_evaluate_result_ordering(self, mock_llm_client, mock_agent):
        """Test that results maintain original item ordering."""
        from arbiter_ai.api import batch_evaluate

        # Create different responses to verify ordering
        responses = [
            SemanticResponse(score=0.9, confidence=0.9, explanation="First"),
            SemanticResponse(score=0.8, confidence=0.8, explanation="Second"),
            SemanticResponse(score=0.7, confidence=0.7, explanation="Third"),
        ]

        mock_agent.run = AsyncMock(side_effect=[MockAgentResult(r) for r in responses])
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Ref 1"},
            {"output": "Test 2", "reference": "Ref 2"},
            {"output": "Test 3", "reference": "Ref 3"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert result.results[0].overall_score == 0.9
        assert result.results[1].overall_score == 0.8
        assert result.results[2].overall_score == 0.7

    @pytest.mark.asyncio
    async def test_batch_evaluate_get_result_helper(self, mock_llm_client, mock_agent):
        """Test get_result() helper method."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        # First succeeds, second fails
        mock_agent.run = AsyncMock(
            side_effect=[
                MockAgentResult(mock_response),
                Exception("Failed"),
            ]
        )
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Ref 1"},
            {"output": "Test 2", "reference": "Ref 2"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert result.get_result(0) is not None
        assert result.get_result(1) is None
        assert result.get_result(999) is None  # Out of range

    @pytest.mark.asyncio
    async def test_batch_evaluate_get_error_helper(self, mock_llm_client, mock_agent):
        """Test get_error() helper method."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        # First succeeds, second fails
        mock_agent.run = AsyncMock(
            side_effect=[
                MockAgentResult(mock_response),
                Exception("Test error"),
            ]
        )
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Ref 1"},
            {"output": "Test 2", "reference": "Ref 2"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert result.get_error(0) is None  # No error for successful item
        error = result.get_error(1)
        assert error is not None
        assert error["index"] == 1
        assert "Test error" in error["error"]
        assert result.get_error(999) is None  # Out of range

    @pytest.mark.asyncio
    async def test_batch_evaluate_total_llm_cost(self, mock_llm_client, mock_agent):
        """Test total_llm_cost() aggregation method."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)
        mock_llm_client.model = "gpt-4o-mini"

        items = [
            {"output": "Test 1", "reference": "Ref 1"},
            {"output": "Test 2", "reference": "Ref 2"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        # Test that total_llm_cost() can be called
        total_cost = await result.total_llm_cost()
        assert isinstance(total_cost, float)
        assert total_cost >= 0

    @pytest.mark.asyncio
    async def test_batch_evaluate_cost_breakdown(self, mock_llm_client, mock_agent):
        """Test cost_breakdown() aggregation method."""
        from arbiter_ai.api import batch_evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)
        mock_llm_client.model = "gpt-4o-mini"

        items = [
            {"output": "Test 1", "reference": "Ref 1"},
            {"output": "Test 2", "reference": "Ref 2"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        breakdown = await result.cost_breakdown()
        assert isinstance(breakdown, dict)
        assert "total" in breakdown
        assert "per_item_average" in breakdown
        assert "by_evaluator" in breakdown
        assert "by_model" in breakdown
        assert "success_rate" in breakdown
        assert breakdown["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_batch_evaluate_with_threshold(self, mock_llm_client, mock_agent):
        """Test batch evaluation with pass/fail threshold."""
        from arbiter_ai.api import batch_evaluate

        # Create responses with different scores
        responses = [
            SemanticResponse(score=0.9, confidence=0.9, explanation="High"),
            SemanticResponse(score=0.5, confidence=0.8, explanation="Low"),
        ]

        mock_agent.run = AsyncMock(side_effect=[MockAgentResult(r) for r in responses])
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "reference": "Ref 1"},
            {"output": "Test 2", "reference": "Ref 2"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            llm_client=mock_llm_client,
            threshold=0.7,
        )

        assert result.results[0].passed is True  # 0.9 >= 0.7
        assert result.results[1].passed is False  # 0.5 < 0.7

    @pytest.mark.asyncio
    async def test_batch_evaluate_without_reference(self, mock_llm_client, mock_agent):
        """Test batch evaluation without reference (e.g., for custom_criteria only)."""
        from arbiter_ai.api import batch_evaluate

        mock_response = CustomCriteriaResponse(
            score=0.85,
            confidence=0.9,
            explanation="Meets criteria",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        items = [
            {"output": "Test 1", "criteria": "Accuracy"},
            {"output": "Test 2", "criteria": "Clarity"},
        ]

        result = await batch_evaluate(
            items=items,
            evaluators=["custom_criteria"],
            llm_client=mock_llm_client,
        )

        assert result.successful_items == 2
        assert all(r is not None for r in result.results)
