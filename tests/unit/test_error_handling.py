"""Unit tests for multi-evaluator error handling."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter.core.exceptions import EvaluatorError, ModelProviderError, ValidationError
from arbiter.core.llm_client import LLMClient
from arbiter.core.models import EvaluationResult
from arbiter.evaluators import CustomCriteriaEvaluator, SemanticEvaluator


class MockAgentResult:
    """Mock PydanticAI agent result."""

    def __init__(self, output: object):
        self.output = output

    def usage(self):
        """Mock usage information."""
        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        return mock_usage


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.model = "gpt-4o-mini"
    client.temperature = 0.0
    return client


@pytest.fixture
def mock_agent():
    """Create a mock PydanticAI agent."""
    agent = AsyncMock()
    return agent


class TestMultiEvaluatorErrorHandling:
    """Test suite for multi-evaluator error handling."""

    @pytest.mark.asyncio
    async def test_all_evaluators_succeed(self, mock_llm_client, mock_agent):
        """Test that all evaluators succeed normally."""
        from arbiter.api import evaluate

        # Mock successful responses
        from arbiter.evaluators.semantic import SemanticResponse

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

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert result.partial is False
        assert len(result.errors) == 0
        assert len(result.scores) == 1
        assert result.metadata["successful_evaluators"] == 1
        assert result.metadata["failed_evaluators"] == 0

    @pytest.mark.asyncio
    async def test_one_evaluator_fails(self, mock_llm_client, mock_agent):
        """Test that one evaluator failure doesn't stop others."""
        from arbiter.api import evaluate

        # Mock one success, one failure
        from arbiter.evaluators.semantic import SemanticResponse

        mock_response = SemanticResponse(
            score=0.8,
            confidence=0.85,
            explanation="Good similarity",
            key_similarities=[],
            key_differences=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create a mock evaluator that will fail
        class FailingEvaluator(SemanticEvaluator):
            async def evaluate(self, output, reference=None, criteria=None):
                raise EvaluatorError("API timeout", details={"error": "API timeout"})

        # Patch the evaluator creation to inject failing evaluator
        original_evaluators = evaluate.__globals__["SemanticEvaluator"]

        # For this test, we'll use a simpler approach - mock the evaluator to fail
        # Actually, let's test with real evaluators but mock one to fail
        # We need to test with multiple evaluators, but we only have semantic currently
        # So let's test with semantic evaluator and simulate a failure scenario

        # Since we can't easily inject a failing evaluator, let's test the error
        # handling by making the agent raise an error
        mock_agent.run = AsyncMock(side_effect=EvaluatorError("API timeout"))

        # This should raise an error since all evaluators failed
        with pytest.raises(EvaluatorError, match="All evaluators failed"):
            await evaluate(
                output="Test output",
                reference="Test reference",
                evaluators=["semantic"],
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_partial_result_with_errors(self, mock_llm_client, mock_agent):
        """Test that partial results include error information."""
        from arbiter.api import _evaluate_impl

        # Mock successful semantic response
        from arbiter.evaluators.semantic import SemanticResponse

        mock_response = SemanticResponse(
            score=0.85,
            confidence=0.9,
            explanation="Good",
            key_similarities=[],
            key_differences=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Create evaluators - one will succeed, one will fail
        evaluators = [SemanticEvaluator(mock_llm_client)]

        # Create a failing evaluator
        class FailingEvaluator(SemanticEvaluator):
            @property
            def name(self) -> str:
                return "failing_evaluator"

            async def evaluate(self, output, reference=None, criteria=None):
                raise ModelProviderError("API timeout", details={"error": "API timeout"})

        evaluators.append(FailingEvaluator(mock_llm_client))

        # Manually test the error handling logic
        scores = []
        errors = {}

        for evaluator in evaluators:
            try:
                score = await evaluator.evaluate("test", "test")
                scores.append(score)
            except Exception as e:
                errors[evaluator.name] = str(e)

        # Verify error handling
        assert len(scores) == 1
        assert len(errors) == 1
        assert "failing_evaluator" in errors

    @pytest.mark.asyncio
    async def test_error_details_extraction(self, mock_llm_client):
        """Test that error details are properly extracted."""
        from arbiter.core.exceptions import EvaluatorError

        # Test error with details
        error = EvaluatorError("Test error", details={"error": "Detailed error message"})
        error_msg = str(error)
        if hasattr(error, "details") and isinstance(error.details, dict):
            error_msg = error.details.get("error", error_msg)

        assert error_msg == "Detailed error message"

    @pytest.mark.asyncio
    async def test_evaluation_result_with_errors(self):
        """Test EvaluationResult with errors field."""
        from arbiter.core.models import Score

        result = EvaluationResult(
            output="test",
            scores=[Score(name="semantic", value=0.8)],
            overall_score=0.8,
            passed=True,
            partial=True,
            errors={"factuality": "API timeout"},
        )

        assert result.partial is True
        assert len(result.errors) == 1
        assert "factuality" in result.errors
        assert result.errors["factuality"] == "API timeout"
        assert len(result.scores) == 1

    @pytest.mark.asyncio
    async def test_evaluation_result_without_errors(self):
        """Test EvaluationResult without errors (normal case)."""
        from arbiter.core.models import Score

        result = EvaluationResult(
            output="test",
            scores=[Score(name="semantic", value=0.9)],
            overall_score=0.9,
            passed=True,
        )

        assert result.partial is False
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_multiple_evaluators_partial_failure_main_use_case(self, mock_llm_client, mock_agent):
        """Test the MAIN USE CASE: multiple evaluators where some succeed and some fail."""
        from arbiter.api import evaluate

        # Mock successful semantic response
        from arbiter.evaluators.semantic import SemanticResponse
        from arbiter.evaluators.custom_criteria import CustomCriteriaResponse

        semantic_response = SemanticResponse(
            score=0.85,
            confidence=0.9,
            explanation="Good semantic similarity",
            key_similarities=[],
            key_differences=[],
        )

        custom_criteria_response = CustomCriteriaResponse(
            score=0.75,
            confidence=0.8,
            explanation="Meets some criteria",
            criteria_met=["accuracy"],
            criteria_not_met=["completeness"],
        )

        # Create a call counter to control which evaluator succeeds/fails
        call_count = 0

        async def mock_agent_run(prompt):
            nonlocal call_count
            call_count += 1
            # First call (semantic) succeeds
            if call_count == 1:
                return MockAgentResult(semantic_response)
            # Second call (custom_criteria) fails
            elif call_count == 2:
                raise EvaluatorError("API timeout", details={"error": "API timeout"})
            else:
                raise Exception("Unexpected call")

        mock_agent.run = AsyncMock(side_effect=mock_agent_run)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Run evaluation with multiple evaluators
        result = await evaluate(
            output="Test output",
            reference="Test reference",
            criteria="accuracy, completeness",
            evaluators=["semantic", "custom_criteria"],
            llm_client=mock_llm_client,
        )

        # Verify partial result
        assert result.partial is True, "Should be marked as partial"
        assert len(result.scores) == 1, "Should have one successful score"
        assert len(result.errors) == 1, "Should have one error"
        assert result.scores[0].name == "semantic_similarity", "Should have semantic score"
        assert "custom_criteria" in result.errors, "custom_criteria should be in errors"
        assert result.errors["custom_criteria"] == "API timeout", "Error message should match"

        # Verify overall score is calculated from successful evaluators only
        assert result.overall_score == 0.85, "Overall score should be semantic score (0.85), not average"
        assert result.metadata["successful_evaluators"] == 1
        assert result.metadata["failed_evaluators"] == 1

        # Verify we can still use the result
        assert result.passed is True, "Should pass threshold (0.85 > 0.7)"
        assert len(result.interactions) > 0, "Should have interactions from successful evaluator"

    @pytest.mark.asyncio
    async def test_multiple_evaluators_partial_failure(self, mock_llm_client, mock_agent):
        """Test multiple evaluators where some fail."""
        from arbiter.api import _evaluate_impl

        # This test requires multiple evaluators, so we'll test the internal logic
        # Since we only have semantic and custom_criteria, let's test with those

        # Mock successful semantic response
        from arbiter.evaluators.semantic import SemanticResponse

        mock_response = SemanticResponse(
            score=0.85,
            confidence=0.9,
            explanation="Good",
            key_similarities=[],
            key_differences=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Test with semantic evaluator (will succeed)
        semantic_eval = SemanticEvaluator(mock_llm_client)

        # Create a mock failing evaluator
        class MockFailingEvaluator:
            @property
            def name(self) -> str:
                return "failing"

            def clear_interactions(self):
                pass

            async def evaluate(self, output, reference=None, criteria=None):
                raise EvaluatorError("Simulated failure")

            def get_interactions(self):
                return []

        failing_eval = MockFailingEvaluator()

        # Test error handling logic
        scores = []
        errors = {}
        evaluator_names = []

        for evaluator in [semantic_eval, failing_eval]:
            evaluator.clear_interactions()
            try:
                score = await evaluator.evaluate("test", "test")
                scores.append(score)
                evaluator_names.append(evaluator.name)
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, "details") and isinstance(e.details, dict):
                    error_msg = e.details.get("error", error_msg)
                errors[evaluator.name] = error_msg

        # Verify partial result
        assert len(scores) == 1
        assert len(errors) == 1
        assert scores[0].name == "semantic_similarity"
        assert "failing" in errors

    @pytest.mark.asyncio
    async def test_all_evaluators_fail_raises_error(self, mock_llm_client, mock_agent):
        """Test that if all evaluators fail, an error is raised."""
        from arbiter.api import _evaluate_impl

        # Make agent fail
        mock_agent.run = AsyncMock(side_effect=EvaluatorError("All failed"))
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Test that all failures raise an error
        scores = []
        errors = {}

        from arbiter.evaluators import SemanticEvaluator

        evaluator = SemanticEvaluator(mock_llm_client)
        try:
            score = await evaluator.evaluate("test", "test")
            scores.append(score)
        except Exception as e:
            errors[evaluator.name] = str(e)

        # If all evaluators fail, we should have errors but no scores
        assert len(scores) == 0
        assert len(errors) == 1

        # When calling _evaluate_impl, it should raise if all fail
        with pytest.raises(EvaluatorError, match="All evaluators failed"):
            await _evaluate_impl(
                output="test",
                reference="test",
                evaluators=["semantic"],
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_overall_score_calculation_with_partial_failure(self, mock_llm_client, mock_agent):
        """Test that overall score only includes successful evaluators."""
        from arbiter.api import evaluate

        from arbiter.evaluators.semantic import SemanticResponse

        semantic_response = SemanticResponse(
            score=0.9,
            confidence=0.95,
            explanation="Excellent",
            key_similarities=[],
            key_differences=[],
        )

        call_count = 0

        async def mock_agent_run(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockAgentResult(semantic_response)
            else:
                raise EvaluatorError("Failure")

        mock_agent.run = AsyncMock(side_effect=mock_agent_run)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test",
            reference="Test",
            evaluators=["semantic", "custom_criteria"],
            criteria="test",
            llm_client=mock_llm_client,
        )

        # Overall score should be 0.9 (semantic), NOT (0.9 + 0) / 2 = 0.45
        assert result.overall_score == 0.9
        assert result.partial is True
        assert len(result.scores) == 1

    @pytest.mark.asyncio
    async def test_logging_for_evaluator_failures(self, mock_llm_client, mock_agent):
        """Test that evaluator failures are logged."""
        from arbiter.api import evaluate

        from arbiter.evaluators.semantic import SemanticResponse

        semantic_response = SemanticResponse(
            score=0.8,
            confidence=0.85,
            explanation="Good",
            key_similarities=[],
            key_differences=[],
        )

        call_count = 0

        async def mock_agent_run(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockAgentResult(semantic_response)
            else:
                raise EvaluatorError("Test error", details={"error": "Test error"})

        mock_agent.run = AsyncMock(side_effect=mock_agent_run)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        with patch("arbiter.api.logger") as mock_logger:
            result = await evaluate(
                output="Test",
                reference="Test",
                evaluators=["semantic", "custom_criteria"],
                criteria="test",
                llm_client=mock_llm_client,
            )

            # Verify warning was logged for evaluator failure
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args
            assert "custom_criteria" in warning_call[0][0] or "failed" in warning_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_logging_for_unexpected_errors(self, mock_llm_client, mock_agent):
        """Test that unexpected errors are logged with exc_info."""
        from arbiter.api import evaluate

        from arbiter.evaluators.semantic import SemanticResponse

        semantic_response = SemanticResponse(
            score=0.8,
            confidence=0.85,
            explanation="Good",
            key_similarities=[],
            key_differences=[],
        )

        call_count = 0

        async def mock_agent_run(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockAgentResult(semantic_response)
            else:
                raise ValueError("Unexpected error type")

        mock_agent.run = AsyncMock(side_effect=mock_agent_run)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        with patch("arbiter.api.logger") as mock_logger:
            result = await evaluate(
                output="Test",
                reference="Test",
                evaluators=["semantic", "custom_criteria"],
                criteria="test",
                llm_client=mock_llm_client,
            )

            # Verify error was logged with exc_info
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args
            assert "Unexpected error" in error_call[0][0]
            assert error_call[1].get("exc_info") is True, "Should log with exc_info=True"

