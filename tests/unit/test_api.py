"""Unit tests for main API functions (evaluate and compare)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter_ai.core.exceptions import EvaluatorError, ValidationError
from arbiter_ai.core.llm_client import LLMClient
from arbiter_ai.core.models import ComparisonResult, EvaluationResult
from arbiter_ai.evaluators.custom_criteria import CustomCriteriaResponse
from arbiter_ai.evaluators.pairwise import PairwiseResponse
from arbiter_ai.evaluators.semantic import SemanticResponse
from tests.conftest import MockAgentResult


class TestEvaluateFunction:
    """Test suite for evaluate() function."""

    @pytest.mark.asyncio
    async def test_evaluate_basic_semantic(self, mock_llm_client, mock_agent):
        """Test basic semantic evaluation."""
        from arbiter_ai.api import evaluate

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
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            llm_client=mock_llm_client,
        )

        assert isinstance(result, EvaluationResult)
        assert result.overall_score == 0.9
        assert result.passed is True
        assert len(result.scores) == 1
        assert result.scores[0].name == "semantic"
        assert result.partial is False
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_evaluate_default_evaluators(self, mock_llm_client, mock_agent):
        """Test that default evaluator is semantic."""
        from arbiter_ai.api import evaluate

        mock_response = SemanticResponse(
            score=0.8,
            confidence=0.8,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            llm_client=mock_llm_client,
        )

        assert len(result.scores) == 1
        assert result.scores[0].name == "semantic"

    @pytest.mark.asyncio
    async def test_evaluate_custom_criteria(self, mock_llm_client, mock_agent):
        """Test custom criteria evaluation."""
        from arbiter_ai.api import evaluate

        mock_response = CustomCriteriaResponse(
            score=0.85,
            confidence=0.9,
            explanation="Meets criteria",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            criteria="Accuracy and clarity",
            evaluators=["custom_criteria"],
            llm_client=mock_llm_client,
        )

        assert len(result.scores) == 1
        assert result.scores[0].name == "custom_criteria"
        assert result.overall_score == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_custom_criteria_without_criteria(self, mock_llm_client):
        """Test that custom_criteria evaluator requires criteria."""
        from arbiter_ai.api import evaluate

        with pytest.raises(ValidationError, match="requires criteria"):
            await evaluate(
                output="Test output",
                evaluators=["custom_criteria"],
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_evaluate_multiple_evaluators(self, mock_llm_client, mock_agent):
        """Test evaluation with multiple evaluators."""
        from arbiter_ai.api import evaluate

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

        # First call returns semantic, second returns criteria
        mock_agent.run = AsyncMock(
            side_effect=[
                MockAgentResult(semantic_response),
                MockAgentResult(criteria_response),
            ]
        )
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            criteria="Accuracy",
            evaluators=["semantic", "custom_criteria"],
            llm_client=mock_llm_client,
        )

        assert len(result.scores) == 2
        assert result.overall_score == (0.9 + 0.85) / 2
        assert result.partial is False

    @pytest.mark.asyncio
    async def test_evaluate_validation_empty_output(self, mock_llm_client):
        """Test validation for empty output."""
        from arbiter_ai.api import evaluate

        with pytest.raises(ValidationError, match="output cannot be empty"):
            await evaluate(output="", llm_client=mock_llm_client)

        with pytest.raises(ValidationError, match="output cannot be empty"):
            await evaluate(output="   ", llm_client=mock_llm_client)

    @pytest.mark.asyncio
    async def test_evaluate_validation_empty_reference(self, mock_llm_client):
        """Test validation for empty reference."""
        from arbiter_ai.api import evaluate

        with pytest.raises(ValidationError, match="reference cannot be empty"):
            await evaluate(output="Test", reference="", llm_client=mock_llm_client)

        with pytest.raises(ValidationError, match="reference cannot be empty"):
            await evaluate(output="Test", reference="   ", llm_client=mock_llm_client)

    @pytest.mark.asyncio
    async def test_evaluate_validation_empty_criteria(self, mock_llm_client):
        """Test validation for empty criteria."""
        from arbiter_ai.api import evaluate

        with pytest.raises(ValidationError, match="criteria cannot be empty"):
            await evaluate(output="Test", criteria="", llm_client=mock_llm_client)

        with pytest.raises(ValidationError, match="criteria cannot be empty"):
            await evaluate(output="Test", criteria="   ", llm_client=mock_llm_client)

    @pytest.mark.asyncio
    async def test_evaluate_unknown_evaluator(self, mock_llm_client):
        """Test that unknown evaluator raises error."""
        from arbiter_ai.api import evaluate

        with pytest.raises(ValidationError, match="Unknown evaluator"):
            await evaluate(
                output="Test",
                evaluators=["unknown_evaluator"],
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_evaluate_threshold(self, mock_llm_client, mock_agent):
        """Test threshold-based pass/fail."""
        from arbiter_ai.api import evaluate

        mock_response = SemanticResponse(
            score=0.8,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Score 0.8 >= threshold 0.7, should pass
        result = await evaluate(
            output="Test",
            reference="Test ref",
            threshold=0.7,
            llm_client=mock_llm_client,
        )
        assert result.passed is True

        # Score 0.8 < threshold 0.9, should fail
        result = await evaluate(
            output="Test",
            reference="Test ref",
            threshold=0.9,
            llm_client=mock_llm_client,
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_evaluate_interaction_tracking(self, mock_llm_client, mock_agent):
        """Test that interactions are tracked."""
        from arbiter_ai.api import evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            llm_client=mock_llm_client,
        )

        assert len(result.interactions) > 0
        assert result.total_tokens > 0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_evaluate_metrics(self, mock_llm_client, mock_agent):
        """Test that metrics are created."""
        from arbiter_ai.api import evaluate

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            llm_client=mock_llm_client,
        )

        assert len(result.metrics) == 1
        assert result.metrics[0].name == "semantic"
        assert result.metrics[0].evaluator == "semantic"
        assert result.metrics[0].model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_evaluate_partial_failure(self, mock_llm_client, mock_agent):
        """Test partial failure scenario."""
        from arbiter_ai.api import evaluate

        semantic_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        # First evaluator succeeds, second fails
        mock_agent.run = AsyncMock(
            side_effect=[
                MockAgentResult(semantic_response),
                Exception("Custom criteria failed"),
            ]
        )
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            criteria="Accuracy",
            evaluators=["semantic", "custom_criteria"],
            llm_client=mock_llm_client,
        )

        assert result.partial is True
        assert len(result.scores) == 1
        assert len(result.errors) == 1
        assert "custom_criteria" in result.errors
        assert result.overall_score == 0.9  # Only successful evaluator

    @pytest.mark.asyncio
    async def test_evaluate_all_failures(self, mock_llm_client, mock_agent):
        """Test that all failures raise error."""
        from arbiter_ai.api import evaluate

        mock_agent.run = AsyncMock(side_effect=Exception("All failed"))
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError, match="All evaluators failed"):
            await evaluate(
                output="Test output",
                reference="Test reference",
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    @patch("arbiter_ai.api.LLMManager")
    async def test_evaluate_creates_client(self, mock_manager, mock_agent):
        """Test that client is created if not provided."""
        from arbiter_ai.api import evaluate

        mock_client = MagicMock(spec=LLMClient)
        mock_client.model = "gpt-4o"
        mock_client.temperature = 0.0
        mock_client.create_agent = MagicMock(return_value=mock_agent)

        mock_manager.get_client = AsyncMock(return_value=mock_client)

        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            model="gpt-4o",
        )

        mock_manager.get_client.assert_called_once()
        assert result.overall_score == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_with_middleware(self, mock_llm_client, mock_agent):
        """Test evaluation with middleware wrapping."""
        from arbiter_ai.api import evaluate
        from arbiter_ai.core.middleware import Middleware, MiddlewarePipeline

        # Create mock middleware
        class TestMiddleware(Middleware):
            async def process(self, output, reference, next_handler, context):
                # Middleware wraps the core evaluation
                result = await next_handler(output, reference)
                # Can modify result here
                return result

        pipeline = MiddlewarePipeline()
        pipeline.add(TestMiddleware())

        mock_response = SemanticResponse(
            score=0.85,
            confidence=0.9,
            explanation="With middleware",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await evaluate(
            output="Test output",
            reference="Test reference",
            middleware=pipeline,
            llm_client=mock_llm_client,
        )

        assert result.overall_score == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_defensive_evaluator_lookup_failure(
        self, mock_llm_client, mock_agent
    ):
        """Test defensive code path when evaluator lookup returns None."""
        from unittest.mock import patch

        from arbiter_ai.api import evaluate

        # Mock validate_evaluator_name to pass but get_evaluator_class to return None
        with patch("arbiter_ai.api.validate_evaluator_name"):
            with patch("arbiter_ai.api.get_evaluator_class", return_value=None):
                with pytest.raises(ValidationError, match="not found in registry"):
                    await evaluate(
                        output="Test",
                        evaluators=["nonexistent"],
                        llm_client=mock_llm_client,
                    )

    @pytest.mark.asyncio
    async def test_evaluate_unexpected_exception_handling(
        self, mock_llm_client, mock_agent
    ):
        """Test generic exception handler for unexpected errors during evaluation."""
        from arbiter_ai.api import evaluate
        from arbiter_ai.core.models import Score

        # Create one evaluator that succeeds and one that raises unexpected exception
        mock_failing_evaluator = MagicMock()
        mock_failing_evaluator.name = "failing_evaluator"
        mock_failing_evaluator.evaluate = AsyncMock(
            side_effect=RuntimeError("Unexpected system error")
        )
        mock_failing_evaluator.get_interactions = MagicMock(return_value=[])

        mock_success_evaluator = MagicMock()
        mock_success_evaluator.name = "success_evaluator"
        mock_success_evaluator.evaluate = AsyncMock(
            return_value=Score(
                name="success_evaluator",
                value=0.8,
                confidence=0.9,
                explanation="Success",
            )
        )
        mock_success_evaluator.get_interactions = MagicMock(return_value=[])

        def get_evaluator_side_effect(name):
            if name == "failing_evaluator":
                return MagicMock(return_value=mock_failing_evaluator)
            else:
                return MagicMock(return_value=mock_success_evaluator)

        with patch(
            "arbiter_ai.api.get_evaluator_class", side_effect=get_evaluator_side_effect
        ):
            with patch("arbiter_ai.api.validate_evaluator_name"):
                result = await evaluate(
                    output="Test",
                    evaluators=["failing_evaluator", "success_evaluator"],
                    llm_client=mock_llm_client,
                )

                # Should handle error gracefully - one succeeds, one fails
                assert result.overall_score == 0.8  # Only successful evaluator
                assert result.passed is True  # At least one succeeded
                assert "failing_evaluator" in result.errors
                assert "Unexpected error" in result.errors["failing_evaluator"]


class TestCompareFunction:
    """Test suite for compare() function."""

    @pytest.mark.asyncio
    async def test_compare_basic(self, mock_llm_client, mock_agent):
        """Test basic comparison."""
        from arbiter_ai.api import compare

        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A is better",
            aspect_scores=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
        )

        assert isinstance(result, ComparisonResult)
        assert result.winner == "output_a"
        assert result.confidence == 0.9
        assert result.output_a == "Output A"
        assert result.output_b == "Output B"

    @pytest.mark.asyncio
    async def test_compare_with_reference(self, mock_llm_client, mock_agent):
        """Test comparison with reference."""
        from arbiter_ai.api import compare

        mock_response = PairwiseResponse(
            winner="output_b",
            confidence=0.85,
            reasoning="Output B matches reference better",
            aspect_scores=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            reference="Reference text",
            llm_client=mock_llm_client,
        )

        assert result.winner == "output_b"
        assert result.reference == "Reference text"

    @pytest.mark.asyncio
    async def test_compare_with_criteria(self, mock_llm_client, mock_agent):
        """Test comparison with criteria."""
        from arbiter_ai.api import compare

        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A meets criteria better",
            aspect_scores=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            criteria="Accuracy and clarity",
            llm_client=mock_llm_client,
        )

        assert result.winner == "output_a"
        assert result.criteria == "Accuracy and clarity"

    @pytest.mark.asyncio
    async def test_compare_tie(self, mock_llm_client, mock_agent):
        """Test tie scenario."""
        from arbiter_ai.api import compare

        mock_response = PairwiseResponse(
            winner="tie",
            confidence=0.7,
            reasoning="Both outputs are equivalent",
            aspect_scores=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
        )

        assert result.winner == "tie"

    @pytest.mark.asyncio
    async def test_compare_aspect_scores(self, mock_llm_client, mock_agent):
        """Test comparison with aspect scores."""
        from arbiter_ai.api import compare
        from arbiter_ai.evaluators.pairwise import AspectComparison

        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A wins overall",
            aspect_comparisons=[
                AspectComparison(
                    aspect="accuracy",
                    output_a_score=0.9,
                    output_b_score=0.7,
                    reasoning="A is more accurate",
                ),
                AspectComparison(
                    aspect="clarity",
                    output_a_score=0.7,
                    output_b_score=0.9,
                    reasoning="B is clearer",
                ),
            ],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
        )

        assert len(result.aspect_scores) == 2
        assert result.get_aspect_score("accuracy", "output_a") == 0.9
        assert result.get_aspect_score("accuracy", "output_b") == 0.7
        assert result.get_aspect_score("clarity", "output_a") == 0.7
        assert result.get_aspect_score("clarity", "output_b") == 0.9

    @pytest.mark.asyncio
    async def test_compare_error_handling(self, mock_llm_client, mock_agent):
        """Test error handling in comparison."""
        from arbiter_ai.api import compare

        mock_agent.run = AsyncMock(side_effect=Exception("Comparison failed"))
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(Exception):
            await compare(
                output_a="Output A",
                output_b="Output B",
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_compare_interaction_tracking(self, mock_llm_client, mock_agent):
        """Test that interactions are tracked."""
        from arbiter_ai.api import compare

        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            aspect_scores=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            llm_client=mock_llm_client,
        )

        assert len(result.interactions) > 0
        assert result.total_tokens > 0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    @patch("arbiter_ai.api.LLMManager")
    async def test_compare_creates_client(self, mock_manager, mock_agent):
        """Test that client is created if not provided."""
        from arbiter_ai.api import compare

        mock_client = MagicMock(spec=LLMClient)
        mock_client.model = "gpt-4o"
        mock_client.temperature = 0.0
        mock_client.create_agent = MagicMock(return_value=mock_agent)

        mock_manager.get_client = AsyncMock(return_value=mock_client)

        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            aspect_scores=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)

        result = await compare(
            output_a="Output A",
            output_b="Output B",
            model="gpt-4o",
        )

        mock_manager.get_client.assert_called_once()
        assert result.winner == "output_a"

    @pytest.mark.asyncio
    async def test_compare_empty_output_a_validation(self, mock_llm_client):
        """Test that empty output_a raises validation error."""
        from arbiter_ai.api import compare

        with pytest.raises(ValidationError, match="output_a cannot be empty"):
            await compare(
                output_a="",
                output_b="Output B",
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_compare_empty_output_b_validation(self, mock_llm_client):
        """Test that empty output_b raises validation error."""
        from arbiter_ai.api import compare

        with pytest.raises(ValidationError, match="output_b cannot be empty"):
            await compare(
                output_a="Output A",
                output_b="",
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_compare_empty_criteria_validation(self, mock_llm_client):
        """Test that empty criteria raises validation error."""
        from arbiter_ai.api import compare

        with pytest.raises(ValidationError, match="criteria cannot be empty"):
            await compare(
                output_a="Output A",
                output_b="Output B",
                criteria="",
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_compare_empty_reference_validation(self, mock_llm_client):
        """Test that empty reference raises validation error."""
        from arbiter_ai.api import compare

        with pytest.raises(ValidationError, match="reference cannot be empty"):
            await compare(
                output_a="Output A",
                output_b="Output B",
                reference="",
                llm_client=mock_llm_client,
            )
