"""Unit tests for PairwiseComparisonEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter.core.exceptions import EvaluatorError
from arbiter.core.models import ComparisonResult
from arbiter.evaluators.pairwise import (
    AspectComparison,
    PairwiseComparisonEvaluator,
    PairwiseResponse,
)
from tests.conftest import MockAgentResult


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a PairwiseComparisonEvaluator instance."""
    return PairwiseComparisonEvaluator(llm_client=mock_llm_client)


class TestPairwiseComparisonEvaluator:
    """Test suite for PairwiseComparisonEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "pairwise_comparison"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "expert evaluator" in prompt.lower()
        assert "compare" in prompt.lower()

    def test_user_prompt_not_used(self, evaluator):
        """Test that _get_user_prompt raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            evaluator._get_user_prompt("output", None, None)

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == PairwiseResponse

    @pytest.mark.asyncio
    async def test_compute_score_not_used(self, evaluator):
        """Test that _compute_score raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await evaluator._compute_score(MagicMock())

    @pytest.mark.asyncio
    async def test_compare_output_a_wins(self, evaluator, mock_agent):
        """Test comparison where output_a wins."""
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A is more accurate and complete",
            aspect_comparisons=[
                AspectComparison(
                    aspect="accuracy",
                    output_a_score=0.95,
                    output_b_score=0.85,
                    reasoning="Output A has more accurate information",
                ),
                AspectComparison(
                    aspect="clarity",
                    output_a_score=0.9,
                    output_b_score=0.9,
                    reasoning="Both are equally clear",
                ),
            ],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        comparison = await evaluator.compare(
            output_a="GPT-4 response: Paris is the capital of France, founded in 3rd century BC.",
            output_b="Claude response: The capital of France is Paris.",
            criteria="accuracy, clarity",
        )

        assert comparison.winner == "output_a"
        assert comparison.confidence == 0.9
        assert "more accurate" in comparison.reasoning.lower()
        assert len(comparison.aspect_scores) == 2
        assert comparison.aspect_scores["accuracy"]["output_a"] == 0.95
        assert comparison.aspect_scores["accuracy"]["output_b"] == 0.85

    @pytest.mark.asyncio
    async def test_compare_output_b_wins(self, evaluator, mock_agent):
        """Test comparison where output_b wins."""
        mock_response = PairwiseResponse(
            winner="output_b",
            confidence=0.85,
            reasoning="Output B is clearer and more concise",
            aspect_comparisons=[
                AspectComparison(
                    aspect="clarity",
                    output_a_score=0.7,
                    output_b_score=0.95,
                    reasoning="Output B is much clearer",
                ),
            ],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        comparison = await evaluator.compare(
            output_a="A long and confusing response",
            output_b="A clear and concise response",
            criteria="clarity",
        )

        assert comparison.winner == "output_b"
        assert comparison.confidence == 0.85

    @pytest.mark.asyncio
    async def test_compare_tie(self, evaluator, mock_agent):
        """Test comparison resulting in a tie."""
        mock_response = PairwiseResponse(
            winner="tie",
            confidence=0.8,
            reasoning="Both outputs are equivalent in quality",
            aspect_comparisons=[
                AspectComparison(
                    aspect="overall",
                    output_a_score=0.85,
                    output_b_score=0.85,
                    reasoning="Both are equally good",
                ),
            ],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        comparison = await evaluator.compare(
            output_a="Similar quality output",
            output_b="Similar quality output",
        )

        assert comparison.winner == "tie"
        assert comparison.confidence == 0.8

    @pytest.mark.asyncio
    async def test_compare_with_reference(self, evaluator, mock_agent):
        """Test comparison with reference context."""
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Output A better addresses the question",
            aspect_comparisons=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        comparison = await evaluator.compare(
            output_a="Answer A",
            output_b="Answer B",
            reference="What is the capital of France?",
        )

        assert comparison.winner == "output_a"
        assert comparison.reference == "What is the capital of France?"
        call_args = mock_agent.run.call_args[0][0]
        assert "What is the capital of France?" in call_args

    @pytest.mark.asyncio
    async def test_compare_with_criteria(self, evaluator, mock_agent):
        """Test comparison with criteria."""
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.88,
            reasoning="Output A better meets the criteria",
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
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        comparison = await evaluator.compare(
            output_a="Output A",
            output_b="Output B",
            criteria="accuracy, clarity",
        )

        assert comparison.criteria == "accuracy, clarity"
        assert len(comparison.aspect_scores) == 2
        assert "accuracy" in comparison.aspect_scores
        assert "clarity" in comparison.aspect_scores

    @pytest.mark.asyncio
    async def test_compare_empty_output_a(self, evaluator):
        """Test that empty output_a raises error."""
        with pytest.raises(ValueError, match="output_a cannot be empty"):
            await evaluator.compare(output_a="", output_b="Valid output")

    @pytest.mark.asyncio
    async def test_compare_empty_output_b(self, evaluator):
        """Test that empty output_b raises error."""
        with pytest.raises(ValueError, match="output_b cannot be empty"):
            await evaluator.compare(output_a="Valid output", output_b="")

    @pytest.mark.asyncio
    async def test_compare_error_handling(self, evaluator, mock_agent):
        """Test error handling during comparison."""
        mock_agent.run = AsyncMock(side_effect=Exception("API error"))
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError, match="Pairwise comparison failed"):
            await evaluator.compare(
                output_a="Output A",
                output_b="Output B",
            )

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.85,
            reasoning="Test reasoning",
            aspect_comparisons=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        evaluator.clear_interactions()

        await evaluator.compare(
            output_a="Output A",
            output_b="Output B",
        )

        interactions = evaluator.get_interactions()
        assert len(interactions) == 1
        interaction = interactions[0]
        assert interaction.purpose == "pairwise_comparison_comparison"
        assert interaction.model == "gpt-4o-mini"
        assert "Output A" in interaction.prompt
        assert "Output B" in interaction.prompt
        assert interaction.metadata["evaluator"] == "pairwise_comparison"
        assert interaction.metadata["winner"] == "output_a"

    @pytest.mark.asyncio
    async def test_comparison_result_structure(self, evaluator, mock_agent):
        """Test that ComparisonResult has correct structure."""
        mock_response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Test reasoning",
            aspect_comparisons=[
                AspectComparison(
                    aspect="test_aspect",
                    output_a_score=0.8,
                    output_b_score=0.7,
                    reasoning="Test",
                ),
            ],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        comparison = await evaluator.compare(
            output_a="A",
            output_b="B",
            criteria="test",
        )

        assert isinstance(comparison, ComparisonResult)
        assert comparison.output_a == "A"
        assert comparison.output_b == "B"
        assert comparison.criteria == "test"
        assert comparison.winner == "output_a"
        assert comparison.confidence == 0.9
        assert comparison.processing_time > 0
        assert len(comparison.interactions) == 1

    def test_get_aspect_score(self):
        """Test ComparisonResult.get_aspect_score method."""
        comparison = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            aspect_scores={
                "accuracy": {"output_a": 0.9, "output_b": 0.8},
                "clarity": {"output_a": 0.85, "output_b": 0.9},
            },
            processing_time=1.0,
        )

        assert comparison.get_aspect_score("accuracy", "output_a") == 0.9
        assert comparison.get_aspect_score("accuracy", "output_b") == 0.8
        assert comparison.get_aspect_score("clarity", "output_a") == 0.85
        assert comparison.get_aspect_score("clarity", "output_b") == 0.9
        assert comparison.get_aspect_score("nonexistent", "output_a") is None


class TestPairwiseResponse:
    """Test PairwiseResponse model."""

    def test_response_creation(self):
        """Test creating a response."""
        response = PairwiseResponse(
            winner="output_a",
            confidence=0.9,
            reasoning="Test reasoning",
            aspect_comparisons=[
                AspectComparison(
                    aspect="test",
                    output_a_score=0.8,
                    output_b_score=0.7,
                    reasoning="Test",
                ),
            ],
        )

        assert response.winner == "output_a"
        assert response.confidence == 0.9
        assert len(response.aspect_comparisons) == 1

    def test_response_defaults(self):
        """Test response with defaults."""
        response = PairwiseResponse(
            winner="tie",
            confidence=0.8,
            reasoning="Test",
        )

        assert response.aspect_comparisons == []

    def test_winner_validation(self):
        """Test that winner must be one of the allowed values."""
        # Valid winners
        response1 = PairwiseResponse(
            winner="output_a", confidence=0.9, reasoning="Test"
        )
        assert response1.winner == "output_a"

        response2 = PairwiseResponse(
            winner="output_b", confidence=0.9, reasoning="Test"
        )
        assert response2.winner == "output_b"

        response3 = PairwiseResponse(winner="tie", confidence=0.9, reasoning="Test")
        assert response3.winner == "tie"


class TestAspectComparison:
    """Test AspectComparison model."""

    def test_aspect_comparison_creation(self):
        """Test creating an aspect comparison."""
        aspect = AspectComparison(
            aspect="accuracy",
            output_a_score=0.9,
            output_b_score=0.8,
            reasoning="Output A is more accurate",
        )

        assert aspect.aspect == "accuracy"
        assert aspect.output_a_score == 0.9
        assert aspect.output_b_score == 0.8
        assert "more accurate" in aspect.reasoning.lower()
