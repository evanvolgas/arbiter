"""Unit tests for SemanticEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter.core.exceptions import EvaluatorError
from arbiter.evaluators.semantic import SemanticEvaluator, SemanticResponse
from tests.conftest import MockAgentResult


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a SemanticEvaluator instance."""
    return SemanticEvaluator(llm_client=mock_llm_client)


class TestSemanticEvaluator:
    """Test suite for SemanticEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "semantic"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "semantic similarity" in prompt.lower()
        assert "meaning" in prompt.lower()

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference."""
        output = "Paris is the capital of France"
        reference = "The capital of France is Paris"
        prompt = evaluator._get_user_prompt(output, reference, None)

        assert output in prompt
        assert reference in prompt
        assert "OUTPUT" in prompt
        assert "REFERENCE" in prompt

    def test_user_prompt_without_reference_with_criteria(self, evaluator):
        """Test user prompt generation without reference but with criteria."""
        output = "Test output"
        criteria = "Clarity and coherence"
        prompt = evaluator._get_user_prompt(output, None, criteria)

        assert output in prompt
        assert criteria in prompt
        assert "semantic quality" in prompt.lower()

    def test_user_prompt_without_reference_or_criteria(self, evaluator):
        """Test user prompt generation without reference or criteria."""
        output = "Test output"
        prompt = evaluator._get_user_prompt(output, None, None)

        assert output in prompt
        assert "semantic coherence" in prompt.lower()

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == SemanticResponse

    @pytest.mark.asyncio
    async def test_compute_score(self, evaluator):
        """Test score computation from response."""
        response = SemanticResponse(
            score=0.85,
            confidence=0.9,
            explanation="High similarity",
            key_similarities=["Both mention Paris", "Both mention France"],
            key_differences=["Different word order"],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "semantic"
        assert score.value == 0.85
        assert score.confidence == 0.9
        assert "High similarity" in score.explanation
        assert "Key Similarities" in score.explanation
        assert "Key Differences" in score.explanation
        assert score.metadata["similarities_count"] == 2
        assert score.metadata["differences_count"] == 1

    @pytest.mark.asyncio
    async def test_compute_score_minimal_response(self, evaluator):
        """Test score computation with minimal response (no similarities/differences)."""
        response = SemanticResponse(
            score=0.75,
            confidence=0.8,
            explanation="Moderate similarity",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "semantic"
        assert score.value == 0.75
        assert score.confidence == 0.8
        assert "Moderate similarity" in score.explanation
        assert score.metadata["similarities_count"] == 0
        assert score.metadata["differences_count"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference text."""
        mock_response = SemanticResponse(
            score=0.92,
            confidence=0.88,
            explanation="Very similar meaning",
            key_similarities=["Both mention capital city"],
            key_differences=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
        )

        assert score.value == 0.92
        assert score.confidence == 0.88
        assert len(evaluator.interactions) == 1
        assert evaluator.interactions[0].purpose == "semantic_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_without_reference(self, evaluator, mock_agent):
        """Test evaluation without reference text."""
        mock_response = SemanticResponse(
            score=0.8,
            confidence=0.75,
            explanation="Good semantic coherence",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="This is a coherent text with clear meaning."
        )

        assert score.value == 0.8
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_criteria(self, evaluator, mock_agent):
        """Test evaluation with criteria."""
        mock_response = SemanticResponse(
            score=0.85,
            confidence=0.82,
            explanation="Meets criteria",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Test output",
            criteria="Clarity and accuracy",
        )

        assert score.value == 0.85
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        mock_agent.run = AsyncMock(side_effect=Exception("LLM API error"))
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError) as exc_info:
            await evaluator.evaluate(
                output="Test output",
                reference="Test reference",
            )

        assert "semantic" in str(exc_info.value)
        assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test explanation",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(
            output="Test output",
            reference="Test reference",
        )

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"
        assert interaction.purpose == "semantic_evaluation"
        assert interaction.tokens_used == 100
        assert interaction.metadata["evaluator"] == "semantic"
        assert interaction.metadata["has_reference"] is True
        assert interaction.metadata["has_criteria"] is False

    @pytest.mark.asyncio
    async def test_multiple_evaluations_tracked(self, evaluator, mock_agent):
        """Test that multiple evaluations are tracked separately."""
        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="First", reference="First ref")
        await evaluator.evaluate(output="Second", reference="Second ref")

        assert len(evaluator.interactions) == 2
        assert evaluator.interactions[0].prompt != evaluator.interactions[1].prompt

    def test_get_interactions(self, evaluator):
        """Test getting interactions returns a copy."""
        # Add a mock interaction
        from arbiter.core.models import LLMInteraction

        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o-mini",
            tokens_used=100,
            latency=0.5,
            purpose="test",
        )
        evaluator.interactions.append(interaction)

        interactions = evaluator.get_interactions()
        assert len(interactions) == 1
        assert interactions[0] == interaction

        # Modifying returned list shouldn't affect internal list
        interactions.append("should not appear")
        assert len(evaluator.interactions) == 1

    def test_clear_interactions(self, evaluator):
        """Test clearing interactions."""
        from arbiter.core.models import LLMInteraction

        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o-mini",
            tokens_used=100,
            latency=0.5,
            purpose="test",
        )
        evaluator.interactions.append(interaction)

        assert len(evaluator.interactions) == 1
        evaluator.clear_interactions()
        assert len(evaluator.interactions) == 0


class TestSemanticResponse:
    """Test suite for SemanticResponse model."""

    def test_response_creation(self):
        """Test creating a SemanticResponse."""
        response = SemanticResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            key_similarities=["Similarity 1"],
            key_differences=["Difference 1"],
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert response.key_similarities == ["Similarity 1"]
        assert response.key_differences == ["Difference 1"]

    def test_response_defaults(self):
        """Test SemanticResponse default values."""
        response = SemanticResponse(
            score=0.8,
            explanation="Test",
        )

        assert response.score == 0.8
        assert response.confidence == 0.85  # Default
        assert response.key_similarities == []
        assert response.key_differences == []

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        SemanticResponse(score=0.0, explanation="test")
        SemanticResponse(score=1.0, explanation="test")
        SemanticResponse(score=0.5, explanation="test")

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            SemanticResponse(score=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            SemanticResponse(score=1.1, explanation="test")
