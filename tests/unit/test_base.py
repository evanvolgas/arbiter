"""Unit tests for BasePydanticEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from arbiter_ai.core.exceptions import EvaluatorError
from arbiter_ai.core.models import LLMInteraction, Score
from arbiter_ai.evaluators.base import BasePydanticEvaluator, EvaluatorResponse
from tests.conftest import MockAgentResult


class ConcreteEvaluator(BasePydanticEvaluator):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test_evaluator"

    def _get_system_prompt(self) -> str:
        return "You are a test evaluator."

    def _get_user_prompt(self, output: str, reference: None, criteria: None) -> str:
        return f"Evaluate: {output}"

    def _get_response_type(self):
        return EvaluatorResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = response  # Type: EvaluatorResponse
        return Score(
            name=self.name,
            value=resp.score,
            confidence=resp.confidence,
            explanation=resp.explanation,
        )


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a ConcreteEvaluator instance."""
    return ConcreteEvaluator(llm_client=mock_llm_client)


class TestBasePydanticEvaluator:
    """Test suite for BasePydanticEvaluator."""

    def test_init_with_client(self, mock_llm_client):
        """Test initialization with LLM client."""
        evaluator = ConcreteEvaluator(llm_client=mock_llm_client)
        assert evaluator._llm_client == mock_llm_client
        assert evaluator._model is None
        assert len(evaluator.interactions) == 0

    def test_init_with_model(self):
        """Test initialization with model parameter."""
        evaluator = ConcreteEvaluator(model="gpt-4o-mini")
        assert evaluator._llm_client is None
        assert evaluator._model == "gpt-4o-mini"

    def test_init_without_client_or_model(self):
        """Test that initialization fails without client or model."""
        with pytest.raises(ValueError, match="Must provide either llm_client or model"):
            ConcreteEvaluator()

    def test_name_property(self, evaluator):
        """Test that name property works."""
        assert evaluator.name == "test_evaluator"

    def test_llm_client_property_with_client(self, evaluator):
        """Test llm_client property when client is provided."""
        assert evaluator.llm_client is not None
        assert evaluator.llm_client.model == "gpt-4o-mini"

    def test_llm_client_property_without_client(self):
        """Test llm_client property raises error when client not initialized."""
        evaluator = ConcreteEvaluator(model="gpt-4o-mini")
        with pytest.raises(RuntimeError, match="LLM client not initialized"):
            _ = evaluator.llm_client

    @pytest.mark.asyncio
    async def test_ensure_client_with_existing_client(self, evaluator):
        """Test _ensure_client when client already exists."""
        await evaluator._ensure_client()
        assert evaluator._llm_client is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires async context manager setup")
    async def test_ensure_client_creates_client(self):
        """Test _ensure_client creates client when needed."""
        # This would require mocking LLMManager.get_client
        # Skipping for now as it's complex to set up
        pass

    def test_get_system_prompt(self, evaluator):
        """Test getting system prompt."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_user_prompt(self, evaluator):
        """Test getting user prompt."""
        prompt = evaluator._get_user_prompt("test output", None, None)
        assert isinstance(prompt, str)
        assert "test output" in prompt

    def test_get_response_type(self, evaluator):
        """Test getting response type."""
        response_type = evaluator._get_response_type()
        assert response_type == EvaluatorResponse

    @pytest.mark.asyncio
    async def test_compute_score(self, evaluator):
        """Test computing score from response."""
        response = EvaluatorResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
        )

        score = await evaluator._compute_score(response)

        assert isinstance(score, Score)
        assert score.name == "test_evaluator"
        assert score.value == 0.85
        assert score.confidence == 0.9
        assert score.explanation == "Test explanation"

    @pytest.mark.asyncio
    async def test_evaluate_success(self, evaluator, mock_agent):
        """Test successful evaluation."""
        mock_response = EvaluatorResponse(
            score=0.9,
            confidence=0.85,
            explanation="High score",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(output="Test output")

        assert score.value == 0.9
        assert score.confidence == 0.85
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference."""
        mock_response = EvaluatorResponse(
            score=0.85,
            confidence=0.8,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Test output",
            reference="Test reference",
        )

        assert score.value == 0.85
        interaction = evaluator.interactions[0]
        assert interaction.metadata["has_reference"] is True

    @pytest.mark.asyncio
    async def test_evaluate_with_criteria(self, evaluator, mock_agent):
        """Test evaluation with criteria."""
        mock_response = EvaluatorResponse(
            score=0.8,
            confidence=0.75,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Test output",
            criteria="Test criteria",
        )

        assert score.value == 0.8
        interaction = evaluator.interactions[0]
        assert interaction.metadata["has_criteria"] is True

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        mock_agent.run = AsyncMock(side_effect=Exception("LLM error"))
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError) as exc_info:
            await evaluator.evaluate(output="Test output")

        assert "test_evaluator" in str(exc_info.value)
        assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_interaction_tracking(self, evaluator, mock_agent):
        """Test that interactions are tracked correctly."""
        mock_response = EvaluatorResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="Test output")

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"
        assert interaction.purpose == "test_evaluator_evaluation"
        assert interaction.tokens_used == 100
        assert interaction.metadata["evaluator"] == "test_evaluator"
        assert "Test output" in interaction.prompt

    @pytest.mark.asyncio
    async def test_evaluate_multiple_interactions(self, evaluator, mock_agent):
        """Test that multiple evaluations create multiple interactions."""
        mock_response = EvaluatorResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="First")
        await evaluator.evaluate(output="Second")

        assert len(evaluator.interactions) == 2
        assert "First" in evaluator.interactions[0].prompt
        assert "Second" in evaluator.interactions[1].prompt

    @pytest.mark.asyncio
    async def test_evaluate_token_usage_extraction(self, evaluator, mock_agent):
        """Test that token usage is extracted correctly."""
        mock_response = EvaluatorResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="Test output")

        interaction = evaluator.interactions[0]
        assert interaction.tokens_used == 100

    @pytest.mark.asyncio
    async def test_evaluate_token_usage_fallback(self, evaluator, mock_agent):
        """Test token usage fallback when usage() fails."""
        mock_response = EvaluatorResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        mock_result = MockAgentResult(mock_response)
        # Make usage() raise an exception
        mock_result.usage = MagicMock(side_effect=Exception("Usage error"))
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="Test output")

        interaction = evaluator.interactions[0]
        # Should fallback to 0 when usage() fails
        assert interaction.tokens_used == 0

    def test_get_interactions(self, evaluator):
        """Test getting interactions returns a copy."""
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

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        # This test verifies that BasePydanticEvaluator is abstract
        # by attempting to create a class without implementing abstract methods

        class IncompleteEvaluator(BasePydanticEvaluator):
            @property
            def name(self) -> str:
                return "incomplete"

            # Missing _get_system_prompt
            # Missing _get_user_prompt
            # Missing _compute_score

        # Can't instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            IncompleteEvaluator(llm_client=MagicMock())


class TestEvaluatorResponse:
    """Test suite for EvaluatorResponse model."""

    def test_response_creation(self):
        """Test creating an EvaluatorResponse."""
        response = EvaluatorResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"

    def test_response_defaults(self):
        """Test EvaluatorResponse default values."""
        response = EvaluatorResponse(
            score=0.8,
            explanation="Test",
        )

        assert response.score == 0.8
        assert response.confidence == 0.8  # Default
        assert response.metadata == {}

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        EvaluatorResponse(score=0.0, explanation="test")
        EvaluatorResponse(score=1.0, explanation="test")
        EvaluatorResponse(score=0.5, explanation="test")

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            EvaluatorResponse(score=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            EvaluatorResponse(score=1.1, explanation="test")

    def test_response_metadata(self):
        """Test metadata field."""
        response = EvaluatorResponse(
            score=0.8,
            explanation="Test",
            metadata={"key": "value"},
        )

        assert response.metadata == {"key": "value"}
