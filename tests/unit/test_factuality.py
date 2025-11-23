"""Unit tests for FactualityEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter.core.exceptions import EvaluatorError
from arbiter.evaluators.factuality import FactualityEvaluator, FactualityResponse
from tests.conftest import MockAgentResult


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a FactualityEvaluator instance."""
    return FactualityEvaluator(llm_client=mock_llm_client)


class TestFactualityEvaluator:
    """Test suite for FactualityEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "factuality"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "fact-checker" in prompt.lower()
        assert "claims" in prompt.lower()
        assert "verify" in prompt.lower()

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference."""
        output = "Paris is the capital of France, founded in 1985"
        reference = "Paris is the capital of France, founded in ancient times"
        prompt = evaluator._get_user_prompt(output, reference, None)

        assert output in prompt
        assert reference in prompt
        assert "OUTPUT" in prompt
        assert "REFERENCE" in prompt
        assert "verify" in prompt.lower() or "fact-check" in prompt.lower()

    def test_user_prompt_without_reference_with_criteria(self, evaluator):
        """Test user prompt generation without reference but with criteria."""
        output = "Water boils at 100°C at sea level"
        criteria = "Scientific accuracy"
        prompt = evaluator._get_user_prompt(output, None, criteria)

        assert output in prompt
        assert criteria in prompt
        assert "fact" in prompt.lower() or "accuracy" in prompt.lower()

    def test_user_prompt_without_reference_or_criteria(self, evaluator):
        """Test user prompt generation without reference or criteria (standalone)."""
        output = "The Earth revolves around the Sun"
        prompt = evaluator._get_user_prompt(output, None, None)

        assert output in prompt
        assert "fact" in prompt.lower()
        assert "verify" in prompt.lower() or "assess" in prompt.lower()

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == FactualityResponse

    @pytest.mark.asyncio
    async def test_compute_score_all_factual(self, evaluator):
        """Test score computation with all factual claims."""
        response = FactualityResponse(
            score=1.0,
            confidence=0.9,
            explanation="All claims are factually correct",
            factual_claims=[
                "Paris is the capital of France",
                "The Eiffel Tower is in Paris",
            ],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "factuality"
        assert score.value == 1.0
        assert score.confidence == 0.9
        assert "All claims are factually correct" in score.explanation
        assert "Factual Claims" in score.explanation
        assert score.metadata["factual_count"] == 2
        assert score.metadata["non_factual_count"] == 0
        assert score.metadata["uncertain_count"] == 0

    @pytest.mark.asyncio
    async def test_compute_score_with_hallucinations(self, evaluator):
        """Test score computation with hallucinations."""
        response = FactualityResponse(
            score=0.5,
            confidence=0.85,
            explanation="Some claims are incorrect",
            factual_claims=["Paris is the capital of France"],
            non_factual_claims=["Paris was founded in 1985"],
            uncertain_claims=[],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "factuality"
        assert score.value == 0.5
        assert score.confidence == 0.85
        assert "Some claims are incorrect" in score.explanation
        assert "Non-Factual Claims" in score.explanation
        assert "Paris was founded in 1985" in score.explanation
        assert score.metadata["factual_count"] == 1
        assert score.metadata["non_factual_count"] == 1

    @pytest.mark.asyncio
    async def test_compute_score_with_uncertain_claims(self, evaluator):
        """Test score computation with uncertain claims."""
        response = FactualityResponse(
            score=0.75,
            confidence=0.7,
            explanation="Some claims cannot be verified",
            factual_claims=["The sky is blue", "Grass is green"],
            non_factual_claims=["The moon is made of cheese"],
            uncertain_claims=["Aliens exist on Mars"],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "factuality"
        assert score.value == 0.75
        assert score.confidence == 0.7
        assert "Uncertain Claims" in score.explanation
        assert "Aliens exist on Mars" in score.explanation
        assert score.metadata["factual_count"] == 2
        assert score.metadata["non_factual_count"] == 1
        assert score.metadata["uncertain_count"] == 1

    @pytest.mark.asyncio
    async def test_compute_score_minimal_response(self, evaluator):
        """Test score computation with minimal response (no claims)."""
        response = FactualityResponse(
            score=1.0,
            confidence=0.8,
            explanation="No factual claims detected",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "factuality"
        assert score.value == 1.0
        assert score.confidence == 0.8
        assert score.metadata["factual_count"] == 0
        assert score.metadata["non_factual_count"] == 0
        assert score.metadata["uncertain_count"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference text."""
        mock_response = FactualityResponse(
            score=0.9,
            confidence=0.88,
            explanation="Mostly accurate",
            factual_claims=["Paris is the capital of France"],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
        )

        assert score.value == 0.9
        assert score.confidence == 0.88
        assert len(evaluator.interactions) == 1
        assert evaluator.interactions[0].purpose == "factuality_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_without_reference(self, evaluator, mock_agent):
        """Test evaluation without reference text (standalone fact-checking)."""
        mock_response = FactualityResponse(
            score=1.0,
            confidence=0.9,
            explanation="All claims are verifiable and correct",
            factual_claims=["Water boils at 100°C at sea level"],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Water boils at 100°C at sea level under standard atmospheric pressure"
        )

        assert score.value == 1.0
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_criteria(self, evaluator, mock_agent):
        """Test evaluation with specific criteria."""
        mock_response = FactualityResponse(
            score=0.8,
            confidence=0.85,
            explanation="Scientific facts are mostly accurate",
            factual_claims=["Temperature claim is correct"],
            non_factual_claims=[],
            uncertain_claims=["Need specific pressure value"],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Water boils at 100°C",
            criteria="Scientific accuracy of temperature claims",
        )

        assert score.value == 0.8
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_detects_hallucination(self, evaluator, mock_agent):
        """Test that evaluator detects hallucinations."""
        mock_response = FactualityResponse(
            score=0.5,
            confidence=0.9,
            explanation="Contains hallucinated date",
            factual_claims=["Paris is the capital of France"],
            non_factual_claims=["Paris was founded in 1985"],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Paris is the capital of France, founded in 1985",
            reference="Paris is the capital of France, founded in ancient times",
        )

        assert score.value == 0.5
        assert score.metadata["non_factual_count"] == 1
        assert "Paris was founded in 1985" in score.metadata["non_factual_claims"]

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

        assert "factuality" in str(exc_info.value)
        assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = FactualityResponse(
            score=0.95,
            confidence=0.9,
            explanation="Highly factual",
            factual_claims=["Claim 1", "Claim 2"],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(
            output="Test output with factual claims",
            reference="Test reference",
        )

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"
        assert interaction.purpose == "factuality_evaluation"
        assert interaction.tokens_used == 100
        assert interaction.metadata["evaluator"] == "factuality"
        assert interaction.metadata["has_reference"] is True
        assert interaction.metadata["has_criteria"] is False

    @pytest.mark.asyncio
    async def test_multiple_evaluations_tracked(self, evaluator, mock_agent):
        """Test that multiple evaluations are tracked separately."""
        mock_response = FactualityResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
            factual_claims=["Claim"],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="First output", reference="First ref")
        await evaluator.evaluate(output="Second output", reference="Second ref")

        assert len(evaluator.interactions) == 2
        assert evaluator.interactions[0].prompt != evaluator.interactions[1].prompt

    def test_get_interactions(self, evaluator):
        """Test getting interactions returns a copy."""
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


class TestFactualityResponse:
    """Test suite for FactualityResponse model."""

    def test_response_creation(self):
        """Test creating a FactualityResponse."""
        response = FactualityResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            factual_claims=["Claim 1"],
            non_factual_claims=["False claim 1"],
            uncertain_claims=["Uncertain claim 1"],
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert response.factual_claims == ["Claim 1"]
        assert response.non_factual_claims == ["False claim 1"]
        assert response.uncertain_claims == ["Uncertain claim 1"]

    def test_response_defaults(self):
        """Test FactualityResponse default values."""
        response = FactualityResponse(
            score=0.8,
            explanation="Test",
        )

        assert response.score == 0.8
        assert response.confidence == 0.85  # Default
        assert response.factual_claims == []
        assert response.non_factual_claims == []
        assert response.uncertain_claims == []

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        FactualityResponse(score=0.0, explanation="test")
        FactualityResponse(score=1.0, explanation="test")
        FactualityResponse(score=0.5, explanation="test")

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            FactualityResponse(score=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            FactualityResponse(score=1.1, explanation="test")
