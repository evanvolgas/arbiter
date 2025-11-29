"""Unit tests for RelevanceEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter_ai.core.exceptions import EvaluatorError
from arbiter_ai.evaluators.relevance import RelevanceEvaluator, RelevanceResponse
from tests.conftest import MockAgentResult


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a RelevanceEvaluator instance."""
    return RelevanceEvaluator(llm_client=mock_llm_client)


class TestRelevanceEvaluator:
    """Test suite for RelevanceEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "relevance"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "relevance" in prompt.lower() or "query" in prompt.lower()
        assert "addressed" in prompt.lower() or "missing" in prompt.lower()

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference (query)."""
        output = "Paris is the capital of France"
        reference = "What is the capital of France?"
        prompt = evaluator._get_user_prompt(output, reference, None)

        assert output in prompt
        assert reference in prompt
        assert "QUERY" in prompt
        assert "OUTPUT" in prompt
        assert "relevance" in prompt.lower() or "addressed" in prompt.lower()

    def test_user_prompt_without_reference_or_criteria(self, evaluator):
        """Test user prompt generation without reference or criteria (should error)."""
        output = "Paris is the capital of France"
        prompt = evaluator._get_user_prompt(output, None, None)

        assert "Error" in prompt or "requires" in prompt.lower()

    def test_user_prompt_with_criteria(self, evaluator):
        """Test user prompt generation with criteria only."""
        output = "Paris is the capital of France with a population of 2 million"
        criteria = "Focus on answering geographical questions directly"
        prompt = evaluator._get_user_prompt(output, None, criteria)

        assert output in prompt
        assert criteria in prompt
        assert "relevance" in prompt.lower()

    def test_user_prompt_with_reference_and_criteria(self, evaluator):
        """Test user prompt generation with both reference and criteria."""
        output = "Paris is the capital"
        reference = "What is the capital of France?"
        criteria = "Direct and complete answers"
        prompt = evaluator._get_user_prompt(output, reference, criteria)

        assert output in prompt
        assert reference in prompt
        assert criteria in prompt

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == RelevanceResponse

    @pytest.mark.asyncio
    async def test_compute_score_fully_relevant(self, evaluator):
        """Test score computation with fully relevant output."""
        response = RelevanceResponse(
            score=1.0,
            confidence=0.9,
            explanation="Output fully addresses the query",
            addressed_points=["Capital city identified", "Country specified"],
            missing_points=[],
            irrelevant_content=[],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "relevance"
        assert score.value == 1.0
        assert score.confidence == 0.9
        assert "fully addresses" in score.explanation
        assert "Addressed Points" in score.explanation
        assert score.metadata["addressed_count"] == 2
        assert score.metadata["missing_count"] == 0
        assert score.metadata["irrelevant_count"] == 0
        assert score.metadata["total_points"] == 2

    @pytest.mark.asyncio
    async def test_compute_score_with_missing_points(self, evaluator):
        """Test score computation with missing points."""
        response = RelevanceResponse(
            score=0.5,
            confidence=0.85,
            explanation="Partially addresses the query",
            addressed_points=["Capital city mentioned"],
            missing_points=["Population not provided"],
            irrelevant_content=[],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "relevance"
        assert score.value == 0.5
        assert score.confidence == 0.85
        assert "Missing Points" in score.explanation
        assert "Population not provided" in score.explanation
        assert score.metadata["addressed_count"] == 1
        assert score.metadata["missing_count"] == 1
        assert score.metadata["total_points"] == 2

    @pytest.mark.asyncio
    async def test_compute_score_with_irrelevant_content(self, evaluator):
        """Test score computation with irrelevant content."""
        response = RelevanceResponse(
            score=0.7,
            confidence=0.8,
            explanation="Mostly relevant with some off-topic content",
            addressed_points=["Main question answered"],
            missing_points=[],
            irrelevant_content=["Unrelated historical facts"],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "relevance"
        assert score.value == 0.7
        assert "Irrelevant Content" in score.explanation
        assert "Unrelated historical facts" in score.explanation
        assert score.metadata["irrelevant_count"] == 1

    @pytest.mark.asyncio
    async def test_compute_score_completely_irrelevant(self, evaluator):
        """Test score computation with completely irrelevant output."""
        response = RelevanceResponse(
            score=0.0,
            confidence=0.95,
            explanation="Output does not address the query at all",
            addressed_points=[],
            missing_points=["Capital city", "Country name", "Geographic information"],
            irrelevant_content=["Sports statistics", "Weather data"],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "relevance"
        assert score.value == 0.0
        assert score.metadata["addressed_count"] == 0
        assert score.metadata["missing_count"] == 3
        assert score.metadata["irrelevant_count"] == 2

    @pytest.mark.asyncio
    async def test_compute_score_minimal_response(self, evaluator):
        """Test score computation with minimal response (no points)."""
        response = RelevanceResponse(
            score=1.0,
            confidence=0.8,
            explanation="No specific points to evaluate",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "relevance"
        assert score.value == 1.0
        assert score.metadata["addressed_count"] == 0
        assert score.metadata["missing_count"] == 0
        assert score.metadata["total_points"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_query(self, evaluator, mock_agent):
        """Test evaluation with query/reference."""
        mock_response = RelevanceResponse(
            score=0.9,
            confidence=0.88,
            explanation="Highly relevant",
            addressed_points=["Question answered directly"],
            missing_points=[],
            irrelevant_content=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Paris is the capital of France",
            reference="What is the capital of France?",
        )

        assert score.value == 0.9
        assert score.confidence == 0.88
        assert len(evaluator.interactions) == 1
        assert evaluator.interactions[0].purpose == "relevance_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_query_alignment(self, evaluator, mock_agent):
        """Test evaluation for query-output alignment."""
        mock_response = RelevanceResponse(
            score=0.67,
            confidence=0.85,
            explanation="Partially relevant",
            addressed_points=["Height mentioned", "Location mentioned"],
            missing_points=["Construction year not provided"],
            irrelevant_content=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="The Eiffel Tower is 300 meters tall and located in Paris",
            reference="How tall is the Eiffel Tower and when was it built?",
        )

        assert score.value == 0.67
        assert score.metadata["addressed_count"] == 2
        assert score.metadata["missing_count"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_criteria(self, evaluator, mock_agent):
        """Test evaluation with specific criteria."""
        mock_response = RelevanceResponse(
            score=0.8,
            confidence=0.85,
            explanation="Meets most criteria",
            addressed_points=["Direct answer provided"],
            missing_points=["Examples not included"],
            irrelevant_content=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Python is a high-level programming language",
            criteria="Provide direct answer with examples",
        )

        assert score.value == 0.8
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_detects_irrelevant_content(self, evaluator, mock_agent):
        """Test that evaluator detects irrelevant content."""
        mock_response = RelevanceResponse(
            score=0.4,
            confidence=0.9,
            explanation="Contains significant off-topic content",
            addressed_points=["Capital mentioned"],
            missing_points=[],
            irrelevant_content=[
                "Historical tourism information",
                "Restaurant recommendations",
            ],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Paris is the capital. Also, Paris has great restaurants and historical sites to visit.",
            reference="What is the capital of France?",
        )

        assert score.value == 0.4
        assert score.metadata["irrelevant_count"] == 2
        assert "Historical tourism information" in score.metadata["irrelevant_content"]

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        mock_agent.run = AsyncMock(side_effect=Exception("LLM API error"))
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError) as exc_info:
            await evaluator.evaluate(
                output="Test output",
                reference="Test query",
            )

        assert "relevance" in str(exc_info.value)
        assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = RelevanceResponse(
            score=0.95,
            confidence=0.9,
            explanation="Highly relevant",
            addressed_points=["Point 1", "Point 2"],
            missing_points=[],
            irrelevant_content=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(
            output="Test output with relevant content",
            reference="Test query",
        )

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"
        assert interaction.purpose == "relevance_evaluation"
        assert interaction.tokens_used == 100
        assert interaction.metadata["evaluator"] == "relevance"
        assert interaction.metadata["has_reference"] is True
        assert interaction.metadata["has_criteria"] is False

    @pytest.mark.asyncio
    async def test_multiple_evaluations_tracked(self, evaluator, mock_agent):
        """Test that multiple evaluations are tracked separately."""
        mock_response = RelevanceResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
            addressed_points=["Point"],
            missing_points=[],
            irrelevant_content=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="First output", reference="First query")
        await evaluator.evaluate(output="Second output", reference="Second query")

        assert len(evaluator.interactions) == 2
        assert evaluator.interactions[0].prompt != evaluator.interactions[1].prompt

    def test_get_interactions(self, evaluator):
        """Test getting interactions returns a copy."""
        from arbiter_ai.core.models import LLMInteraction

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
        from arbiter_ai.core.models import LLMInteraction

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


class TestRelevanceResponse:
    """Test suite for RelevanceResponse model."""

    def test_response_creation(self):
        """Test creating a RelevanceResponse."""
        response = RelevanceResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            addressed_points=["Point 1"],
            missing_points=["Point 2"],
            irrelevant_content=["Off-topic 1"],
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert response.addressed_points == ["Point 1"]
        assert response.missing_points == ["Point 2"]
        assert response.irrelevant_content == ["Off-topic 1"]

    def test_response_defaults(self):
        """Test RelevanceResponse default values."""
        response = RelevanceResponse(
            score=0.8,
            explanation="Test explanation for relevance assessment",
            addressed_points=[
                "Key point was addressed"
            ],  # Need points for non-extreme scores
        )

        assert response.score == 0.8
        assert response.confidence == 0.85  # Default
        assert len(response.addressed_points) == 1
        assert response.missing_points == []
        assert response.irrelevant_content == []

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores (extremes don't need points)
        RelevanceResponse(score=0.0, explanation="All points missing")
        RelevanceResponse(score=1.0, explanation="All points addressed")
        # Non-extreme scores need points
        RelevanceResponse(
            score=0.5, explanation="Some points addressed", addressed_points=["Point 1"]
        )

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            RelevanceResponse(score=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            RelevanceResponse(score=1.1, explanation="test")

    def test_response_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        # Valid confidence (low confidence doesn't need points)
        RelevanceResponse(score=0.8, confidence=0.0, explanation="Low confidence")
        # High confidence needs points for non-extreme scores
        RelevanceResponse(score=1.0, confidence=1.0, explanation="Fully relevant")
        RelevanceResponse(
            score=0.8,
            confidence=0.75,
            explanation="Mostly relevant",
            addressed_points=["Point 1"],
        )

        # Invalid confidence
        with pytest.raises(Exception):  # Pydantic validation error
            RelevanceResponse(score=0.8, confidence=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            RelevanceResponse(score=0.8, confidence=1.5, explanation="test")

    def test_response_with_complex_points(self):
        """Test response with multiple addressed/missing/irrelevant points."""
        addressed = ["Capital identified", "Country specified", "Location accurate"]
        missing = ["Population data", "Area size"]
        irrelevant = ["Historical trivia", "Tourist attractions", "Weather info"]

        response = RelevanceResponse(
            score=0.6,
            explanation="Mixed relevance",
            addressed_points=addressed,
            missing_points=missing,
            irrelevant_content=irrelevant,
        )

        assert len(response.addressed_points) == 3
        assert len(response.missing_points) == 2
        assert len(response.irrelevant_content) == 3
        assert all(point in response.addressed_points for point in addressed)
        assert all(point in response.missing_points for point in missing)
        assert all(content in response.irrelevant_content for content in irrelevant)
