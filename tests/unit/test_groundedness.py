"""Unit tests for GroundednessEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter.core.exceptions import EvaluatorError
from arbiter.evaluators.groundedness import GroundednessEvaluator, GroundednessResponse
from tests.conftest import MockAgentResult


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a GroundednessEvaluator instance."""
    return GroundednessEvaluator(llm_client=mock_llm_client)


class TestGroundednessEvaluator:
    """Test suite for GroundednessEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "groundedness"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert (
            "source attribution" in prompt.lower() or "groundedness" in prompt.lower()
        )
        assert "statements" in prompt.lower()
        assert "grounded" in prompt.lower()

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference."""
        output = "Paris is the capital of France with population of 2.2M"
        reference = (
            "Paris, the capital city of France, has a population of 2.16 million"
        )
        prompt = evaluator._get_user_prompt(output, reference, None)

        assert output in prompt
        assert reference in prompt
        assert "OUTPUT" in prompt
        assert "SOURCE" in prompt or "REFERENCE" in prompt
        assert "grounded" in prompt.lower() or "attribution" in prompt.lower()

    def test_user_prompt_without_reference(self, evaluator):
        """Test user prompt generation without reference (should error)."""
        output = "Paris is the capital of France"
        prompt = evaluator._get_user_prompt(output, None, None)

        assert "Error" in prompt or "requires reference" in prompt.lower()

    def test_user_prompt_with_criteria(self, evaluator):
        """Test user prompt generation with reference and criteria."""
        output = "The Eiffel Tower is 324 meters tall"
        reference = "The Eiffel Tower stands at 300 meters in height"
        criteria = "Focus on numerical accuracy"
        prompt = evaluator._get_user_prompt(output, reference, criteria)

        assert output in prompt
        assert reference in prompt
        assert criteria in prompt
        assert "grounded" in prompt.lower()

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == GroundednessResponse

    @pytest.mark.asyncio
    async def test_compute_score_fully_grounded(self, evaluator):
        """Test score computation with fully grounded output."""
        response = GroundednessResponse(
            score=1.0,
            confidence=0.9,
            explanation="All statements are grounded in sources",
            grounded_statements=[
                "Paris is the capital of France",
                "The Eiffel Tower is in Paris",
            ],
            ungrounded_statements=[],
            citations={
                "Paris is the capital of France": "Paris, the capital city of France...",
                "The Eiffel Tower is in Paris": "The Eiffel Tower, located in Paris...",
            },
        )

        score = await evaluator._compute_score(response)

        assert score.name == "groundedness"
        assert score.value == 1.0
        assert score.confidence == 0.9
        assert "All statements are grounded" in score.explanation
        assert "Grounded Statements" in score.explanation
        assert "Citation Mapping" in score.explanation
        assert score.metadata["grounded_count"] == 2
        assert score.metadata["ungrounded_count"] == 0
        assert score.metadata["total_statements"] == 2

    @pytest.mark.asyncio
    async def test_compute_score_with_ungrounded_statements(self, evaluator):
        """Test score computation with ungrounded statements."""
        response = GroundednessResponse(
            score=0.5,
            confidence=0.85,
            explanation="Some statements lack source support",
            grounded_statements=["Paris is the capital of France"],
            ungrounded_statements=["Paris has population of 2.2M"],
            citations={
                "Paris is the capital of France": "Paris, the capital city of France..."
            },
        )

        score = await evaluator._compute_score(response)

        assert score.name == "groundedness"
        assert score.value == 0.5
        assert score.confidence == 0.85
        assert "Ungrounded Statements" in score.explanation
        assert "Paris has population of 2.2M" in score.explanation
        assert score.metadata["grounded_count"] == 1
        assert score.metadata["ungrounded_count"] == 1
        assert score.metadata["total_statements"] == 2

    @pytest.mark.asyncio
    async def test_compute_score_completely_ungrounded(self, evaluator):
        """Test score computation with completely ungrounded output."""
        response = GroundednessResponse(
            score=0.0,
            confidence=0.95,
            explanation="No statements are supported by sources",
            grounded_statements=[],
            ungrounded_statements=[
                "Invented fact 1",
                "Invented fact 2",
                "Hallucinated claim",
            ],
            citations={},
        )

        score = await evaluator._compute_score(response)

        assert score.name == "groundedness"
        assert score.value == 0.0
        assert score.confidence == 0.95
        assert "Ungrounded Statements" in score.explanation
        assert "Invented fact 1" in score.explanation
        assert score.metadata["grounded_count"] == 0
        assert score.metadata["ungrounded_count"] == 3
        assert score.metadata["total_statements"] == 3
        assert len(score.metadata["citations"]) == 0

    @pytest.mark.asyncio
    async def test_compute_score_minimal_response(self, evaluator):
        """Test score computation with minimal response (no statements)."""
        response = GroundednessResponse(
            score=1.0,
            confidence=0.8,
            explanation="No statements detected",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "groundedness"
        assert score.value == 1.0
        assert score.confidence == 0.8
        assert score.metadata["grounded_count"] == 0
        assert score.metadata["ungrounded_count"] == 0
        assert score.metadata["total_statements"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference text."""
        mock_response = GroundednessResponse(
            score=0.9,
            confidence=0.88,
            explanation="Mostly grounded",
            grounded_statements=["Paris is the capital of France"],
            ungrounded_statements=[],
            citations={"Paris is the capital of France": "Source text..."},
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
        assert evaluator.interactions[0].purpose == "groundedness_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_rag_validation(self, evaluator, mock_agent):
        """Test RAG output validation with multiple source documents."""
        mock_response = GroundednessResponse(
            score=0.67,
            confidence=0.85,
            explanation="Partially grounded in sources",
            grounded_statements=[
                "The Eiffel Tower is in Paris",
                "It was built in 1889",
            ],
            ungrounded_statements=["It is 324 meters tall"],
            citations={
                "The Eiffel Tower is in Paris": "Located in Paris, France...",
                "It was built in 1889": "Constructed in 1889 for the World's Fair...",
            },
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="The Eiffel Tower is in Paris, built in 1889, and is 324 meters tall",
            reference="The Eiffel Tower, located in Paris, was constructed in 1889 for the World's Fair. It stands at 300 meters in height.",
        )

        assert score.value == 0.67
        assert score.metadata["grounded_count"] == 2
        assert score.metadata["ungrounded_count"] == 1
        assert len(score.metadata["citations"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_with_criteria(self, evaluator, mock_agent):
        """Test evaluation with specific criteria."""
        mock_response = GroundednessResponse(
            score=0.8,
            confidence=0.85,
            explanation="Numerical facts are mostly grounded",
            grounded_statements=["Built in 1889"],
            ungrounded_statements=["Height 324 meters"],
            citations={"Built in 1889": "Constructed in 1889..."},
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Built in 1889, height 324 meters",
            reference="Constructed in 1889, stands at 300 meters",
            criteria="Focus on numerical accuracy",
        )

        assert score.value == 0.8
        assert len(evaluator.interactions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_detects_hallucination(self, evaluator, mock_agent):
        """Test that evaluator detects hallucinated content."""
        mock_response = GroundednessResponse(
            score=0.33,
            confidence=0.9,
            explanation="Contains hallucinated information",
            grounded_statements=["Paris is the capital of France"],
            ungrounded_statements=["Population of 5 million", "Founded in 1200 AD"],
            citations={"Paris is the capital of France": "Paris, the capital..."},
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Paris is the capital of France with population of 5 million, founded in 1200 AD",
            reference="Paris, the capital city of France, has a population of 2.16 million and was founded in ancient times",
        )

        assert score.value == 0.33
        assert score.metadata["ungrounded_count"] == 2
        assert "Population of 5 million" in score.metadata["ungrounded_statements"]
        assert "Founded in 1200 AD" in score.metadata["ungrounded_statements"]

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

        assert "groundedness" in str(exc_info.value)
        assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = GroundednessResponse(
            score=0.95,
            confidence=0.9,
            explanation="Highly grounded",
            grounded_statements=["Statement 1", "Statement 2"],
            ungrounded_statements=[],
            citations={"Statement 1": "Source 1", "Statement 2": "Source 2"},
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(
            output="Test output with grounded statements",
            reference="Test source documents",
        )

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"
        assert interaction.purpose == "groundedness_evaluation"
        assert interaction.tokens_used == 100
        assert interaction.metadata["evaluator"] == "groundedness"
        assert interaction.metadata["has_reference"] is True
        assert interaction.metadata["has_criteria"] is False

    @pytest.mark.asyncio
    async def test_multiple_evaluations_tracked(self, evaluator, mock_agent):
        """Test that multiple evaluations are tracked separately."""
        mock_response = GroundednessResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
            grounded_statements=["Statement"],
            ungrounded_statements=[],
            citations={"Statement": "Source"},
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="First output", reference="First source")
        await evaluator.evaluate(output="Second output", reference="Second source")

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


class TestGroundednessResponse:
    """Test suite for GroundednessResponse model."""

    def test_response_creation(self):
        """Test creating a GroundednessResponse."""
        response = GroundednessResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            grounded_statements=["Statement 1"],
            ungrounded_statements=["Statement 2"],
            citations={"Statement 1": "Source text"},
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert response.grounded_statements == ["Statement 1"]
        assert response.ungrounded_statements == ["Statement 2"]
        assert response.citations == {"Statement 1": "Source text"}

    def test_response_defaults(self):
        """Test GroundednessResponse default values."""
        response = GroundednessResponse(
            score=0.8,
            explanation="Test",
        )

        assert response.score == 0.8
        assert response.confidence == 0.85  # Default
        assert response.grounded_statements == []
        assert response.ungrounded_statements == []
        assert response.citations == {}

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        GroundednessResponse(score=0.0, explanation="test")
        GroundednessResponse(score=1.0, explanation="test")
        GroundednessResponse(score=0.5, explanation="test")

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            GroundednessResponse(score=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            GroundednessResponse(score=1.1, explanation="test")

    def test_response_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        # Valid confidence
        GroundednessResponse(score=0.8, confidence=0.0, explanation="test")
        GroundednessResponse(score=0.8, confidence=1.0, explanation="test")
        GroundednessResponse(score=0.8, confidence=0.75, explanation="test")

        # Invalid confidence
        with pytest.raises(Exception):  # Pydantic validation error
            GroundednessResponse(score=0.8, confidence=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            GroundednessResponse(score=0.8, confidence=1.5, explanation="test")

    def test_response_with_complex_citations(self):
        """Test response with multiple citations."""
        citations = {
            "Paris is the capital": "Paris, the capital city...",
            "Population is 2.16M": "The city has 2.16 million inhabitants...",
            "Built in 1889": "Constructed in 1889 for the exposition...",
        }

        response = GroundednessResponse(
            score=0.9,
            explanation="Well grounded",
            grounded_statements=list(citations.keys()),
            citations=citations,
        )

        assert len(response.citations) == 3
        assert response.grounded_statements == list(citations.keys())
        assert all(stmt in response.citations for stmt in response.grounded_statements)
