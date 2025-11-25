"""Unit tests for CustomCriteriaEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter_ai.core.exceptions import EvaluatorError
from arbiter_ai.evaluators.custom_criteria import (
    CustomCriteriaEvaluator,
    CustomCriteriaResponse,
    MultiCriteriaResponse,
)
from tests.conftest import MockAgentResult


@pytest.fixture
def mock_agent():
    """Create a mock PydanticAI agent."""
    agent = AsyncMock()
    return agent


@pytest.fixture
def evaluator(mock_llm_client):
    """Create a CustomCriteriaEvaluator instance."""
    return CustomCriteriaEvaluator(llm_client=mock_llm_client)


class TestCustomCriteriaEvaluator:
    """Test suite for CustomCriteriaEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "custom_criteria"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "expert evaluator" in prompt.lower()
        assert "criteria" in prompt.lower()

    def test_user_prompt_with_criteria(self, evaluator):
        """Test user prompt generation with criteria."""
        output = "Test output"
        criteria = "Medical accuracy, HIPAA compliance"
        prompt = evaluator._get_user_prompt(output, None, criteria)

        assert output in prompt
        assert criteria in prompt
        assert "OUTPUT TO EVALUATE" in prompt
        assert "EVALUATION CRITERIA" in prompt

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference."""
        output = "Test output"
        reference = "Reference text"
        criteria = "Accuracy"
        prompt = evaluator._get_user_prompt(output, reference, criteria)

        assert output in prompt
        assert reference in prompt
        assert criteria in prompt
        assert "REFERENCE CONTEXT" in prompt

    def test_user_prompt_requires_criteria(self, evaluator):
        """Test that user prompt raises error without criteria."""
        with pytest.raises(ValueError, match="requires criteria"):
            evaluator._get_user_prompt("output", None, None)

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == CustomCriteriaResponse

    @pytest.mark.asyncio
    async def test_compute_score(self, evaluator):
        """Test score computation from response."""
        response = CustomCriteriaResponse(
            score=0.85,
            confidence=0.9,
            explanation="The output meets most criteria",
            criteria_met=["Medical accuracy", "HIPAA compliance"],
            criteria_not_met=["Appropriate tone"],
        )

        score = await evaluator._compute_score(response)

        assert score.name == "custom_criteria"
        assert score.value == 0.85
        assert score.confidence == 0.9
        assert score.explanation == "The output meets most criteria"
        assert score.metadata["criteria_met"] == [
            "Medical accuracy",
            "HIPAA compliance",
        ]
        assert score.metadata["criteria_not_met"] == ["Appropriate tone"]
        assert score.metadata["criteria_met_count"] == 2
        assert score.metadata["criteria_not_met_count"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_single_criteria(self, evaluator, mock_agent):
        """Test evaluation with single criteria."""
        # Mock the response
        mock_response = CustomCriteriaResponse(
            score=0.92,
            confidence=0.88,
            explanation="The output meets all specified criteria",
            criteria_met=["Medical accuracy", "HIPAA compliance", "Appropriate tone"],
            criteria_not_met=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)

        # Mock client.create_agent
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Run evaluation
        score = await evaluator.evaluate(
            output="Medical advice about diabetes management",
            criteria="Medical accuracy, HIPAA compliance, appropriate tone for patients",
        )

        # Verify results
        assert score.value == 0.92
        assert score.confidence == 0.88
        assert len(score.metadata["criteria_met"]) == 3
        assert len(score.metadata["criteria_not_met"]) == 0

        # Verify agent was called
        assert mock_agent.run.called
        call_args = mock_agent.run.call_args[0][0]
        assert "Medical advice" in call_args
        assert "Medical accuracy" in call_args

        # Verify interactions were tracked
        interactions = evaluator.get_interactions()
        assert len(interactions) == 1
        assert interactions[0].purpose == "custom_criteria_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference text."""
        mock_response = CustomCriteriaResponse(
            score=0.75,
            confidence=0.85,
            explanation="Output partially meets criteria",
            criteria_met=["Accuracy"],
            criteria_not_met=["Completeness"],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Output text",
            reference="Reference text",
            criteria="Accuracy, completeness",
        )

        assert score.value == 0.75
        call_args = mock_agent.run.call_args[0][0]
        assert "Output text" in call_args
        assert "Reference text" in call_args

    @pytest.mark.asyncio
    async def test_evaluate_multi_criteria(self, evaluator, mock_agent):
        """Test multi-criteria evaluation."""
        mock_response = MultiCriteriaResponse(
            criteria_scores={
                "accuracy": 0.9,
                "persuasiveness": 0.8,
                "brand_voice": 0.85,
            },
            overall_score=0.85,
            confidence=0.88,
            explanation="Overall good performance across criteria",
            criteria_details={
                "accuracy": {"met": "yes", "reasoning": "Factually correct"},
                "persuasiveness": {
                    "met": "partial",
                    "reasoning": "Could be more compelling",
                },
                "brand_voice": {"met": "yes", "reasoning": "Matches brand guidelines"},
            },
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        criteria = {
            "accuracy": "Factually correct product information",
            "persuasiveness": "Compelling call-to-action",
            "brand_voice": "Matches company brand guidelines",
        }

        scores = await evaluator.evaluate_multi(
            output="Product description",
            criteria=criteria,
        )

        # Should return one score per criterion plus overall
        assert len(scores) == 4  # 3 criteria + 1 overall

        # Check individual criterion scores
        accuracy_score = next(s for s in scores if s.name == "custom_criteria_accuracy")
        assert accuracy_score.value == 0.9
        assert accuracy_score.metadata["criterion"] == "accuracy"
        assert accuracy_score.metadata["met_status"] == "yes"

        # Check overall score
        overall_score = next(s for s in scores if s.name == "custom_criteria_overall")
        assert overall_score.value == 0.85
        assert overall_score.metadata["criterion_count"] == 3

        # Verify agent was called with multi-criteria prompt
        assert mock_agent.run.called
        call_args = mock_agent.run.call_args[0][0]
        assert "accuracy" in call_args
        assert "persuasiveness" in call_args

    @pytest.mark.asyncio
    async def test_evaluate_multi_empty_criteria(self, evaluator):
        """Test that multi-criteria raises error with empty dict."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await evaluator.evaluate_multi(output="test", criteria={})

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        # Mock agent to raise an error
        mock_agent.run = AsyncMock(side_effect=Exception("API error"))
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError, match="Evaluation failed"):
            await evaluator.evaluate(
                output="Test output",
                criteria="Test criteria",
            )

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = CustomCriteriaResponse(
            score=0.8,
            confidence=0.85,
            explanation="Test explanation",
            criteria_met=[],
            criteria_not_met=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Clear any existing interactions
        evaluator.clear_interactions()

        await evaluator.evaluate(
            output="Test output",
            criteria="Test criteria",
        )

        interactions = evaluator.get_interactions()
        assert len(interactions) == 1
        interaction = interactions[0]
        assert interaction.purpose == "custom_criteria_evaluation"
        assert interaction.model == "gpt-4o-mini"
        assert "Test output" in interaction.prompt
        assert interaction.metadata["evaluator"] == "custom_criteria"

    def test_clear_interactions(self, evaluator):
        """Test clearing interactions."""
        # Add a mock interaction

        from arbiter_ai.core.models import LLMInteraction

        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="test",
            tokens_used=10,
            latency=0.1,
            purpose="test",
        )
        evaluator.interactions.append(interaction)

        assert len(evaluator.interactions) == 1
        evaluator.clear_interactions()
        assert len(evaluator.interactions) == 0


class TestCustomCriteriaResponse:
    """Test CustomCriteriaResponse model."""

    def test_response_creation(self):
        """Test creating a response."""
        response = CustomCriteriaResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            criteria_met=["Criterion 1"],
            criteria_not_met=["Criterion 2"],
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert len(response.criteria_met) == 1
        assert len(response.criteria_not_met) == 1

    def test_response_defaults(self):
        """Test response with defaults."""
        response = CustomCriteriaResponse(
            score=0.8,
            explanation="Test",
        )

        assert response.confidence == 0.85  # Default
        assert response.criteria_met == []  # Default
        assert response.criteria_not_met == []  # Default

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid score
        response = CustomCriteriaResponse(score=0.5, explanation="Test")
        assert response.score == 0.5

        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            CustomCriteriaResponse(score=1.5, explanation="Test")

        with pytest.raises(Exception):
            CustomCriteriaResponse(score=-0.1, explanation="Test")


class TestMultiCriteriaResponse:
    """Test MultiCriteriaResponse model."""

    def test_multi_response_creation(self):
        """Test creating a multi-criteria response."""
        response = MultiCriteriaResponse(
            criteria_scores={"accuracy": 0.9, "tone": 0.8},
            overall_score=0.85,
            confidence=0.88,
            explanation="Overall good",
            criteria_details={
                "accuracy": {"met": "yes", "reasoning": "Good"},
            },
        )

        assert len(response.criteria_scores) == 2
        assert response.overall_score == 0.85
        assert response.confidence == 0.88
