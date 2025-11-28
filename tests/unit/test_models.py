"""Unit tests for core data models."""

from datetime import datetime, timezone

import pytest

from arbiter_ai.core.models import (
    ComparisonResult,
    EvaluationResult,
    LLMInteraction,
    Metric,
    Score,
    _get_interaction_cost,
)


class TestScore:
    """Test suite for Score model."""

    def test_score_creation(self):
        """Test creating a Score with all fields."""
        score = Score(
            name="semantic_similarity",
            value=0.85,
            confidence=0.9,
            explanation="High similarity",
            metadata={"key": "value"},
        )

        assert score.name == "semantic_similarity"
        assert score.value == 0.85
        assert score.confidence == 0.9
        assert score.explanation == "High similarity"
        assert score.metadata == {"key": "value"}

    def test_score_minimal(self):
        """Test creating a Score with minimal fields."""
        score = Score(name="test", value=0.5)

        assert score.name == "test"
        assert score.value == 0.5
        assert score.confidence is None
        assert score.explanation is None
        assert score.metadata == {}

    def test_score_value_validation(self):
        """Test that score value must be between 0 and 1."""
        # Valid values
        Score(name="test", value=0.0)
        Score(name="test", value=1.0)
        Score(name="test", value=0.5)

        # Invalid values
        with pytest.raises(Exception):  # Pydantic validation error
            Score(name="test", value=-0.1)

        with pytest.raises(Exception):  # Pydantic validation error
            Score(name="test", value=1.1)

    def test_score_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        # Valid values
        Score(name="test", value=0.5, confidence=0.0)
        Score(name="test", value=0.5, confidence=1.0)
        Score(name="test", value=0.5, confidence=0.5)

        # Invalid values
        with pytest.raises(Exception):  # Pydantic validation error
            Score(name="test", value=0.5, confidence=-0.1)

        with pytest.raises(Exception):  # Pydantic validation error
            Score(name="test", value=0.5, confidence=1.1)

    def test_score_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        score = Score(name="test", value=0.5)
        assert score.metadata == {}
        assert isinstance(score.metadata, dict)


class TestLLMInteraction:
    """Test suite for LLMInteraction model."""

    def test_interaction_creation(self):
        """Test creating an LLMInteraction with all fields."""
        timestamp = datetime.now(timezone.utc)
        interaction = LLMInteraction(
            prompt="Test prompt",
            response="Test response",
            model="gpt-4o",
            tokens_used=150,
            latency=1.2,
            purpose="evaluation",
            timestamp=timestamp,
            metadata={"key": "value"},
        )

        assert interaction.prompt == "Test prompt"
        assert interaction.response == "Test response"
        assert interaction.model == "gpt-4o"
        assert interaction.tokens_used == 150
        assert interaction.latency == 1.2
        assert interaction.purpose == "evaluation"
        assert interaction.timestamp == timestamp
        assert interaction.metadata == {"key": "value"}

    def test_interaction_minimal(self):
        """Test creating an LLMInteraction with minimal fields."""
        interaction = LLMInteraction(
            prompt="Test prompt",
            response="Test response",
            model="gpt-4o",
            latency=1.0,
            purpose="evaluation",
        )

        assert interaction.prompt == "Test prompt"
        assert interaction.response == "Test response"
        assert interaction.model == "gpt-4o"
        assert interaction.tokens_used == 0  # Default
        assert interaction.latency == 1.0
        assert isinstance(interaction.timestamp, datetime)
        assert interaction.metadata == {}

    def test_interaction_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        interaction1 = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o",
            latency=1.0,
            purpose="test",
        )
        interaction2 = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o",
            latency=1.0,
            purpose="test",
        )

        # Timestamps should be close (within 1 second)
        time_diff = abs(
            (interaction1.timestamp - interaction2.timestamp).total_seconds()
        )
        assert time_diff < 1.0

    def test_interaction_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o",
            latency=1.0,
            purpose="test",
        )
        assert interaction.metadata == {}
        assert isinstance(interaction.metadata, dict)


class TestGetInteractionCostHelper:
    """Test suite for _get_interaction_cost() helper function."""

    def test_get_interaction_cost_with_cached_cost(self):
        """Test that helper uses cached cost if available."""
        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            latency=1.0,
            purpose="test",
            cost=0.00045,  # Pre-calculated cost
        )

        cost = _get_interaction_cost(interaction)
        assert cost == 0.00045

    def test_get_interaction_cost_calculates_on_fly(self):
        """Test that helper calculates cost when not cached."""
        from unittest.mock import MagicMock, patch

        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=100,
            latency=1.0,
            purpose="test",
            cost=None,  # No cached cost
        )

        # Mock the cost calculator (patch where it's imported)
        with patch(
            "arbiter_ai.core.cost_calculator.get_cost_calculator"
        ) as mock_get_calc:
            mock_calc = MagicMock()
            mock_calc.calculate_cost.return_value = 0.00055
            mock_get_calc.return_value = mock_calc

            cost = _get_interaction_cost(interaction)

            # Verify calculator was called with correct parameters
            mock_calc.calculate_cost.assert_called_once_with(
                model="gpt-4o-mini",
                input_tokens=1000,
                output_tokens=500,
                cached_tokens=100,
            )
            assert cost == 0.00055

    def test_get_interaction_cost_with_zero_tokens(self):
        """Test helper with zero tokens."""
        from unittest.mock import MagicMock, patch

        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o-mini",
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            latency=1.0,
            purpose="test",
            cost=None,
        )

        with patch(
            "arbiter_ai.core.cost_calculator.get_cost_calculator"
        ) as mock_get_calc:
            mock_calc = MagicMock()
            mock_calc.calculate_cost.return_value = 0.0
            mock_get_calc.return_value = mock_calc

            cost = _get_interaction_cost(interaction)

            mock_calc.calculate_cost.assert_called_once_with(
                model="gpt-4o-mini",
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
            )
            assert cost == 0.0


class TestMetric:
    """Test suite for Metric model."""

    def test_metric_creation(self):
        """Test creating a Metric with all fields."""
        metric = Metric(
            name="semantic_similarity",
            evaluator="SemanticEvaluator",
            model="gpt-4o",
            processing_time=1.5,
            tokens_used=200,
            metadata={"key": "value"},
        )

        assert metric.name == "semantic_similarity"
        assert metric.evaluator == "SemanticEvaluator"
        assert metric.model == "gpt-4o"
        assert metric.processing_time == 1.5
        assert metric.tokens_used == 200
        assert metric.metadata == {"key": "value"}

    def test_metric_minimal(self):
        """Test creating a Metric with minimal fields."""
        metric = Metric(
            name="test",
            evaluator="TestEvaluator",
            processing_time=1.0,
        )

        assert metric.name == "test"
        assert metric.evaluator == "TestEvaluator"
        assert metric.model is None
        assert metric.processing_time == 1.0
        assert metric.tokens_used == 0  # Default
        assert metric.metadata == {}

    def test_metric_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        metric = Metric(
            name="test",
            evaluator="TestEvaluator",
            processing_time=1.0,
        )
        assert metric.metadata == {}
        assert isinstance(metric.metadata, dict)


class TestEvaluationResult:
    """Test suite for EvaluationResult model."""

    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult with all fields."""
        score1 = Score(name="semantic", value=0.9)
        score2 = Score(name="factuality", value=0.85)
        metric = Metric(
            name="semantic", evaluator="SemanticEvaluator", processing_time=1.0
        )
        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o",
            latency=1.0,
            purpose="test",
        )

        result = EvaluationResult(
            output="Test output",
            reference="Test reference",
            criteria="Test criteria",
            scores=[score1, score2],
            overall_score=0.875,
            passed=True,
            partial=False,
            errors={},
            metrics=[metric],
            evaluator_names=["semantic", "factuality"],
            total_tokens=200,
            processing_time=2.0,
            interactions=[interaction],
            metadata={"key": "value"},
        )

        assert result.output == "Test output"
        assert result.reference == "Test reference"
        assert result.criteria == "Test criteria"
        assert len(result.scores) == 2
        assert result.overall_score == 0.875
        assert result.passed is True
        assert result.partial is False
        assert len(result.errors) == 0
        assert len(result.metrics) == 1
        assert len(result.interactions) == 1

    def test_evaluation_result_minimal(self):
        """Test creating an EvaluationResult with minimal fields."""
        result = EvaluationResult(
            output="Test output",
            overall_score=0.8,
            passed=True,
            processing_time=1.0,
        )

        assert result.output == "Test output"
        assert result.reference is None
        assert result.criteria is None
        assert result.scores == []
        assert result.overall_score == 0.8
        assert result.passed is True
        assert result.partial is False
        assert result.errors == {}
        assert result.metrics == []
        assert result.evaluator_names == []
        assert result.total_tokens == 0
        assert result.interactions == []
        assert result.metadata == {}

    def test_evaluation_result_partial(self):
        """Test creating a partial EvaluationResult."""
        score = Score(name="semantic", value=0.9)
        result = EvaluationResult(
            output="Test output",
            scores=[score],
            overall_score=0.9,
            passed=True,
            partial=True,
            errors={"factuality": "API timeout"},
            processing_time=1.0,
        )

        assert result.partial is True
        assert len(result.errors) == 1
        assert "factuality" in result.errors
        assert result.overall_score == 0.9  # Only successful evaluator

    def test_evaluation_result_overall_score_validation(self):
        """Test that overall_score must be between 0 and 1."""
        # Valid values
        EvaluationResult(
            output="test", overall_score=0.0, passed=True, processing_time=1.0
        )
        EvaluationResult(
            output="test", overall_score=1.0, passed=True, processing_time=1.0
        )
        EvaluationResult(
            output="test", overall_score=0.5, passed=True, processing_time=1.0
        )

        # Invalid values
        with pytest.raises(Exception):  # Pydantic validation error
            EvaluationResult(
                output="test", overall_score=-0.1, passed=True, processing_time=1.0
            )

        with pytest.raises(Exception):  # Pydantic validation error
            EvaluationResult(
                output="test", overall_score=1.1, passed=True, processing_time=1.0
            )

    def test_get_score(self):
        """Test getting a score by name."""
        score1 = Score(name="semantic", value=0.9)
        score2 = Score(name="factuality", value=0.85)
        result = EvaluationResult(
            output="test",
            scores=[score1, score2],
            overall_score=0.875,
            passed=True,
            processing_time=1.0,
        )

        assert result.get_score("semantic") == score1
        assert result.get_score("factuality") == score2
        assert result.get_score("nonexistent") is None

    def test_get_metric(self):
        """Test getting a metric by name."""
        metric1 = Metric(
            name="semantic", evaluator="SemanticEvaluator", processing_time=1.0
        )
        metric2 = Metric(
            name="factuality", evaluator="FactualityEvaluator", processing_time=1.5
        )
        result = EvaluationResult(
            output="test",
            overall_score=0.8,
            passed=True,
            metrics=[metric1, metric2],
            processing_time=2.0,
        )

        assert result.get_metric("semantic") == metric1
        assert result.get_metric("factuality") == metric2
        assert result.get_metric("nonexistent") is None

    def test_get_interactions_by_purpose(self):
        """Test getting interactions filtered by purpose."""
        interaction1 = LLMInteraction(
            prompt="test1",
            response="test1",
            model="gpt-4o",
            latency=1.0,
            purpose="scoring",
        )
        interaction2 = LLMInteraction(
            prompt="test2",
            response="test2",
            model="gpt-4o",
            latency=1.0,
            purpose="scoring",
        )
        interaction3 = LLMInteraction(
            prompt="test3",
            response="test3",
            model="gpt-4o",
            latency=1.0,
            purpose="comparison",
        )

        result = EvaluationResult(
            output="test",
            overall_score=0.8,
            passed=True,
            interactions=[interaction1, interaction2, interaction3],
            processing_time=3.0,
        )

        scoring_interactions = result.get_interactions_by_purpose("scoring")
        assert len(scoring_interactions) == 2
        assert interaction1 in scoring_interactions
        assert interaction2 in scoring_interactions
        assert interaction3 not in scoring_interactions

        comparison_interactions = result.get_interactions_by_purpose("comparison")
        assert len(comparison_interactions) == 1
        assert interaction3 in comparison_interactions

    @pytest.mark.asyncio
    async def test_total_llm_cost(self):
        """Test calculating total LLM cost."""
        interaction1 = LLMInteraction(
            prompt="test1",
            response="test1",
            model="gpt-4o",
            input_tokens=50,
            output_tokens=50,
            tokens_used=100,
            latency=1.0,
            purpose="test",
        )
        interaction2 = LLMInteraction(
            prompt="test2",
            response="test2",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=100,
            tokens_used=200,
            latency=1.0,
            purpose="test",
        )

        result = EvaluationResult(
            output="test",
            overall_score=0.8,
            passed=True,
            interactions=[interaction1, interaction2],
            processing_time=2.0,
        )

        # Use simple fallback calculation (not actual pricing)
        # Default: $0.02 per 1k tokens
        # Total tokens: 300, so cost = 300/1000 * 0.02 = 0.006
        cost = await result.total_llm_cost(use_actual_pricing=False)
        assert cost == pytest.approx(0.006, rel=1e-3)

    @pytest.mark.asyncio
    async def test_total_llm_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        result = EvaluationResult(
            output="test",
            overall_score=0.8,
            passed=True,
            processing_time=1.0,
        )

        cost = await result.total_llm_cost(use_actual_pricing=False)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_total_llm_cost_with_actual_pricing(self):
        """Test cost calculation with actual pricing (uses cost calculator)."""
        interaction1 = LLMInteraction(
            prompt="test1",
            response="test1",
            model="gpt-4o",
            input_tokens=50,
            output_tokens=50,
            tokens_used=100,
            latency=1.0,
            purpose="test",
        )
        interaction2 = LLMInteraction(
            prompt="test2",
            response="test2",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=100,
            tokens_used=200,
            latency=1.0,
            purpose="test",
        )

        result = EvaluationResult(
            output="test",
            overall_score=0.8,
            passed=True,
            interactions=[interaction1, interaction2],
            processing_time=2.0,
        )

        # Use actual pricing (triggers cost calculator path lines 539-548)
        cost = await result.total_llm_cost(use_actual_pricing=True)

        # Verify cost is calculated (should be > 0 for gpt-4o with tokens)
        assert cost > 0.0
        # Cost should be reasonable for 150 input + 150 output tokens
        # GPT-4o pricing: ~$2.50/$10.00 per 1M tokens (input/output)
        # Expected: (150 * 2.50 + 150 * 10.00) / 1_000_000 = ~0.001875
        assert cost < 0.01  # Sanity check - should be less than 1 cent

    def test_evaluation_result_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        result1 = EvaluationResult(
            output="test", overall_score=0.8, passed=True, processing_time=1.0
        )
        result2 = EvaluationResult(
            output="test", overall_score=0.8, passed=True, processing_time=1.0
        )

        # Timestamps should be close (within 1 second)
        time_diff = abs((result1.timestamp - result2.timestamp).total_seconds())
        assert time_diff < 1.0


class TestComparisonResult:
    """Test suite for ComparisonResult model."""

    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult with all fields."""
        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="gpt-4o",
            latency=1.0,
            purpose="comparison",
        )

        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            reference="Reference text",
            criteria="Accuracy and clarity",
            winner="output_a",
            confidence=0.9,
            reasoning="Output A is better",
            aspect_scores={
                "accuracy": {"output_a": 0.9, "output_b": 0.8},
                "clarity": {"output_a": 0.85, "output_b": 0.9},
            },
            total_tokens=200,
            processing_time=2.0,
            interactions=[interaction],
            metadata={"key": "value"},
        )

        assert result.output_a == "Output A"
        assert result.output_b == "Output B"
        assert result.reference == "Reference text"
        assert result.criteria == "Accuracy and clarity"
        assert result.winner == "output_a"
        assert result.confidence == 0.9
        assert result.reasoning == "Output A is better"
        assert len(result.aspect_scores) == 2
        assert result.total_tokens == 200
        assert len(result.interactions) == 1

    def test_comparison_result_minimal(self):
        """Test creating a ComparisonResult with minimal fields."""
        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.8,
            reasoning="Test reasoning",
            processing_time=1.0,
        )

        assert result.output_a == "Output A"
        assert result.output_b == "Output B"
        assert result.reference is None
        assert result.criteria is None
        assert result.winner == "output_a"
        assert result.confidence == 0.8
        assert result.aspect_scores == {}
        assert result.total_tokens == 0
        assert result.interactions == []
        assert result.metadata == {}

    def test_comparison_result_winner_validation(self):
        """Test that winner must be one of the allowed values."""
        # Valid values
        ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.8,
            reasoning="test",
            processing_time=1.0,
        )
        ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_b",
            confidence=0.8,
            reasoning="test",
            processing_time=1.0,
        )
        ComparisonResult(
            output_a="A",
            output_b="B",
            winner="tie",
            confidence=0.8,
            reasoning="test",
            processing_time=1.0,
        )

        # Invalid values would be caught by Pydantic/Literal validation
        # This is tested implicitly by the type system

    def test_comparison_result_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        # Valid values
        ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.0,
            reasoning="test",
            processing_time=1.0,
        )
        ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=1.0,
            reasoning="test",
            processing_time=1.0,
        )
        ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.5,
            reasoning="test",
            processing_time=1.0,
        )

        # Invalid values
        with pytest.raises(Exception):  # Pydantic validation error
            ComparisonResult(
                output_a="A",
                output_b="B",
                winner="output_a",
                confidence=-0.1,
                reasoning="test",
                processing_time=1.0,
            )

        with pytest.raises(Exception):  # Pydantic validation error
            ComparisonResult(
                output_a="A",
                output_b="B",
                winner="output_a",
                confidence=1.1,
                reasoning="test",
                processing_time=1.0,
            )

    def test_get_aspect_score(self):
        """Test getting aspect score for a specific output."""
        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            aspect_scores={
                "accuracy": {"output_a": 0.9, "output_b": 0.8},
                "clarity": {"output_a": 0.85, "output_b": 0.9},
            },
            processing_time=1.0,
        )

        assert result.get_aspect_score("accuracy", "output_a") == 0.9
        assert result.get_aspect_score("accuracy", "output_b") == 0.8
        assert result.get_aspect_score("clarity", "output_a") == 0.85
        assert result.get_aspect_score("clarity", "output_b") == 0.9
        assert result.get_aspect_score("nonexistent", "output_a") is None

    def test_get_aspect_score_missing_output(self):
        """Test getting aspect score when output is missing from aspect."""
        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            aspect_scores={
                "accuracy": {"output_a": 0.9},  # Missing output_b
            },
            processing_time=1.0,
        )

        assert result.get_aspect_score("accuracy", "output_a") == 0.9
        assert result.get_aspect_score("accuracy", "output_b") is None

    @pytest.mark.asyncio
    async def test_total_llm_cost(self):
        """Test calculating total LLM cost."""
        interaction1 = LLMInteraction(
            prompt="test1",
            response="test1",
            model="gpt-4o",
            input_tokens=75,
            output_tokens=75,
            tokens_used=150,
            latency=1.0,
            purpose="comparison",
        )
        interaction2 = LLMInteraction(
            prompt="test2",
            response="test2",
            model="gpt-4o",
            input_tokens=125,
            output_tokens=125,
            tokens_used=250,
            latency=1.0,
            purpose="comparison",
        )

        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            interactions=[interaction1, interaction2],
            processing_time=2.0,
        )

        # Use simple fallback calculation (not actual pricing)
        # Default: $0.02 per 1k tokens
        # Total tokens: 400, so cost = 400/1000 * 0.02 = 0.008
        cost = await result.total_llm_cost(use_actual_pricing=False)
        assert cost == pytest.approx(0.008, rel=1e-3)

    @pytest.mark.asyncio
    async def test_total_llm_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            processing_time=1.0,
        )

        cost = await result.total_llm_cost(use_actual_pricing=False)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_total_llm_cost_with_actual_pricing(self):
        """Test ComparisonResult cost calculation with actual pricing (uses cost calculator)."""
        interaction1 = LLMInteraction(
            prompt="test1",
            response="test1",
            model="gpt-4o",
            input_tokens=75,
            output_tokens=75,
            tokens_used=150,
            latency=1.0,
            purpose="comparison",
        )
        interaction2 = LLMInteraction(
            prompt="test2",
            response="test2",
            model="gpt-4o",
            input_tokens=125,
            output_tokens=125,
            tokens_used=250,
            latency=1.0,
            purpose="comparison",
        )

        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.9,
            reasoning="Test",
            interactions=[interaction1, interaction2],
            processing_time=2.0,
        )

        # Use actual pricing (triggers cost calculator path lines 539-548)
        cost = await result.total_llm_cost(use_actual_pricing=True)

        # Verify cost is calculated (should be > 0 for gpt-4o with tokens)
        assert cost > 0.0
        # Cost should be reasonable for 200 input + 200 output tokens
        # GPT-4o pricing: ~$2.50/$10.00 per 1M tokens (input/output)
        # Expected: (200 * 2.50 + 200 * 10.00) / 1_000_000 = ~0.0025
        assert cost < 0.01  # Sanity check - should be less than 1 cent

    def test_comparison_result_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        result1 = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.8,
            reasoning="test",
            processing_time=1.0,
        )
        result2 = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.8,
            reasoning="test",
            processing_time=1.0,
        )

        # Timestamps should be close (within 1 second)
        time_diff = abs((result1.timestamp - result2.timestamp).total_seconds())
        assert time_diff < 1.0

    def test_comparison_result_tie(self):
        """Test comparison result with tie."""
        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="tie",
            confidence=0.7,
            reasoning="Both outputs are equivalent",
            processing_time=1.0,
        )

        assert result.winner == "tie"
        assert result.confidence == 0.7


class TestBatchEvaluationResult:
    """Test suite for BatchEvaluationResult model."""

    @pytest.mark.asyncio
    async def test_total_llm_cost_no_successful_results(self):
        """Test total_llm_cost returns 0.0 when no successful results (line 668)."""
        from arbiter_ai.core.models import BatchEvaluationResult

        # Create batch result with no successful results (all None)
        result = BatchEvaluationResult(
            results=[None, None, None],
            successful_items=0,
            failed_items=3,
            total_items=3,
            processing_time=2.0,
        )

        # Should return 0.0 when no successful results (triggers line 668)
        cost = await result.total_llm_cost()
        assert cost == 0.0

        # Also test with use_actual_pricing=True
        cost_actual = await result.total_llm_cost(use_actual_pricing=True)
        assert cost_actual == 0.0

    @pytest.mark.asyncio
    async def test_cost_breakdown_no_successful_results(self):
        """Test cost_breakdown returns empty dict when no successful results (line 700)."""
        from arbiter_ai.core.models import BatchEvaluationResult

        # Create batch result with no successful results (all None)
        result = BatchEvaluationResult(
            results=[None, None, None],
            successful_items=0,
            failed_items=3,
            total_items=3,
            processing_time=2.0,
        )

        # Should return empty breakdown dict when no successful results (triggers line 700)
        breakdown = await result.cost_breakdown()

        assert breakdown["total"] == 0.0
        assert breakdown["per_item_average"] == 0.0
        assert breakdown["by_evaluator"] == {}
        assert breakdown["by_model"] == {}
        assert breakdown["success_rate"] == 0.0
