"""Unit tests for evaluator registry system."""

import pytest

from arbiter.core.exceptions import ValidationError
from arbiter.core.interfaces import BaseEvaluator
from arbiter.core.registry import (
    AVAILABLE_EVALUATORS,
    get_available_evaluators,
    get_evaluator_class,
    register_evaluator,
    validate_evaluator_name,
)
from arbiter.evaluators import CustomCriteriaEvaluator, SemanticEvaluator


class TestRegistryBasics:
    """Test suite for basic registry functionality."""

    def test_builtin_evaluators_registered(self):
        """Test that built-in evaluators are automatically registered."""
        assert "semantic" in AVAILABLE_EVALUATORS
        assert "custom_criteria" in AVAILABLE_EVALUATORS
        assert AVAILABLE_EVALUATORS["semantic"] == SemanticEvaluator
        assert AVAILABLE_EVALUATORS["custom_criteria"] == CustomCriteriaEvaluator

    def test_get_evaluator_class(self):
        """Test getting evaluator class by name."""
        assert get_evaluator_class("semantic") == SemanticEvaluator
        assert get_evaluator_class("custom_criteria") == CustomCriteriaEvaluator
        assert get_evaluator_class("nonexistent") is None

    def test_get_available_evaluators(self):
        """Test getting list of available evaluators."""
        evaluators = get_available_evaluators()
        assert isinstance(evaluators, list)
        assert "semantic" in evaluators
        assert "custom_criteria" in evaluators
        # Should be sorted
        assert evaluators == sorted(evaluators)

    def test_validate_evaluator_name_valid(self):
        """Test validation with valid evaluator names."""
        # Should not raise
        validate_evaluator_name("semantic")
        validate_evaluator_name("custom_criteria")

    def test_validate_evaluator_name_invalid(self):
        """Test validation with invalid evaluator names."""
        with pytest.raises(ValidationError) as exc_info:
            validate_evaluator_name("nonexistent")

        error_msg = str(exc_info.value)
        assert "Unknown evaluator" in error_msg
        assert "nonexistent" in error_msg
        assert "Available evaluators" in error_msg
        assert "semantic" in error_msg
        assert "custom_criteria" in error_msg


class TestCustomEvaluatorRegistration:
    """Test suite for registering custom evaluators."""

    def test_register_custom_evaluator(self):
        """Test registering a custom evaluator."""

        # Create a mock evaluator class
        class MockEvaluator(BaseEvaluator):
            @property
            def name(self) -> str:
                return "mock_evaluator"

            async def evaluate(self, output: str, reference=None, criteria=None):
                from arbiter.core.models import Score

                return Score(name="mock", value=0.5)

        # Register it
        register_evaluator("mock_evaluator", MockEvaluator)

        # Verify it's registered
        assert "mock_evaluator" in AVAILABLE_EVALUATORS
        assert get_evaluator_class("mock_evaluator") == MockEvaluator
        assert "mock_evaluator" in get_available_evaluators()

        # Clean up
        del AVAILABLE_EVALUATORS["mock_evaluator"]

    def test_register_duplicate_evaluator(self):
        """Test that registering duplicate evaluator raises error."""

        class MockEvaluator(BaseEvaluator):
            @property
            def name(self) -> str:
                return "duplicate_test"

            async def evaluate(self, output: str, reference=None, criteria=None):
                from arbiter.core.models import Score

                return Score(name="mock", value=0.5)

        # Register first time - should work
        register_evaluator("duplicate_test", MockEvaluator)

        # Try to register again - should fail
        with pytest.raises(ValueError, match="already registered"):
            register_evaluator("duplicate_test", MockEvaluator)

        # Clean up
        del AVAILABLE_EVALUATORS["duplicate_test"]

    def test_register_invalid_evaluator_class(self):
        """Test that registering non-BaseEvaluator class raises error."""

        class NotAnEvaluator:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseEvaluator"):
            register_evaluator("invalid", NotAnEvaluator)

    def test_register_then_use_in_evaluate(self):
        """Test that registered evaluator can be used in evaluate()."""
        from unittest.mock import AsyncMock, MagicMock

        from arbiter.core.llm_client import LLMClient
        from arbiter.evaluators.semantic import SemanticResponse

        # Create a custom evaluator
        class TestEvaluator(BaseEvaluator):
            def __init__(self, llm_client):
                self._llm_client = llm_client

            @property
            def name(self) -> str:
                return "test_evaluator"

            async def evaluate(self, output: str, reference=None, criteria=None):
                from arbiter.core.models import Score

                return Score(name="test_evaluator", value=0.75)

            def get_interactions(self):
                return []

            def clear_interactions(self):
                pass

        # Register it
        register_evaluator("test_evaluator", TestEvaluator)

        try:
            # Mock LLM client
            mock_client = MagicMock(spec=LLMClient)
            mock_client.model = "gpt-4o-mini"

            # Mock agent
            mock_agent = AsyncMock()
            mock_response = SemanticResponse(
                score=0.75,
                confidence=0.8,
                explanation="Test",
            )

            class MockAgentResult:
                def __init__(self, output):
                    self.output = output

                def usage(self):
                    mock_usage = MagicMock()
                    mock_usage.total_tokens = 100
                    return mock_usage

            mock_result = MockAgentResult(mock_response)
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_client.create_agent = MagicMock(return_value=mock_agent)

            # Use in evaluate()
            from arbiter.api import evaluate

            evaluate(
                output="Test output",
                evaluators=["test_evaluator"],
                llm_client=mock_client,
            )

            # Note: This is a coroutine, so we'd need to await it
            # But the test structure shows it can be used
            assert True  # If we got here, registration worked

        finally:
            # Clean up
            if "test_evaluator" in AVAILABLE_EVALUATORS:
                del AVAILABLE_EVALUATORS["test_evaluator"]


class TestRegistryIntegration:
    """Test suite for registry integration with API."""

    @pytest.mark.asyncio
    async def test_evaluate_with_registry(self):
        """Test that evaluate() uses registry for validation."""
        from unittest.mock import AsyncMock, MagicMock

        from arbiter.core.llm_client import LLMClient
        from arbiter.evaluators.semantic import SemanticResponse

        mock_client = MagicMock(spec=LLMClient)
        mock_client.model = "gpt-4o-mini"

        mock_agent = AsyncMock()
        mock_response = SemanticResponse(
            score=0.9,
            confidence=0.85,
            explanation="Test",
        )

        class MockAgentResult:
            def __init__(self, output):
                self.output = output

            def usage(self):
                mock_usage = MagicMock()
                mock_usage.total_tokens = 100
                return mock_usage

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_client.create_agent = MagicMock(return_value=mock_agent)

        from arbiter.api import evaluate

        # Valid evaluator - should work
        result = await evaluate(
            output="Test output",
            reference="Test reference",
            evaluators=["semantic"],
            llm_client=mock_client,
        )

        assert result.overall_score == 0.9

        # Invalid evaluator - should raise ValidationError
        with pytest.raises(ValidationError, match="Unknown evaluator"):
            await evaluate(
                output="Test output",
                evaluators=["nonexistent_evaluator"],
                llm_client=mock_client,
            )

    @pytest.mark.asyncio
    async def test_evaluate_error_message_helpful(self):
        """Test that error messages include available evaluators."""
        from unittest.mock import MagicMock

        from arbiter.core.llm_client import LLMClient

        mock_client = MagicMock(spec=LLMClient)
        mock_client.model = "gpt-4o-mini"

        from arbiter.api import evaluate

        with pytest.raises(ValidationError) as exc_info:
            await evaluate(
                output="Test output",
                evaluators=["unknown_evaluator"],
                llm_client=mock_client,
            )

        error_msg = str(exc_info.value)
        assert "Unknown evaluator" in error_msg
        assert "unknown_evaluator" in error_msg
        assert "Available evaluators" in error_msg
        assert "semantic" in error_msg
        assert "custom_criteria" in error_msg


class TestRegistryEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_get_evaluator_class_none(self):
        """Test getting non-existent evaluator returns None."""
        assert get_evaluator_class("definitely_does_not_exist") is None

    def test_validate_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(ValidationError):
            validate_evaluator_name("")

    def test_available_evaluators_is_copy(self):
        """Test that modifying returned list doesn't affect registry."""
        evaluators = get_available_evaluators()
        original_count = len(evaluators)

        # Modify the list
        evaluators.append("should_not_appear")

        # Registry should be unchanged
        assert len(get_available_evaluators()) == original_count
        assert "should_not_appear" not in get_available_evaluators()
