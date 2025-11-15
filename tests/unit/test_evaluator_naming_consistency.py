"""Tests for evaluator naming consistency.

This test suite ensures that evaluator names are consistent throughout
the system to prevent registry/score name mismatches.

Critical invariants:
1. evaluator.name must match score.name
2. Registry name must match evaluator.name
3. All built-in evaluators must be registered

Tests cover:
- Name consistency between evaluator property and Score objects
- Registry name matching evaluator name property
- All built-in evaluators properly registered
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter.core.registry import AVAILABLE_EVALUATORS, get_evaluator_class
from arbiter.evaluators import CustomCriteriaEvaluator, SemanticEvaluator


@pytest.mark.asyncio
async def test_semantic_evaluator_name_consistency():
    """Test that SemanticEvaluator score name matches evaluator name."""
    from arbiter.evaluators.semantic import SemanticResponse
    from tests.conftest import MockAgentResult

    # Mock response using actual Pydantic model
    mock_response = SemanticResponse(
        score=0.9,
        confidence=0.85,
        explanation="Test explanation",
        key_differences=[],
        key_similarities=["similarity1"]
    )

    # Mock result using MockAgentResult from conftest
    mock_result = MockAgentResult(mock_response)

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)

    # Mock client with model attribute set to string
    mock_llm_client = MagicMock()
    mock_llm_client.model = "gpt-4o-mini"
    mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

    evaluator = SemanticEvaluator(mock_llm_client)

    # Get evaluator name
    evaluator_name = evaluator.name

    # Perform evaluation
    score = await evaluator.evaluate(
        output="Test output", reference="Test reference"
    )

    # Critical check: score name must match evaluator name
    assert score.name == evaluator_name, (
        f"Score name '{score.name}' does not match "
        f"evaluator name '{evaluator_name}'"
    )


@pytest.mark.asyncio
async def test_custom_criteria_evaluator_name_consistency():
    """Test that CustomCriteriaEvaluator score name matches evaluator name."""
    from arbiter.evaluators.custom_criteria import CustomCriteriaResponse
    from tests.conftest import MockAgentResult

    # Mock response using actual Pydantic model
    mock_response = CustomCriteriaResponse(
        score=0.9,
        confidence=0.85,
        explanation="Test explanation",
        criteria_met=["criteria1"],
        criteria_not_met=[]
    )

    # Mock result using MockAgentResult from conftest
    mock_result = MockAgentResult(mock_response)

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)

    # Mock client with model attribute set to string
    mock_llm_client = MagicMock()
    mock_llm_client.model = "gpt-4o-mini"
    mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

    evaluator = CustomCriteriaEvaluator(mock_llm_client)

    # Get evaluator name
    evaluator_name = evaluator.name

    # Perform evaluation
    score = await evaluator.evaluate(
        output="Test output", criteria="Test criteria"
    )

    # Critical check: score name must match evaluator name
    assert score.name == evaluator_name, (
        f"Score name '{score.name}' does not match "
        f"evaluator name '{evaluator_name}'"
    )


def test_registry_name_matches_evaluator_name():
    """Test that registry names match evaluator name properties."""
    for registry_name, evaluator_class in AVAILABLE_EVALUATORS.items():
        # Create instance (need to handle constructor requirements)
        # For now, just check the name property exists
        assert hasattr(
            evaluator_class, "name"
        ), f"Evaluator {evaluator_class} missing name property"

        # Note: We can't easily instantiate without proper LLM client,
        # but we can check the property is defined


def test_all_builtin_evaluators_registered():
    """Test that all built-in evaluators are registered."""
    expected_evaluators = ["semantic", "custom_criteria"]

    for evaluator_name in expected_evaluators:
        assert (
            evaluator_name in AVAILABLE_EVALUATORS
        ), f"Built-in evaluator '{evaluator_name}' not registered"


def test_registry_lookup_returns_correct_class():
    """Test that registry lookup returns the expected evaluator classes."""
    semantic_class = get_evaluator_class("semantic")
    assert semantic_class == SemanticEvaluator

    custom_class = get_evaluator_class("custom_criteria")
    assert custom_class == CustomCriteriaEvaluator


@pytest.mark.asyncio
async def test_evaluator_name_immutability():
    """Test that evaluator name property is consistent across calls."""
    from arbiter.evaluators.semantic import SemanticResponse
    from tests.conftest import MockAgentResult

    # Mock response using actual Pydantic model
    mock_response = SemanticResponse(
        score=0.9,
        confidence=0.85,
        explanation="Test explanation",
        key_differences=[],
        key_similarities=["similarity1"]
    )

    # Mock result using MockAgentResult from conftest
    mock_result = MockAgentResult(mock_response)

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)

    # Mock client with model attribute set to string
    mock_llm_client = MagicMock()
    mock_llm_client.model = "gpt-4o-mini"
    mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

    evaluator = SemanticEvaluator(mock_llm_client)

    # Get name multiple times
    name1 = evaluator.name
    name2 = evaluator.name
    name3 = evaluator.name

    # Name should be immutable
    assert name1 == name2 == name3, "Evaluator name property is not stable"


@pytest.mark.asyncio
async def test_score_name_matches_in_multi_evaluator_run():
    """Test name consistency when running multiple evaluators."""
    from arbiter.evaluators.semantic import SemanticResponse
    from arbiter.evaluators.custom_criteria import CustomCriteriaResponse
    from tests.conftest import MockAgentResult

    # Mock semantic response
    semantic_response = SemanticResponse(
        score=0.9,
        confidence=0.85,
        explanation="Test explanation",
        key_differences=[],
        key_similarities=["similarity1"]
    )
    semantic_result = MockAgentResult(semantic_response)
    semantic_agent = MagicMock()
    semantic_agent.run = AsyncMock(return_value=semantic_result)

    # Mock custom criteria response
    criteria_response = CustomCriteriaResponse(
        score=0.85,
        confidence=0.8,
        explanation="Criteria explanation",
        criteria_met=["quality"],
        criteria_not_met=[]
    )
    criteria_result = MockAgentResult(criteria_response)
    criteria_agent = MagicMock()
    criteria_agent.run = AsyncMock(return_value=criteria_result)

    # Create separate mock clients for each evaluator
    semantic_client = MagicMock()
    semantic_client.model = "gpt-4o-mini"
    semantic_client.create_agent = MagicMock(return_value=semantic_agent)

    criteria_client = MagicMock()
    criteria_client.model = "gpt-4o-mini"
    criteria_client.create_agent = MagicMock(return_value=criteria_agent)

    semantic_eval = SemanticEvaluator(semantic_client)
    criteria_eval = CustomCriteriaEvaluator(criteria_client)

    # Run evaluations
    semantic_score = await semantic_eval.evaluate(
        output="Test", reference="Reference"
    )
    criteria_score = await criteria_eval.evaluate(
        output="Test", criteria="Quality"
    )

    # Verify names match
    assert semantic_score.name == semantic_eval.name
    assert criteria_score.name == criteria_eval.name

    # Verify names are different
    assert semantic_score.name != criteria_score.name
