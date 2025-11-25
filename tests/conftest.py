"""Shared test fixtures and utilities for Arbiter tests.

This module provides common fixtures used across all test files to reduce
duplication and improve maintainability.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter_ai.core.llm_client import LLMClient


class MockAgentResult:
    """Mock PydanticAI agent result for testing.

    This mock simulates the result object returned by PydanticAI agents,
    providing both the output and usage statistics.
    """

    def __init__(self, output: object):
        """Initialize mock result with output.

        Args:
            output: The response object to return (e.g., SemanticResponse)
        """
        self.output = output

    def usage(self):
        """Mock usage information for token tracking.

        Returns:
            Mock usage object with total_tokens attribute
        """
        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        return mock_usage


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing.

    Returns:
        Mock LLMClient with standard test configuration
    """
    client = MagicMock(spec=LLMClient)
    client.model = "gpt-4o-mini"
    client.temperature = 0.0
    return client


@pytest.fixture
def mock_agent():
    """Create a mock PydanticAI agent for testing.

    Returns:
        AsyncMock agent that can be configured with responses
    """
    agent = AsyncMock()
    return agent
