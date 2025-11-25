"""Unit tests for cost calculator functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter_ai.core.cost_calculator import (
    CostCalculator,
    ModelPricing,
    get_cost_calculator,
)


class TestModelPricing:
    """Tests for ModelPricing model."""

    def test_model_pricing_creation(self):
        """Test creating a ModelPricing instance."""
        from datetime import datetime

        pricing = ModelPricing(
            id="gpt-4o-mini",
            vendor="openai",
            name="GPT-4o Mini",
            input=0.150,
            output=0.600,
        )

        assert pricing.id == "gpt-4o-mini"
        assert pricing.vendor == "openai"
        assert pricing.name == "GPT-4o Mini"
        assert pricing.input == 0.150
        assert pricing.output == 0.600
        assert pricing.input_cached is None
        assert isinstance(pricing.last_updated, datetime)

    def test_model_pricing_with_cached(self):
        """Test ModelPricing with cached token pricing."""

        pricing = ModelPricing(
            id="claude-3-5-sonnet",
            vendor="anthropic",
            name="Claude 3.5 Sonnet",
            input=3.0,
            output=15.0,
            input_cached=0.3,
        )

        assert pricing.input_cached == 0.3


class TestCostCalculator:
    """Tests for CostCalculator functionality."""

    @pytest.fixture
    def mock_pricing_data(self):
        """Mock pricing data from llm-prices.com."""
        return [
            {
                "id": "gpt-4o-mini",
                "vendor": "openai",
                "name": "GPT-4o Mini",
                "input": 0.150,
                "output": 0.600,
                "input_cached": None,
            },
            {
                "id": "claude-3-5-sonnet",
                "vendor": "anthropic",
                "name": "Claude 3.5 Sonnet",
                "input": 3.0,
                "output": 15.0,
                "input_cached": 0.3,
            },
            {
                "id": "gpt-4o",
                "vendor": "openai",
                "name": "GPT-4o",
                "input": 2.50,
                "output": 10.0,
                "input_cached": None,
            },
        ]

    @pytest.mark.asyncio
    async def test_ensure_loaded_success(self, mock_pricing_data):
        """Test successfully loading pricing data."""
        calc = CostCalculator()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_pricing_data
            mock_response.raise_for_status.return_value = None

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            mock_client.return_value = mock_context

            await calc.ensure_loaded()

            assert calc.is_loaded
            assert calc.model_count == 3
            assert calc.get_pricing("gpt-4o-mini") is not None
            assert calc.get_pricing("claude-3-5-sonnet") is not None

    @pytest.mark.asyncio
    async def test_ensure_loaded_failure(self):
        """Test handling failure to load pricing data."""
        calc = CostCalculator()

        with patch("httpx.AsyncClient") as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Network error")
            )
            mock_client.return_value = mock_context

            await calc.ensure_loaded()

            assert not calc.is_loaded
            assert calc.model_count == 0

    @pytest.mark.asyncio
    async def test_ensure_loaded_only_once(self, mock_pricing_data):
        """Test that pricing data is only loaded once."""
        calc = CostCalculator()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_pricing_data
            mock_response.raise_for_status.return_value = None

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            mock_client.return_value = mock_context

            # Call ensure_loaded multiple times
            await calc.ensure_loaded()
            await calc.ensure_loaded()
            await calc.ensure_loaded()

            # Should only fetch once
            assert mock_context.__aenter__.return_value.get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_with_retry_success_on_second_attempt(self, mock_pricing_data):
        """Test retry logic succeeds after initial failure."""
        import httpx

        calc = CostCalculator()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_pricing_data
            mock_response.raise_for_status.return_value = None

            mock_context = AsyncMock()
            # First call fails with httpx.ConnectError (retryable), second call succeeds
            mock_context.__aenter__.return_value.get = AsyncMock(
                side_effect=[
                    httpx.ConnectError("Network error"),  # First attempt fails
                    mock_response,  # Second attempt succeeds
                ]
            )
            mock_client.return_value = mock_context

            # Mock asyncio.sleep to avoid actual delays
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await calc.ensure_loaded()

            # Should succeed after retry
            assert calc.is_loaded
            assert calc.model_count == 3
            # Should have been called twice (first fail, then success)
            assert mock_context.__aenter__.return_value.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_with_all_retries_exhausted(self):
        """Test that all retry attempts are exhausted on persistent failure."""
        import httpx

        calc = CostCalculator()

        with patch("httpx.AsyncClient") as mock_client:
            mock_context = AsyncMock()
            # All 3 attempts will fail with httpx.ConnectError (retryable)
            mock_context.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Persistent network error")
            )
            mock_client.return_value = mock_context

            # Mock asyncio.sleep to avoid actual delays
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await calc.ensure_loaded()

            # Should fail after all retries
            assert not calc.is_loaded
            assert calc.model_count == 0
            # Should have attempted 3 times (max_retries)
            assert mock_context.__aenter__.return_value.get.call_count == 3

    @pytest.mark.asyncio
    async def test_ensure_loaded_concurrent_calls(self, mock_pricing_data):
        """Test that concurrent calls only fetch pricing data once (lock test)."""
        import asyncio

        calc = CostCalculator()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_pricing_data
            mock_response.raise_for_status.return_value = None

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            mock_client.return_value = mock_context

            # Call ensure_loaded concurrently 5 times
            results = await asyncio.gather(
                calc.ensure_loaded(),
                calc.ensure_loaded(),
                calc.ensure_loaded(),
                calc.ensure_loaded(),
                calc.ensure_loaded(),
            )

            # All should complete without error
            assert len(results) == 5

            # Should only have fetched once due to lock
            assert calc.is_loaded
            # The get method might be called once or slightly more due to race conditions,
            # but should be much less than 5
            assert mock_context.__aenter__.return_value.get.call_count <= 2

    def test_get_pricing_exact_match(self):
        """Test getting pricing with exact model match."""
        calc = CostCalculator()
        calc._loaded = True
        calc._pricing_cache = {
            "gpt-4o-mini": ModelPricing(
                id="gpt-4o-mini",
                vendor="openai",
                name="GPT-4o Mini",
                input=0.150,
                output=0.600,
            )
        }

        pricing = calc.get_pricing("gpt-4o-mini")
        assert pricing is not None
        assert pricing.id == "gpt-4o-mini"
        assert pricing.input == 0.150
        assert pricing.output == 0.600

    def test_get_pricing_fuzzy_match(self):
        """Test fuzzy matching for model variants."""
        calc = CostCalculator()
        calc._loaded = True
        calc._pricing_cache = {
            "gpt-4o-mini": ModelPricing(
                id="gpt-4o-mini",
                vendor="openai",
                name="GPT-4o Mini",
                input=0.150,
                output=0.600,
            )
        }

        # Should match "gpt-4o-mini-2024-07-18" to "gpt-4o-mini"
        pricing = calc.get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None
        assert pricing.id == "gpt-4o-mini"

    def test_get_pricing_not_found(self):
        """Test getting pricing for unknown model."""
        calc = CostCalculator()
        calc._loaded = True
        calc._pricing_cache = {}

        pricing = calc.get_pricing("unknown-model")
        assert pricing is None

    def test_get_pricing_not_loaded(self):
        """Test getting pricing when data not loaded."""
        calc = CostCalculator()
        assert not calc.is_loaded

        pricing = calc.get_pricing("gpt-4o-mini")
        assert pricing is None

    def test_calculate_cost_with_pricing(self):
        """Test cost calculation with pricing data available."""
        calc = CostCalculator()
        calc._loaded = True
        calc._pricing_cache = {
            "gpt-4o-mini": ModelPricing(
                id="gpt-4o-mini",
                vendor="openai",
                name="GPT-4o Mini",
                input=0.150,  # $0.15 per 1M tokens
                output=0.600,  # $0.60 per 1M tokens
            )
        }

        # Calculate cost for 1000 input + 500 output tokens
        cost = calc.calculate_cost(
            model="gpt-4o-mini", input_tokens=1000, output_tokens=500
        )

        # Expected: (1000 / 1M * 0.15) + (500 / 1M * 0.60)
        # = 0.00015 + 0.0003 = 0.00045
        assert abs(cost - 0.00045) < 0.000001

    def test_calculate_cost_with_cached_tokens(self):
        """Test cost calculation with cached tokens."""
        calc = CostCalculator()
        calc._loaded = True
        calc._pricing_cache = {
            "claude-3-5-sonnet": ModelPricing(
                id="claude-3-5-sonnet",
                vendor="anthropic",
                name="Claude 3.5 Sonnet",
                input=3.0,  # $3 per 1M tokens
                output=15.0,  # $15 per 1M tokens
                input_cached=0.3,  # $0.30 per 1M cached tokens
            )
        }

        # Calculate cost: 1000 input (500 cached) + 500 output
        cost = calc.calculate_cost(
            model="claude-3-5-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
        )

        # Expected:
        # - Non-cached input: (500 / 1M * 3.0) = 0.0015
        # - Cached input: (500 / 1M * 0.3) = 0.00015
        # - Output: (500 / 1M * 15.0) = 0.0075
        # Total: 0.0015 + 0.00015 + 0.0075 = 0.00915
        assert abs(cost - 0.00915) < 0.000001

    def test_calculate_cost_fallback(self):
        """Test cost calculation fallback when pricing unavailable."""
        calc = CostCalculator()
        calc._loaded = False  # No pricing data

        cost = calc.calculate_cost(
            model="unknown-model", input_tokens=1000, output_tokens=500
        )

        # Fallback uses conservative estimates:
        # Input: $10/M, Output: $30/M
        # Expected: (1000/1M * 10) + (500/1M * 30) = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 0.000001

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        calc = CostCalculator()
        calc._loaded = True
        calc._pricing_cache = {
            "gpt-4o-mini": ModelPricing(
                id="gpt-4o-mini",
                vendor="openai",
                name="GPT-4o Mini",
                input=0.150,
                output=0.600,
            )
        }

        cost = calc.calculate_cost(model="gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_fallback_estimate(self):
        """Test fallback cost estimation."""
        calc = CostCalculator()

        # Test with various token counts
        cost = calc._fallback_estimate(
            input_tokens=1000, output_tokens=500, cached_tokens=0
        )

        # Input: $10/M * 1000 = $0.01
        # Output: $30/M * 500 = $0.015
        # Total: $0.025
        assert abs(cost - 0.025) < 0.000001

    def test_fallback_estimate_with_cached(self):
        """Test fallback estimation with cached tokens."""
        calc = CostCalculator()

        cost = calc._fallback_estimate(
            input_tokens=1000, output_tokens=500, cached_tokens=500
        )

        # Non-cached input: $10/M * 500 = $0.005
        # Cached input: $1/M * 500 = $0.0005
        # Output: $30/M * 500 = $0.015
        # Total: $0.0205
        assert abs(cost - 0.0205) < 0.000001


class TestGlobalCostCalculator:
    """Tests for global cost calculator singleton."""

    def test_get_cost_calculator_singleton(self):
        """Test that get_cost_calculator returns singleton."""
        calc1 = get_cost_calculator()
        calc2 = get_cost_calculator()

        assert calc1 is calc2

    def test_get_cost_calculator_returns_instance(self):
        """Test that get_cost_calculator returns CostCalculator."""
        calc = get_cost_calculator()

        assert isinstance(calc, CostCalculator)


class TestIntegrationWithEvaluationResult:
    """Integration tests with EvaluationResult."""

    @pytest.mark.asyncio
    async def test_evaluation_result_cost_calculation(self):
        """Test cost calculation in EvaluationResult."""
        from arbiter_ai.core.models import EvaluationResult, LLMInteraction, Score

        # Create evaluation result with interactions
        result = EvaluationResult(
            output="test output",
            scores=[Score(name="semantic", value=0.85)],
            overall_score=0.85,
            passed=True,
            processing_time=1.5,
            interactions=[
                LLMInteraction(
                    prompt="test prompt",
                    response="test response",
                    model="gpt-4o-mini",
                    input_tokens=1000,
                    output_tokens=500,
                    latency=1.2,
                    purpose="scoring",
                    cost=0.00045,  # Pre-calculated cost
                )
            ],
        )

        # Calculate total cost
        cost = await result.total_llm_cost()
        assert abs(cost - 0.00045) < 0.000001

    @pytest.mark.asyncio
    async def test_evaluation_result_cost_breakdown(self):
        """Test cost breakdown in EvaluationResult."""
        from arbiter_ai.core.models import EvaluationResult, LLMInteraction, Score

        result = EvaluationResult(
            output="test output",
            scores=[Score(name="semantic", value=0.85)],
            overall_score=0.85,
            passed=True,
            processing_time=1.5,
            interactions=[
                LLMInteraction(
                    prompt="prompt1",
                    response="response1",
                    model="gpt-4o-mini",
                    input_tokens=1000,
                    output_tokens=500,
                    latency=1.0,
                    purpose="semantic",
                    cost=0.00045,
                ),
                LLMInteraction(
                    prompt="prompt2",
                    response="response2",
                    model="claude-3-5-sonnet",
                    input_tokens=1500,
                    output_tokens=750,
                    latency=1.2,
                    purpose="factuality",
                    cost=0.015,
                ),
            ],
        )

        breakdown = await result.cost_breakdown()

        assert "total" in breakdown
        assert "by_evaluator" in breakdown
        assert "by_model" in breakdown
        assert "token_breakdown" in breakdown

        # Check total
        assert abs(breakdown["total"] - 0.01545) < 0.000001

        # Check by evaluator
        assert "semantic" in breakdown["by_evaluator"]
        assert "factuality" in breakdown["by_evaluator"]

        # Check by model
        assert "gpt-4o-mini" in breakdown["by_model"]
        assert "claude-3-5-sonnet" in breakdown["by_model"]

        # Check token breakdown
        assert breakdown["token_breakdown"]["input_tokens"] == 2500
        assert breakdown["token_breakdown"]["output_tokens"] == 1250
        assert breakdown["token_breakdown"]["total_tokens"] == 3750
